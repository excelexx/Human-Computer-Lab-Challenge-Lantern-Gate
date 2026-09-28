[CmdletBinding()]
param(
    [string]$ArtifactsRoot = $env:CHECKIN_HOME,
    [string]$PythonPath,
    [ValidateRange(1, 65535)]
    [int]$Port = 7860
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ArtifactsRoot) { $ArtifactsRoot = Join-Path $repoRoot '.artifacts' }
$ArtifactsRoot = [System.IO.Path]::GetFullPath($ArtifactsRoot)
if (-not $PythonPath) { $PythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Python environment not found. Follow the README setup or supply -PythonPath.' }
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
foreach ($stage in @('vision','text','fusion')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ArtifactsRoot ('checkpoints\' + $stage + '.pt')) -PathType Leaf)) {
        throw ('Missing trained ' + $stage + ' head. Run the training steps first.')
    }
}
$logs = Join-Path $ArtifactsRoot 'logs'
$runtime = Join-Path $ArtifactsRoot 'runtime'
$recordPath = Join-Path $runtime 'app.json'

function Get-VerifiedAppProcess($Identity) {
    try {
        $candidate = Get-Process -Id ([int]$Identity.pid) -ErrorAction Stop
        if (-not [string]::Equals($candidate.Path, [string]$Identity.binary, [System.StringComparison]::OrdinalIgnoreCase)) { return $null }
        if ($candidate.StartTime.ToUniversalTime().Ticks -ne ([DateTime]$Identity.started_at).ToUniversalTime().Ticks) { return $null }
        return $candidate
    } catch { return $null }
}

function New-AppIdentity($Process, [int]$Depth = 0) {
    # In Windows PowerShell 5, Start-Process -PassThru can briefly expose a null
    # Path. Requery the same process identity before writing its ownership file.
    $expectedStart = $Process.StartTime.ToUniversalTime().Ticks
    $expectedBinary = if ($Depth -eq 0) { $PythonPath } else { $Process.Path }
    $fresh = $null
    for ($identityAttempt = 0; $identityAttempt -lt 20; $identityAttempt++) {
        $candidate = Get-Process -Id $Process.Id -ErrorAction Stop
        if ($candidate.StartTime.ToUniversalTime().Ticks -ne $expectedStart) { throw 'App PID changed before its ownership could be recorded.' }
        if ([string]::Equals($candidate.Path, $expectedBinary, [System.StringComparison]::OrdinalIgnoreCase)) { $fresh = $candidate; break }
        Start-Sleep -Milliseconds 50
    }
    if ($null -eq $fresh) { throw 'Could not verify the newly started app executable.' }
    return [pscustomobject]@{
        pid = $fresh.Id; binary = $fresh.Path
        started_at = $fresh.StartTime.ToUniversalTime().ToString('o')
        depth = $Depth
    }
}

function Get-VerifiedAppFamily($Identities) {
    $verified = @{}
    foreach ($identity in @($Identities)) {
        if ($null -ne (Get-VerifiedAppProcess $identity)) { $verified[[int]$identity.pid] = $identity }
    }
    if ($verified.Count -eq 0) { return @() }
    # Windows Store Python uses a venv launcher plus the actual Python worker.
    # Only Python descendants running this exact app/home/port can be adopted.
    $snapshot = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($child in $snapshot) {
            $childId = [int]$child.ProcessId
            $parentId = [int]$child.ParentProcessId
            if ($verified.ContainsKey($childId) -or -not $verified.ContainsKey($parentId)) { continue }
            if ([string]$child.Name -notmatch '(?i)^python(?:\d+(?:\.\d+)?)?w?\.exe$') { continue }
            $command = [string]$child.CommandLine
            if ($command -notmatch '(?i)(?:^|\s)-m\s+"?checkin\.app"?(?:\s|$)') { continue }
            if ($command -notmatch ('(?:^|\s)--port(?:=|\s+)' + $Port + '(?:\s|$)')) { continue }
            $homeMatch = [regex]::Match($command, '(?i)(?:^|\s)--home\s+(?:"([^"]+)"|(\S+))')
            if (-not $homeMatch.Success) { continue }
            $commandHome = if ($homeMatch.Groups[1].Success) { $homeMatch.Groups[1].Value } else { $homeMatch.Groups[2].Value }
            if (-not [string]::Equals($commandHome.TrimEnd('\'), $ArtifactsRoot.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            $parent = Get-VerifiedAppProcess $verified[$parentId]
            if ($null -eq $parent) { continue }
            try {
                $process = Get-Process -Id $childId -ErrorAction Stop
                if ($process.StartTime.ToUniversalTime().Ticks -lt $parent.StartTime.ToUniversalTime().Ticks) { continue }
                if (-not [string]::Equals($process.Path, [string]$child.ExecutablePath, [System.StringComparison]::OrdinalIgnoreCase)) { continue }
                # Recheck the parent after querying the child to catch PID reuse.
                if ($null -eq (Get-VerifiedAppProcess $verified[$parentId])) { continue }
                $verified[$childId] = New-AppIdentity $process ([int]$verified[$parentId].depth + 1)
                $changed = $true
            } catch { continue }
        }
    }
    return @($verified.Values | Where-Object { $null -ne (Get-VerifiedAppProcess $_) })
}

function Save-AppRecord($Identities, [int]$PrimaryId) {
    $primary = @($Identities | Where-Object { [int]$_.pid -eq $PrimaryId })[0]
    if ($null -eq $primary) { throw 'Cannot record ownership of an unverified app process.' }
    @{
        schema_version = 2; pid = $primary.pid; binary = $primary.binary
        started_at = $primary.started_at; launcher_binary = $PythonPath
        home = $ArtifactsRoot; port = $Port; processes = @($Identities)
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath -Encoding utf8
}

$ownedIdentities = @()
if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
    $verifiedRecord = $false
    try {
        $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
        $recordLauncher = if ($record.launcher_binary) { [string]$record.launcher_binary } else { [string]$record.binary }
        $sameLauncher = [string]::Equals($recordLauncher, $PythonPath, [System.StringComparison]::OrdinalIgnoreCase)
        $sameHome = [string]::Equals([string]$record.home, $ArtifactsRoot, [System.StringComparison]::OrdinalIgnoreCase)
        if ($sameLauncher -and $sameHome) {
            $saved = if ($record.processes) { @($record.processes) } else { @($record) }
            $live = @($saved | Where-Object { $null -ne (Get-VerifiedAppProcess $_) })
            if ($live.Count -gt 0) {
                $verifiedRecord = $true
                if ([int]$record.port -ne $Port) { throw 'This artifacts directory already has an app running on another port. Stop it before changing ports.' }
                $ownedIdentities = @(Get-VerifiedAppFamily $live)
            }
        }
    } catch {
        if ($verifiedRecord) { throw }
        # A stale identity never authorizes adopting or stopping a process.
    }
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    if ([int]$listener.OwningProcess -notin @($ownedIdentities | ForEach-Object { [int]$_.pid })) {
        throw "Port $Port is already in use by a process this project cannot verify as its own. No process was changed."
    }
}
& (Join-Path $PSScriptRoot 'start-generator.ps1') -ArtifactsRoot $ArtifactsRoot
$env:CHECKIN_HOME = $ArtifactsRoot
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
New-Item -ItemType Directory -Force -Path $logs,$runtime | Out-Null
$reused = $ownedIdentities.Count -gt 0
if (-not $reused) {
    # A trailing backslash must be doubled inside Windows command-line quotes.
    $quotedHome = '"' + ($ArtifactsRoot -replace '(\\+)$', '$1$1') + '"'
    $argsForApp = @('-m','checkin.app','--home',$quotedHome,'--host','127.0.0.1','--port',"$Port")
    $launcher = Start-Process -FilePath $PythonPath -ArgumentList $argsForApp -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'app.stdout.log') -RedirectStandardError (Join-Path $logs 'app.stderr.log')
    $ownedIdentities = @(New-AppIdentity $launcher)
    Save-AppRecord $ownedIdentities $launcher.Id
}
$ready = $false
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    $ownedIdentities = @(Get-VerifiedAppFamily $ownedIdentities)
    if ($ownedIdentities.Count -eq 0) { throw ('App exited. Inspect ' + (Join-Path $logs 'app.stderr.log')) }
    Save-AppRecord $ownedIdentities ([int]$ownedIdentities[0].pid)
    $currentListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    $ownedIds = @($ownedIdentities | ForEach-Object { [int]$_.pid })
    if ($currentListeners.Count -gt 0 -and @($currentListeners | Where-Object { [int]$_.OwningProcess -notin $ownedIds }).Count -eq 0) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                # Save the real listener, not the venv launcher, as the primary.
                Save-AppRecord $ownedIdentities ([int]$currentListeners[0].OwningProcess)
                $ready = $true
                break
            }
        } catch { }
    }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw 'App did not become ready. Inspect its local logs.' }
# A new app opens its own browser; only a reused process needs a new browser tab.
if ($reused) { Start-Process "http://127.0.0.1:$Port/" -WindowStyle Hidden }
Write-Output "Check-in ready at http://127.0.0.1:$Port/"
