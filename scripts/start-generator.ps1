[CmdletBinding()]
param(
    [Alias('Home')]
    [string]$ArtifactsRoot,
    [ValidateRange(5, 600)]
    [int]$StartupTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ArtifactsRoot) {
    $ArtifactsRoot = if ($env:CHECKIN_HOME) { $env:CHECKIN_HOME } else { Join-Path $repoRoot '.artifacts' }
}
$ArtifactsRoot = [System.IO.Path]::GetFullPath($ArtifactsRoot)
$modelPath = Join-Path $ArtifactsRoot 'models/qwen/Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf'
$binaryPath = Join-Path $ArtifactsRoot 'vendor/llama/llama-server.exe'
$runtimePath = Join-Path $ArtifactsRoot 'runtime'
$pidPath = Join-Path $runtimePath 'generator.json'
$stdoutPath = Join-Path $runtimePath 'generator.stdout.log'
$stderrPath = Join-Path $runtimePath 'generator.stderr.log'
$healthUrl = 'http://127.0.0.1:8081/health'

foreach ($requiredPath in @($modelPath, $binaryPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required generator artifact is missing: $requiredPath. Run the project setup/download command first."
    }
}
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null

function Test-GeneratorHealth {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
        return $health.status -eq 'ok'
    } catch {
        return $false
    }
}

$ownedProcess = $null
if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
    try {
        $record = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json
        $candidate = Get-Process -Id ([int]$record.pid) -ErrorAction Stop
        $samePath = [string]::Equals($candidate.Path, $binaryPath, [System.StringComparison]::OrdinalIgnoreCase)
        $sameStart = $candidate.StartTime.ToUniversalTime().Ticks -eq ([DateTime]$record.started_at).ToUniversalTime().Ticks
        if ($samePath -and $sameStart) {
            $ownedProcess = $candidate
        }
    } catch {
        # A stale PID record is not permission to stop a different process.
    }
}

$listeners = @(Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    if ($null -eq $ownedProcess -or [int]$listener.OwningProcess -ne $ownedProcess.Id) {
        throw 'Port 8081 is already in use by a process this project cannot verify as its own. No process was changed.'
    }
}

if (Test-GeneratorHealth) {
    if ($null -eq $ownedProcess) {
        throw 'Port 8081 already has a healthy service that this project cannot verify as its own. No process was changed.'
    }
    Write-Host "Local generator is already ready at http://127.0.0.1:8081 (PID $($ownedProcess.Id))."
    return
}

if ($null -eq $ownedProcess) {
    # Start-Process joins ArgumentList on Windows, so quote the model pathname.
    $arguments = @(
        '--model', ('"' + $modelPath + '"'),
        '--alias', 'checkin-qwen',
        '--gpu-layers', 'all', '--flash-attn', 'on',
        '--ctx-size', '4096', '--parallel', '1', '--jinja',
        '--host', '127.0.0.1', '--port', '8081',
        '--n-predict', '96'
    )
    $ownedProcess = Start-Process -FilePath $binaryPath -ArgumentList $arguments `
        -WorkingDirectory (Split-Path -Parent $binaryPath) -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $record = [ordered]@{
        pid = $ownedProcess.Id
        started_at = $ownedProcess.StartTime.ToUniversalTime().ToString('o')
        binary = $binaryPath
        model = $modelPath
        base_url = 'http://127.0.0.1:8081'
        stdout = $stdoutPath
        stderr = $stderrPath
    }
    $record | ConvertTo-Json | Set-Content -LiteralPath $pidPath -Encoding UTF8
}

$deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
while ([DateTime]::UtcNow -lt $deadline) {
    $ownedProcess.Refresh()
    if ($ownedProcess.HasExited) {
        throw "The local generator exited. See $stderrPath. No other process was stopped."
    }
    $currentListeners = @(Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue)
    $ownsListener = $currentListeners.Count -gt 0 -and @($currentListeners | Where-Object { [int]$_.OwningProcess -ne $ownedProcess.Id }).Count -eq 0
    if ($ownsListener -and (Test-GeneratorHealth)) {
        Write-Host "Local generator ready at http://127.0.0.1:8081 (PID $($ownedProcess.Id))."
        return
    }
    Start-Sleep -Milliseconds 500
}
throw "Generator readiness timed out; its recorded process was left running. See $stderrPath and $pidPath."
