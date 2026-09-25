param([string]$ArtifactsRoot = $env:CHECKIN_HOME)
$ErrorActionPreference = 'Stop'
if (-not $ArtifactsRoot) { $ArtifactsRoot = Join-Path (Split-Path -Parent $PSScriptRoot) '.artifacts' }
$ArtifactsRoot = [System.IO.Path]::GetFullPath($ArtifactsRoot)

function Get-VerifiedOwnedProcess($Identity) {
    $process = Get-Process -Id ([int]$Identity.pid) -ErrorAction SilentlyContinue
    if (-not $process -or $process.HasExited) { return $null }
    try {
        $observedPath = $process.Path
        $observedStart = $process.StartTime.ToUniversalTime()
    } catch {
        if ($process.HasExited) { return $null }
        throw
    }
    # The venv launcher can finish between Get-Process and reading its identity.
    if ($process.HasExited) { return $null }
    $expected = [System.IO.Path]::GetFullPath([string]$Identity.binary)
    if (-not [string]::Equals($observedPath, $expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw ('Refusing to stop an unrelated process with reused PID ' + $Identity.pid)
    }
    if ($observedStart.Ticks -ne ([DateTime]$Identity.started_at).ToUniversalTime().Ticks) {
        throw ('Refusing to stop a process whose start time differs from the saved process ' + $Identity.pid)
    }
    return $process
}

foreach ($name in @('app','generator')) {
    $recordPath = Join-Path $ArtifactsRoot ('runtime\' + $name + '.json')
    if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) { continue }
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    if ($record.home -and -not [string]::Equals([string]$record.home, $ArtifactsRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw ('Refusing to use an ownership record from another artifacts directory: ' + $name)
    }
    $identities = if ($record.processes) { @($record.processes) } else { @($record) }
    # Validate every live recorded identity before changing any process. An
    # unrelated/reused PID aborts cleanup instead of granting tree-wide kill.
    foreach ($identity in $identities) { $null = Get-VerifiedOwnedProcess $identity }
    $ordered = @($identities | Sort-Object -Property @{Expression = { [int]$_.depth }; Descending = $true}, @{Expression = { ([DateTime]$_.started_at).ToUniversalTime().Ticks }; Descending = $true})
    foreach ($identity in $ordered) {
        # The launcher can exit automatically when its worker exits. Check each
        # identity again immediately before stopping it; never kill by PID alone.
        $process = Get-VerifiedOwnedProcess $identity
        if ($null -ne $process) {
            try {
                Stop-Process -InputObject $process
                Write-Output ('Stopped ' + $name + ' process ' + $identity.pid)
            } catch {
                if (-not $process.HasExited) { throw }
            }
        }
    }
}
