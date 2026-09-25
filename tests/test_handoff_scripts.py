"""No real processes or model services are started/stopped by these tests."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.fixture
def installer():
    spec = importlib.util.spec_from_file_location("install_heads", SCRIPTS / "install-heads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def supplied_heads(tmp_path):
    source = tmp_path / "supplied heads"
    source.mkdir()
    for stage in ("vision", "text", "fusion"):
        (source / f"{stage}.pt").write_bytes(f"fixture only: {stage}".encode())
    return source


def test_head_installer_copies_all_then_is_idempotent(installer, tmp_path):
    source = supplied_heads(tmp_path)
    home = tmp_path / "runtime with spaces"
    first = installer.install(source, home)
    assert [record["status"] for record in first["heads"]] == ["installed"] * 3
    for stage in installer.STAGES:
        assert (home / "checkpoints" / f"{stage}.pt").read_bytes() == (source / f"{stage}.pt").read_bytes()
    second = installer.install(source, home)
    assert [record["status"] for record in second["heads"]] == ["already_identical"] * 3
    assert not list((home / "checkpoints").glob("*.installing"))


def test_head_conflict_is_detected_before_any_install(installer, tmp_path):
    source = supplied_heads(tmp_path)
    home = tmp_path / "runtime"
    (home / "checkpoints").mkdir(parents=True)
    existing = home / "checkpoints/fusion.pt"
    existing.write_bytes(b"different existing checkpoint")
    with pytest.raises(FileExistsError, match="differs"):
        installer.install(source, home)
    assert existing.read_bytes() == b"different existing checkpoint"
    assert not (home / "checkpoints/vision.pt").exists()


def test_missing_supplied_head_cannot_create_partial_install(installer, tmp_path):
    source = supplied_heads(tmp_path)
    (source / "text.pt").unlink()
    home = tmp_path / "runtime"
    with pytest.raises(FileNotFoundError, match="missing"):
        installer.install(source, home)
    assert not home.exists()


HARNESS = r'''
param([string]$Scripts, [string]$ArtifactsRoot, [string]$Scenario)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$global:started = New-Object System.Collections.ArrayList
$global:stopped = New-Object System.Collections.ArrayList
$global:spawned = $false
$global:scenarioName = $Scenario
$global:fixturePython = Join-Path (Split-Path -Parent $Scripts) 'python environment\python.exe'
$global:fixtureBinary = if ($Scenario.StartsWith('generator')) { Join-Path $ArtifactsRoot 'vendor\llama\llama-server.exe' } else { $global:fixturePython }
$global:fakeProcess = [pscustomobject]@{Id=44001;Path=$global:fixtureBinary;StartTime=([DateTime]'2026-09-25T12:34:56.1234567Z');HasExited=$false}
$global:fakeProcess | Add-Member -MemberType ScriptMethod -Name Refresh -Value { }
$global:fakeWorker = [pscustomobject]@{Id=44002;Path='C:\fixture\python3.12.exe';StartTime=$global:fakeProcess.StartTime.AddSeconds(1);HasExited=$false}
$global:fakeWorker | Add-Member -MemberType ScriptMethod -Name Refresh -Value { }
$global:hasWorker = $Scenario -in @('app-new-worker','app-reuse-worker','app-foreign-worker','stop-family','stop-family-conflict')
$global:reuse = $Scenario -in @('generator-reuse','app-reuse','app-other-port','app-reuse-worker','app-foreign-worker','stop-match','stop-wrong-time','stop-wrong-path','stop-family','stop-family-conflict')

function Get-Process { param($Id,$ErrorAction) if ($Id -eq 44001) { return $global:fakeProcess }; if ($Id -eq 44002 -and $global:hasWorker) { return $global:fakeWorker }; throw 'Unrecognized fixture PID' }
function Get-CimInstance {
    param($ClassName,$ErrorAction)
    if ($global:hasWorker -and ($global:reuse -or $global:spawned)) {
        $commandHome = if ($global:scenarioName -eq 'app-foreign-worker') { 'C:\unrelated-runtime' } else { $ArtifactsRoot }
        return [pscustomobject]@{ProcessId=44002;ParentProcessId=44001;Name='python3.12.exe';ExecutablePath=$global:fakeWorker.Path;CommandLine=('python.exe -m checkin.app --home "' + $commandHome + '" --host 127.0.0.1 --port 7860')}
    }
    return @()
}
function Start-Process {
    param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,[switch]$PassThru,$RedirectStandardOutput,$RedirectStandardError)
    [void]$global:started.Add(@{file=$FilePath;arguments=@($ArgumentList);window=$WindowStyle;working=$WorkingDirectory})
    if ($PassThru) { $global:spawned=$true; return $global:fakeProcess }
}
function Stop-Process { param($InputObject,$Id) if ($null -eq $InputObject) { throw 'Expected verified process object' }; [void]$global:stopped.Add($InputObject.Id) }
function Get-NetTCPConnection {
    param($LocalPort,$State,$ErrorAction)
    if ($global:scenarioName -like '*-foreign-port') { return [pscustomobject]@{OwningProcess=99002} }
    if ($global:reuse -or $global:spawned) { return [pscustomobject]@{OwningProcess=$(if($global:hasWorker){44002}else{44001})} }
    return @()
}
function Invoke-RestMethod { param($Uri,$Method,$TimeoutSec) return @{status=$(if($global:reuse -or $global:spawned){'ok'}else{'loading'})} }
function Invoke-WebRequest { param($Uri,[switch]$UseBasicParsing,$TimeoutSec) return @{StatusCode=200} }
function Start-Sleep { param($Milliseconds) }

$name = if ($Scenario.StartsWith('generator')) { 'generator' } else { 'app' }
if ($global:reuse) {
    $recordedTime = $global:fakeProcess.StartTime.ToUniversalTime()
    if ($Scenario -eq 'stop-wrong-time') { $recordedTime=$recordedTime.AddTicks(-1) }
    $recordedBinary = if ($Scenario -eq 'stop-wrong-path') { 'C:\unrelated\python.exe' } else { $global:fixtureBinary }
    $record=@{pid=44001;binary=$recordedBinary;started_at=$recordedTime.ToString('o');home=$ArtifactsRoot;port=$(if($Scenario -eq 'app-other-port'){7862}else{7860})}
    if ($Scenario -like 'stop-family*') {
        $workerTime=$global:fakeWorker.StartTime.ToUniversalTime()
        if ($Scenario -eq 'stop-family-conflict') { $workerTime=$workerTime.AddTicks(-1) }
        $record.processes=@(@{pid=44001;binary=$recordedBinary;started_at=$recordedTime.ToString('o');depth=0},@{pid=44002;binary=$global:fakeWorker.Path;started_at=$workerTime.ToString('o');depth=1})
    }
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ArtifactsRoot "runtime\$name.json") -Encoding UTF8
}
$failure = $null
try {
    if ($Scenario.StartsWith('generator')) { & (Join-Path $Scripts 'start-generator.ps1') -ArtifactsRoot $ArtifactsRoot -StartupTimeoutSeconds 5 }
    elseif ($Scenario.StartsWith('stop')) { & (Join-Path $Scripts 'stop.ps1') -ArtifactsRoot $ArtifactsRoot }
    else {
        $global:LASTEXITCODE=17
        Push-Location (Split-Path -Parent $Scripts)
        try { & (Join-Path $Scripts 'start.ps1') -ArtifactsRoot $ArtifactsRoot -PythonPath '.\python environment\python.exe' }
        finally { Pop-Location }
    }
} catch { $failure=$_.Exception.Message }
$recordFile=Join-Path $ArtifactsRoot "runtime\$name.json"
$saved = if(Test-Path -LiteralPath $recordFile){Get-Content -LiteralPath $recordFile -Raw|ConvertFrom-Json}else{$null}
@{failure=$failure;started=@($global:started.ToArray());stopped=@($global:stopped.ToArray());record=$saved;version=$PSVersionTable.PSVersion.Major} | ConvertTo-Json -Compress -Depth 10
'''


@pytest.mark.skipif(not SHELLS, reason="Windows PowerShell is unavailable")
@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("scenario", ["generator-new", "generator-reuse", "generator-foreign-port", "app-new", "app-reuse", "app-foreign-port", "app-other-port", "app-new-worker", "app-reuse-worker", "app-foreign-worker", "stop-match", "stop-wrong-time", "stop-wrong-path", "stop-family", "stop-family-conflict"])
def test_launcher_ownership_under_powershell_versions(tmp_path, shell, scenario):
    repo = tmp_path / "copied repository"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("start.ps1", "start-generator.ps1", "stop.ps1"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    if scenario.startswith("app"):
        # Isolate app-launch tests from the generator script; no real service.
        (scripts / "start-generator.ps1").write_text("param([string]$ArtifactsRoot)\n", encoding="utf-8")
    home = repo / "artifacts with spaces"
    for folder in ("models/qwen", "vendor/llama", "runtime", "checkpoints"):
        (home / folder).mkdir(parents=True)
    for name in ("models/qwen/Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf", "vendor/llama/llama-server.exe", "checkpoints/vision.pt", "checkpoints/text.pt", "checkpoints/fusion.pt"):
        (home / name).touch()
    (repo / "python environment").mkdir()
    (repo / "python environment/python.exe").touch()
    harness = tmp_path / "launcher-fixture.ps1"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness), "-Scripts", str(scripts), "-ArtifactsRoot", str(home), "-Scenario", scenario], capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    if scenario.endswith("foreign-port") or scenario in {"app-other-port", "app-foreign-worker", "stop-wrong-time", "stop-wrong-path", "stop-family-conflict"}:
        assert parsed["failure"], parsed
        assert parsed["started"] == []
        assert parsed["stopped"] == []
    else:
        assert parsed["failure"] is None, parsed
    if scenario in {"app-new", "app-new-worker", "generator-new"}:
        assert len(parsed["started"]) == 1
        assert parsed["started"][0]["window"] == "Hidden"
        assert Path(parsed["record"]["binary"]).is_absolute()
        quoted = parsed["started"][0]["arguments"][3 if scenario.startswith("app") else 1]
        assert quoted.startswith('"') and quoted.endswith('"')
        assert "artifacts with spaces" in quoted
    if scenario == "generator-reuse":
        assert parsed["started"] == []
    if scenario in {"app-reuse", "app-reuse-worker"}:
        assert len(parsed["started"]) == 1
        assert parsed["started"][0]["file"] == "http://127.0.0.1:7860/"
    if scenario == "stop-match":
        assert parsed["stopped"] == [44001]
    if scenario in {"app-new-worker", "app-reuse-worker"}:
        assert parsed["record"]["schema_version"] == 2
        assert parsed["record"]["pid"] == 44002
        assert {item["pid"] for item in parsed["record"]["processes"]} == {44001, 44002}
    if scenario == "stop-family":
        assert parsed["stopped"] == [44002, 44001]


@pytest.mark.skipif(not SHELLS, reason="Windows PowerShell is unavailable")
@pytest.mark.parametrize("shell", SHELLS)
def test_launcher_powershell_syntax(shell):
    command = "$ErrorActionPreference='Stop'; $errorsFound=@(); Get-ChildItem -LiteralPath $args[0] -Filter '*.ps1' | ForEach-Object { $tokens=$null; $parseErrors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$parseErrors); $errorsFound+=@($parseErrors) }; if($errorsFound.Count){throw ($errorsFound|Out-String)}"
    # -Command positional arguments have inconsistent parsing across PS 5/7;
    # use an environment value for this read-only syntax inspection.
    import os
    command = command.replace("$args[0]", "$env:CHECKIN_SCRIPT_TEST_ROOT")
    result = subprocess.run([shell, "-NoProfile", "-Command", command], env={**os.environ, "CHECKIN_SCRIPT_TEST_ROOT": str(SCRIPTS)}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
