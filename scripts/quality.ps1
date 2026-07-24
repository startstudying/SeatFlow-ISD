param(
  [switch]$IncludeBuild,
  [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$env:PYTHONPATH = "src"

function Find-Python {
  $candidates = @(
    ".omx\integration-venv\Scripts\python.exe",
    ".venv\Scripts\python.exe",
    "python"
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command) {
      return $command.Source
    }
  }
  throw "Python 3.11+ was not found on PATH or in known local virtualenvs."
}

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][scriptblock]$Command
  )

  Write-Host "==> $Name"
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
}

$Python = Find-Python
Write-Host "Using Python: $Python"

if (-not $SkipDoctor) {
  Invoke-Step "doctor" { powershell -NoProfile -ExecutionPolicy Bypass -File scripts\doctor.ps1 }
}

Invoke-Step "unittest" { & $Python -m unittest discover -s tests -v }
Invoke-Step "ruff" { & $Python -m ruff check . }
Invoke-Step "mypy" { & $Python -m mypy src tests }
Invoke-Step "preflight" { & $Python -m seatflow preflight --config configs/default.toml }
Invoke-Step "compileall" { & $Python -m compileall src tests }

if ($IncludeBuild) {
  Invoke-Step "build" { & $Python -m build }
}

Write-Host "Quality checks passed."
