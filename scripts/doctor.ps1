param(
  [switch]$StrictOptional
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Get-CommandVersion {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string[]]$Arguments = @("--version")
  )

  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $command) {
    return [ordered]@{ available = $false; version = $null; path = $null }
  }

  $output = try {
    (& $command.Source @Arguments 2>&1 | Select-Object -First 1) -join "`n"
  } catch {
    $_.Exception.Message
  }
  return [ordered]@{ available = $true; version = $output; path = $command.Source }
}

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
  return $null
}

function Get-PythonAudit {
  param([string]$Python)

  $code = @'
import importlib.util
import json
import platform

def module_version(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "version": None}
    module = __import__(name)
    return {"available": True, "version": getattr(module, "__version__", "unknown")}

result = {
    "python": platform.python_version(),
    "executable": platform.python_compiler(),
    "torch": {"available": False, "version": None, "cuda_available": False, "cuda_version": None},
    "torch_geometric": module_version("torch_geometric"),
    "mlflow": module_version("mlflow"),
    "pyg_cpu_smoke": {"status": "skipped", "reason": "torch or torch_geometric not installed"},
}

torch_spec = importlib.util.find_spec("torch")
pyg_spec = importlib.util.find_spec("torch_geometric")
if torch_spec is not None:
    import torch
    result["torch"] = {
        "available": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
if torch_spec is not None and pyg_spec is not None:
    try:
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["stock"].x = torch.ones((1, 1), dtype=torch.float32)
        data["seat"].x = torch.ones((1, 1), dtype=torch.float32)
        data["seat", "buys", "stock"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        result["pyg_cpu_smoke"] = {
            "status": "passed",
            "node_types": list(data.node_types),
            "edge_types": [list(edge_type) for edge_type in data.edge_types],
        }
    except Exception as error:
        result["pyg_cpu_smoke"] = {"status": "failed", "reason": str(error)}

print(json.dumps(result, ensure_ascii=False))
'@

  $raw = $code | & $Python -
  if ($LASTEXITCODE -ne 0) {
    return [ordered]@{
      available = $false
      path = $Python
      error = "Python audit failed with exit code $LASTEXITCODE"
    }
  }
  return $raw | ConvertFrom-Json
}

$pythonPath = Find-Python
$pythonAudit = if ($pythonPath) { Get-PythonAudit $pythonPath } else { $null }
$docker = Get-CommandVersion "docker"
$dockerCompose = [ordered]@{ available = $false; version = $null; path = $null }
if ($docker.available) {
  $composeVersion = try {
    (& $docker.path compose version 2>&1 | Select-Object -First 1) -join "`n"
  } catch {
    $_.Exception.Message
  }
  $dockerCompose = [ordered]@{
    available = $composeVersion -match "Docker Compose"
    version = $composeVersion
    path = $docker.path
  }
}

$audit = [ordered]@{
  generated_at = (Get-Date).ToUniversalTime().ToString("o")
  repository = "$RepoRoot"
  python = if ($pythonAudit) { $pythonAudit } else { [ordered]@{ available = $false } }
  jdk = Get-CommandVersion "java" @("-version")
  maven = Get-CommandVersion "mvn" @("-version")
  node = Get-CommandVersion "node" @("--version")
  npm = Get-CommandVersion "npm" @("--version")
  nextjs = Get-CommandVersion "next" @("--version")
  docker = $docker
  docker_compose = $dockerCompose
  optional_policy = [ordered]@{
    torch_pyg = "optional extra: seatflow[ml]; missing packages do not block dependency-free CPU baseline"
    mlflow = "optional extra: seatflow[tracking]; missing package falls back to local lineage in later milestones"
  }
}

$audit | ConvertTo-Json -Depth 8

if ($StrictOptional -and $pythonAudit) {
  if (-not $pythonAudit.torch.available -or -not $pythonAudit.torch_geometric.available) {
    throw "Strict optional check failed: torch/torch_geometric is not installed."
  }
  if ($pythonAudit.pyg_cpu_smoke.status -ne "passed") {
    throw "Strict optional check failed: PyG CPU smoke did not pass."
  }
}
