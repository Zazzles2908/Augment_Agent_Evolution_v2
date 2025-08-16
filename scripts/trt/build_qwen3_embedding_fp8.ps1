param(
  [string]$User = $env:USERNAME,
  [string]$ModelDirWin = "containers\four-brain\triton\model_repository\qwen3_embedding\1",
  [string]$OutRepoWin = "containers\four-brain\triton\model_repository\qwen3_embedding_trt\1",
  [string]$MinShape = "1x128",
  [string]$OptShape = "4x256",
  [string]$MaxShape = "4x512",
  [int]$WorkspaceMiB = 8192,
  [switch]$EnableWeightStreaming,
  [string]$WeightStreamingBudget = "8G"
)

$ErrorActionPreference = 'Stop'
Write-Host '(AugmentAI) Build Qwen3-4B Embedding FP8 TensorRT engine (Ubuntu 24.04 / WSL)' -ForegroundColor Cyan
$EnableFp8 = $false

# 1) Resolve Windows and WSL paths
$ModelOnnxWin = Join-Path $ModelDirWin 'model.onnx'
if (!(Test-Path $ModelOnnxWin)) { throw "ONNX not found: $ModelOnnxWin" }
$ModelDirFull = (Resolve-Path -LiteralPath $ModelDirWin).Path
$OutRepoFull = (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force $OutRepoWin)).Path
# Robust conversion to WSL paths (manual to avoid backslash issues)
function Convert-ToWslPath([string]$winPath) {
  $p = $winPath.Replace(':','').Trim()
  if ($p.StartsWith('\')) { $p = $p.Substring(1) }
  return '/mnt/' + ($winPath.Substring(0,1).ToLower()) + '/' + ($winPath.Substring(3).Replace('\','/'))
}
$ModelDirWsl = Convert-ToWslPath $ModelDirFull
$OutPlanWin = Join-Path $OutRepoFull 'model.plan'
$OutPlanWsl = Convert-ToWslPath $OutPlanWin

Write-Host ("Model (Win): " + $ModelOnnxWin) -ForegroundColor Green
Write-Host ("Model (WSL): " + $ModelDirWsl) -ForegroundColor Green
Write-Host ("Output (WSL): " + $OutPlanWsl) -ForegroundColor Green

# 2) Build FP8 engine via trtexec inside WSL
$ShapesMin = "input_ids:$MinShape,attention_mask:$MinShape"
$ShapesOpt = "input_ids:$OptShape,attention_mask:$OptShape"
$ShapesMax = "input_ids:$MaxShape,attention_mask:$MaxShape"

$wsArgs = @()
if ($EnableWeightStreaming) {
  if ($EnableFp8) {
    Write-Warning "(AugmentAI) TensorRT 10.13: --fp8 is not allowed with --stronglyTyped (required for weight streaming). Disabling weight streaming to keep FP8."
  } else {
    $wsArgs += "--allowWeightStreaming"
    $wsArgs += "--stronglyTyped"
    if ($WeightStreamingBudget) { $wsArgs += "--weightStreamingBudget=$WeightStreamingBudget" }
  }
}
$wsArgsString = ($wsArgs -join ' ')

$Cmd = @(
  "cd '$ModelDirWsl'",
  "&&",
  "trtexec",
  "--onnx=model.onnx",
  "--saveEngine='$OutPlanWsl'",
  "--minShapes=$ShapesMin",
  "--optShapes=$ShapesOpt",
  "--maxShapes=$ShapesMax",
  "--memPoolSize=workspace:${WorkspaceMiB}MiB",
  "--builderOptimizationLevel=5",
  "$wsArgsString",
  "--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN",
  "--skipInference --verbose=0"
) -join ' '

Write-Host "(AugmentAI) Running: wsl -e bash -lc \"$Cmd\"" -ForegroundColor Yellow
& wsl.exe -e bash -lc "$Cmd"
if ($LASTEXITCODE -ne 0) { throw "trtexec build failed (exit $LASTEXITCODE)" }

# 3) Quick load check
& wsl.exe -e bash -lc "trtexec --loadEngine='$OutPlanWsl' --dumpLayerInfo --skipInference --verbose=0 | head -n 20"
if ($LASTEXITCODE -ne 0) { throw "Engine load check failed (exit $LASTEXITCODE)" }

Write-Host '(AugmentAI) ✅ FP8 engine built and placed under qwen3_embedding_trt/1/model.plan' -ForegroundColor Cyan

