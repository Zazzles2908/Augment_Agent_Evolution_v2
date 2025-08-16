Param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$OnnxModelName = 'model.onnx',
  [string]$PlanName = 'model.plan',
  [string]$MinShape = 'input_ids:1x128,attention_mask:1x128',
  [string]$OptShape = 'input_ids:4x256,attention_mask:4x256',
  [string]$MaxShape = 'input_ids:4x512,attention_mask:4x512',
  [int]$WorkspaceMB = 4096,
  [int]$BuilderOpt = 5,
  [int]$MaxTactics = 2048,
  [string]$TimingCacheFile = '',
  [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
Write-Host "(AugmentAI) Building INT4 TensorRT engine for $Model using trtexec (25.06)" -ForegroundColor Cyan

# Paths: assume ONNX under <base>/1/model.onnx and output under <model>_trt/1/model.plan when applicable
$onnxDir = Join-Path $Repo $Model
$onnxPath = Join-Path (Join-Path $onnxDir '1') $OnnxModelName

# If model already *_trt, keep output in same folder
if ($Model.ToLower().EndsWith('_trt')) {
  $planDir = Join-Path $Repo $Model
} else {
  $planDir = Join-Path $Repo ("${Model}_trt")
}
$planPath = Join-Path (Join-Path $planDir '1') $PlanName

# Ensure output dir exists
if (-not (Test-Path (Split-Path $planPath))) { New-Item -ItemType Directory -Force -Path (Split-Path $planPath) | Out-Null }

if (-not (Test-Path $onnxPath)) {
  throw "ONNX not found: $onnxPath"
}

# Build args
$dockerImage = 'nvcr.io/nvidia/tensorrt:25.06-py3'
$dockerArgs = @(
  'run','--rm','--gpus','all',
  '-v',"${Repo}:/models",
  $dockerImage,
  'trtexec',
  "--onnx=/models/$Model/1/$OnnxModelName",
  "--saveEngine=/models/$($planDir.Substring($Repo.Length+1).Replace('\\','/') + '/1/' + $PlanName)",
  '--int4',
  "--minShapes=$MinShape",
  "--optShapes=$OptShape",
  "--maxShapes=$MaxShape",
  "--memPoolSize=workspace:${WorkspaceMB}",
  "--builderOptimizationLevel=$BuilderOpt",
  "--maxTactics=$MaxTactics",
  '--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN',
  '--skipInference'
)

if ($TimingCacheFile) { $dockerArgs += "--timingCacheFile=/models/$($TimingCacheFile)" }
if ($Verbose) { $dockerArgs += '--verbose=1' }

Write-Host "(AugmentAI) Running: docker $($dockerArgs -join ' ')" -ForegroundColor Yellow
$proc = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru -Wait
if ($proc.ExitCode -ne 0) { throw "trtexec build failed with exit code $($proc.ExitCode)" }

Write-Host "(AugmentAI) ✅ INT4 engine built: $planPath" -ForegroundColor Green

