Param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [string]$MinShape = 'input_ids:1x128,attention_mask:1x128',
  [string]$OptShape = 'input_ids:4x256,attention_mask:4x256',
  [string]$MaxShape = 'input_ids:4x512,attention_mask:4x512',
  [int]$WorkspaceMB = 4096,
  [string]$TimingCacheFile = '',
  [int]$BuilderOpt = 3,
  [int]$MaxTactics = 1024,
  [switch]$NoVerbose,
  [string]$ExtraArgs = ''
)

Write-Host "Using Repo: $Repo"
Write-Host "Shapes: min=$MinShape opt=$OptShape max=$MaxShape"
Write-Host "Workspace: ${WorkspaceMB}MB"
if ($TimingCacheFile) { Write-Host "TimingCacheFile: $TimingCacheFile" }
Write-Host "BuilderOpt: $BuilderOpt"
Write-Host "MaxTactics: $MaxTactics"
if ($ExtraArgs) { Write-Host "ExtraArgs: $ExtraArgs" }

$env:POWERSHELL_TELEMETRY_OPTOUT = '1'

$dockerArgs = @(
  'run','--rm','--gpus','all',
  '-v',"${Repo}:/models",
  'nvcr.io/nvidia/tensorrt:25.06-py3',
  'trtexec',
  '--onnx=/models/qwen3_embedding/1/model.onnx',
  '--saveEngine=/models/qwen3_embedding_trt/1/model.plan',
  '--allowWeightStreaming','--weightStreamingBudget=6G','--fp16',
  "--minShapes=$MinShape",
  "--optShapes=$OptShape",
  "--maxShapes=$MaxShape",
  "--memPoolSize=workspace:${WorkspaceMB}",
  "--builderOptimizationLevel=$BuilderOpt",
  "--maxTactics=$MaxTactics",
  '--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN'
)
if ($TimingCacheFile) {
  $dockerArgs += "--timingCacheFile=$TimingCacheFile"
}
if ($ExtraArgs) {
  $dockerArgs += $ExtraArgs.Split(' ')
}
if (-not $NoVerbose) { $dockerArgs += '--verbose' }

Write-Host ("Running: docker " + ($dockerArgs -join ' '))
$proc = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru -Wait
exit $proc.ExitCode

