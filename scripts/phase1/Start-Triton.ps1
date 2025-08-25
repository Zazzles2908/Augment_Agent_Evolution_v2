param(
  [string]$ModelRepo = "containers/four-brain/triton/model_repository",
  [int]$HttpPort = 8000,
  [int]$MetricsPort = 8002,
  [switch]$DryRun = $true
)
$cmd = "tritonserver --model-repository=$ModelRepo --model-control-mode=explicit --http-port=$HttpPort --metrics-port=$MetricsPort"
Write-Host "Start Triton: $cmd"
if (-not $DryRun) { iex $cmd }

