<#
Phase 1 Runner (Windows PowerShell)
- Builds TRT engines, starts Triton, runs smoke tests
- Integrates Zen validation at each critical step
- Supports -DryRun to preview commands and provide context files for Zen
#>
param(
  [switch]$DryRun = $true,
  [string]$ModelRepo = "containers/four-brain/triton/model_repository",
  [string]$Precision = "nvfp4",
  [string]$RedisUrl = "redis://localhost:6379",
  [int]$TritonHttp = 8000,
  [int]$TritonMetrics = 8002,
  [string]$Zen = "zen",
  [switch]$UseCompose = $true
)

function Invoke-Step($Name, $Cmd) {
  Write-Host ("`n==> " + $Name)
  if ($DryRun) {
    Write-Host "[DRY-RUN] $Cmd"
  } else {
    iex $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name" }
  }
}

function Check-GPU {
  try { nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits } catch { Write-Host "nvidia-smi not available" }
}
function Check-Disk { Get-PSDrive -Name (Get-Location).Drive.Name | Select-Object Name,Used,Free }

# Zen audit of scripts
Invoke-Step "Zen audit TensorRT scripts" "$Zen analyze_zen --target scripts/tensorrt/ --context documents/stack --checks precision,shapes,hardcoded"

# Build engines
Check-GPU; Check-Disk
Invoke-Step "Build qwen3_4b_embedding" "python scripts/tensorrt/convert_model.py --repo $ModelRepo --model qwen3_4b_embedding --config containers/four-brain/triton/config/embed.yaml --precision $Precision"
Check-GPU; Check-Disk
Invoke-Step "Build qwen3_0_6b_reranking" "python scripts/tensorrt/convert_model.py --repo $ModelRepo --model qwen3_0_6b_reranking --config containers/four-brain/triton/config/rerank.yaml --precision $Precision"
Check-GPU; Check-Disk
Invoke-Step "Build glm45_air" "python scripts/tensorrt/convert_model.py --repo $ModelRepo --model glm45_air --config containers/four-brain/triton/config/generate.yaml --precision $Precision"

# Triton load tracing
Invoke-Step "Zen trace Triton load" "$Zen tracer_zen --command 'tritonserver --model-repository=$ModelRepo --model-control-mode=explicit' --output traces/model_loading"

# Smoke tests with tracing
Invoke-Step "Repo ops" "$Zen tracer_zen --command 'bash scripts/smoke/triton_repository.sh' --output traces/model_ops"
Invoke-Step "Embed infer" "$Zen tracer_zen --command 'python scripts/smoke/embed_infer.py' --output traces/embed"
Invoke-Step "Rerank infer" "$Zen tracer_zen --command 'python scripts/smoke/rerank_infer.py' --output traces/rerank"
Invoke-Step "Generate infer" "$Zen tracer_zen --command 'python scripts/smoke/generate_infer.py' --output traces/generate"

# Ingestion and query with Zen checks
Invoke-Step "Ingestion (Redis optional)" "set REDIS_URL=$RedisUrl; python services/ingestion/docling_ingest.py --input ./documents --batch-size 16"
Invoke-Step "Zen analyze cache function" "$Zen analyze_zen --function cache_embedding --target services/ingestion/docling_ingest.py --context documents/stack"
Invoke-Step "Query path" "set REDIS_URL=$RedisUrl; python services/query/e2e_query_service.py 'what is redis?'"
Invoke-Step "Zen trace query" "$Zen tracer_zen --command 'python services/query/e2e_query_service.py "what is redis?"' --output traces/query"

# Report generation
Invoke-Step "Generate validation report" "$Zen docgen_zen --source traces/ --output documents/reports/phase1_validation_report.md --context documents/stack"

Write-Host "`nAll Phase 1 steps orchestrated. Use -DryRun:$false to execute for real." -ForegroundColor Green

