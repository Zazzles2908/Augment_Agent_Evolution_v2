Param(
  [string]$TritonUrl = "http://localhost:8000"
)
Write-Host "== Triton Health Check =="
try {
  $ready = Invoke-RestMethod -Method GET -Uri "$TritonUrl/v2/health/ready" -TimeoutSec 10
  Write-Host "Ready: OK"
} catch { Write-Error "Health ready failed: $($_.Exception.Message)" }

try {
  $index = Invoke-RestMethod -Method GET -Uri "$TritonUrl/v2/repository/index" -TimeoutSec 15
  Write-Host "Models listed:" ($index | ConvertTo-Json -Depth 3)
} catch { Write-Error "Repository index failed: $($_.Exception.Message)" }

