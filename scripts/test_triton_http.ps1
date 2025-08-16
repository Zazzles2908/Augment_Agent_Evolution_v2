Param(
  [string]$Url = 'http://localhost:8000/v2/health/ready'
)
Write-Host "Checking Triton health at $Url"
try {
  $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
  if ($r.StatusCode -eq 200) {
    Write-Host "Triton READY"
    exit 0
  } else {
    Write-Host "Triton not ready, code: $($r.StatusCode)"
    exit 1
  }
} catch {
  Write-Host "Request failed: $($_.Exception.Message)"
  exit 1
}

