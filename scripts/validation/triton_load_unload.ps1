Param(
  [string]$TritonUrl = "http://localhost:8000",
  [string[]]$Models = @("qwen3_embedding_trt","qwen3_reranker_trt","docling_gpu","glm45_air")
)

Function Invoke-JsonPost($Url, $Body) {
  Invoke-RestMethod -Method POST -Uri $Url -ContentType 'application/json' -Body ($Body | ConvertTo-Json -Compress)
}

Write-Host "== Triton Load/Unload Tests =="
foreach ($m in $Models) {
  Write-Host "-- Loading $m"
  try { Invoke-JsonPost "$TritonUrl/v2/repository/models/$m/load" @{} | Out-Null } catch { Write-Warning "Load failed: $m - $($_.Exception.Message)" }
  Start-Sleep -Seconds 1
  try {
    $meta = Invoke-RestMethod -Method GET -Uri "$TritonUrl/v2/models/$m" -TimeoutSec 15
    $ready = $null
    if ($meta.PSObject.Properties.Name -contains 'state') { $ready = $meta.state }
    elseif ($meta.PSObject.Properties.Name -contains 'status') { $ready = $meta.status }
    elseif ($meta.PSObject.Properties.Name -contains 'ready') { $ready = $meta.ready }
    Write-Host "$m READY?" $ready
  } catch { Write-Warning "Meta failed: $m - $($_.Exception.Message)" }

  Write-Host "-- Unloading $m"
  try { Invoke-JsonPost "$TritonUrl/v2/repository/models/$m/unload" @{} | Out-Null } catch { Write-Warning "Unload failed: $m - $($_.Exception.Message)" }
  Start-Sleep -Seconds 1
}

