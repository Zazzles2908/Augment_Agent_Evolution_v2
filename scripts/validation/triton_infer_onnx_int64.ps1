Param(
  [string]$TritonUrl = "http://localhost:8000",
  [string]$ModelName = "qwen3_embedding"
)

$payload = @{ 
  inputs = @(
    @{ name = "input_ids"; datatype = "INT64"; shape = @(1,8); data = @(101,2009,2001,1037,2204,2154,102,0) },
    @{ name = "attention_mask"; datatype = "INT64"; shape = @(1,8); data = @(1,1,1,1,1,1,1,0) }
  );
  outputs = @(@{ name = "embedding"; binary_data = $false })
}

Write-Host "== ONNX INT64 minimal inference ($ModelName) =="
# Ensure model loaded
try { Invoke-RestMethod -Method POST -Uri "$TritonUrl/v2/repository/models/$ModelName/load" -ContentType 'application/json' -Body '{}' } catch {}

$result = Invoke-RestMethod -Method POST -Uri "$TritonUrl/v2/models/$ModelName/infer" -ContentType 'application/json' -Body ($payload | ConvertTo-Json -Compress)
$dim = ($result.outputs | Where-Object { $_.name -eq 'embedding' }).shape
Write-Host "Output shape:" ($dim -join 'x')

