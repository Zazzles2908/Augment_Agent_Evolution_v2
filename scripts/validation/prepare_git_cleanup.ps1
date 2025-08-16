<#
DO NOT EXECUTE WITHOUT EXPLICIT APPROVAL
Prepares git rm --cached commands to purge tracked model artifacts from Triton model_repository.
Reference: docs/augment_code/AUDIT_2025-08-16.md
#>
Param(
  [switch]$WhatIf = $true
)

$paths = @(
  "containers/four-brain/triton/model_repository/hrm_h_trt/1/**",
  "containers/four-brain/triton/model_repository/hrm_l_trt/1/**",
  "containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/model.plan",
  "containers/four-brain/triton/model_repository/qwen3_reranker_trt/1/model.onnx",
  "containers/four-brain/triton/model_repository/qwen3_embedding/1/**",
  "containers/four-brain/triton/model_repository/docling_gpu/1/**"
)

Write-Host "== GIT CLEANUP PREPARATION =="
Write-Host "These commands will remove tracked artifacts from the Git index only. Files remain on disk."

foreach ($p in $paths) {
  $cmd = "git rm -r --cached `"$p`""
  Write-Host $cmd
  if (-not $WhatIf) {
    & git rm -r --cached $p
  }
}

Write-Host "-- Ensure .gitignore contains: containers/**/triton/model_repository/**/[0-9]*/"

