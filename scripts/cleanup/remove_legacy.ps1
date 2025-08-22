# Remove legacy/junk directories and backups (approved by maintainer)
# Usage: run from repo root
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup/remove_legacy.ps1

$ErrorActionPreference = 'Continue'

$targets = @(
  'docker_config_backup',
  'hrm_config_backup',
  'hrm_phase0',
  'zen-mcp-server',
  'model_backup_20250819_152158.tar.gz',
  'model_backup_20250819_152624.tar.gz',
  'qwen3_embedding_trt_backup.plan'
)

$results = @()
foreach ($p in $targets) {
  if (Test-Path -LiteralPath $p) {
    try {
      Write-Host ("Removing: {0}" -f $p) -ForegroundColor Yellow
      Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Stop
      $results += @{ path = $p; status = 'removed' }
    }
    catch {
      Write-Warning ("Failed to remove {0}: {1}" -f $p, $_.Exception.Message)
      $results += @{ path = $p; status = 'failed'; error = $_.Exception.Message }
    }
  }
  else {
    Write-Host ("Skip (not found): {0}" -f $p) -ForegroundColor DarkGray
    $results += @{ path = $p; status = 'not_found' }
  }
}

Write-Host "\nSummary:" -ForegroundColor Cyan
$results | ForEach-Object { Write-Host (" - {0}: {1}" -f $_.path, $_.status) }

# Exit code: 0 if all removed or not found; 1 if any failed
if ($results | Where-Object { $_.status -eq 'failed' }) { exit 1 } else { exit 0 }

