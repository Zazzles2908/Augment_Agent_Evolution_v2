<#
Windows SSD Cleanup Helper
#>
[CmdletBinding()]
param(
  [switch]$Execute,
  [string[]]$Roots
)
function Get-SizeGB {
  param([string]$Path)
  try {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($item.PSIsContainer) {
      $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
      return [math]::Round(($sum/1GB), 2)
    } else {
      return [math]::Round((($item.Length)/1GB), 2)
    }
  } catch { return 0 }
}
function Safe-Remove {
  param([string]$Path)
  try {
    if (Test-Path -LiteralPath $Path) {
      Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    }
  } catch {}
}
$report = @()
$User = $Env:USERPROFILE

# FIXED: Correctly define the array with separate Join-Path calls
$Candidates = @(
  # ML / AI caches
  (Join-Path $User ".cache\huggingface"),
  (Join-Path $User ".cache\torch"),
  (Join-Path $User ".cache\pip"),
  # Python / package caches
  (Join-Path $User "AppData\Local\pip\Cache"),
  (Join-Path $User "AppData\Local\pypoetry\Cache"),
  # Node / JS caches
  (Join-Path $User "AppData\Local\npm-cache"),
  (Join-Path $User "AppData\Local\Yarn\Cache"),
  (Join-Path $User "AppData\Local\pnpm\store"),
  # General temp
  (Join-Path $User "AppData\Local\Temp"),
  # Docker WSL VHDX (file)
  "C:\Users\Jazeel-Home\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
)

Write-Host "Scanning common caches..." -ForegroundColor Cyan
foreach ($c in $Candidates) {
  $size = Get-SizeGB -Path $c
  
  # PowerShell 5.1 compatible replacement for ternary operator
  if (Test-Path $c) {
      $item = Get-Item $c
      if ($item.PSIsContainer) {
          $type = 'Folder'
      } else {
          $type = 'File'
      }
  } else {
      $type = 'Missing'
  }
  
  $report += [pscustomobject]@{ Path = $c; Type = $type; SizeGB = $size }
}
# Optional roots scan for heavy folders
$HeavyNames = @('node_modules', '.venv', 'venv', 'dist', 'build', '__pycache__', '.pytest_cache')
if ($Roots) {
  Write-Host "Scanning specified roots for heavy folders..." -ForegroundColor Cyan
  foreach ($root in $Roots) {
    if (Test-Path $root) {
      try {
        Get-ChildItem -LiteralPath $root -Directory -Recurse -ErrorAction SilentlyContinue |
          Where-Object { $HeavyNames -contains $_.Name } |
          ForEach-Object {
            $p = $_.FullName
            $size = Get-SizeGB -Path $p
            $report += [pscustomobject]@{ Path = $p; Type = 'Folder'; SizeGB = $size }
          }
      } catch {}
    }
  }
}
$report = $report | Sort-Object SizeGB -Descending
Write-Host "`nTop 30 heaviest paths:" -ForegroundColor Yellow
$report | Select-Object -First 30 | Format-Table -AutoSize
if (-not $Execute) {
  Write-Host "`nDry run complete. Re-run with -Execute to delete known caches (except VHDX)." -ForegroundColor Green
  Write-Host "For Windows Disk Cleanup & Storage Sense, see docs/SSD_CLEANUP_GUIDE.md" -ForegroundColor Green
  exit 0
}
Write-Host "`nExecute mode: deleting common caches (this may take time)..." -ForegroundColor Red
foreach ($c in $Candidates) {
  # Never delete the VHDX automatically; provide instructions instead
  if ($c -like '*.vhdx') { continue }
  Safe-Remove -Path $c
}
if ($Roots) {
  foreach ($root in $Roots) {
    if (Test-Path $root) {
      foreach ($name in $HeavyNames) {
        Get-ChildItem -LiteralPath $root -Directory -Recurse -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -eq $name } |
          ForEach-Object { Safe-Remove -Path $_.FullName }
      }
    }
  }
}
Write-Host "`nDeletion pass complete. Consider optimizing your Docker WSL VHDX (requires admin):" -ForegroundColor Yellow
Write-Host "  wsl --shutdown" -ForegroundColor Gray
Write-Host "  Optimize-VHD -Path 'C:\\Users\\Jazeel-Home\\AppData\\Local\\Docker\\wsl\\disk\\docker_data.vhdx' -Mode Full" -ForegroundColor Gray
Write-Host "(Run in an elevated PowerShell with Hyper-V module available)" -ForegroundColor Gray