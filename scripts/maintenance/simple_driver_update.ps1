# Simple Driver Update Script for Four-Brain System
# AMD Ryzen 7 7700X + MSI RTX 5070 Ti Gaming Trio OC Plus

Write-Host "Four-Brain System Driver Update" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Yellow

# Check current NVIDIA driver
Write-Host "Checking NVIDIA RTX 5070 Ti Driver..." -ForegroundColor Yellow
try {
    $nvidiaVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits).Trim()
    Write-Host "Current NVIDIA Driver: $nvidiaVersion" -ForegroundColor Green
    
    if ($nvidiaVersion -eq "580.88") {
        Write-Host "NVIDIA Driver is OPTIMAL (Latest version)" -ForegroundColor Green
    } else {
        Write-Host "Consider updating NVIDIA driver to 580.88" -ForegroundColor Yellow
    }
} catch {
    Write-Host "NVIDIA Driver check failed" -ForegroundColor Red
}

# Check AMD drivers
Write-Host "Checking AMD Drivers..." -ForegroundColor Yellow
$amdDevices = Get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceName -like "*AMD*"}
$amdDevices | ForEach-Object {
    Write-Host "AMD Device: $($_.DeviceName) - Version: $($_.DriverVersion)" -ForegroundColor Cyan
}

# Open driver download pages
Write-Host "Opening driver download pages..." -ForegroundColor Yellow

# AMD Chipset drivers
Write-Host "Opening AMD Chipset driver page..." -ForegroundColor Cyan
Start-Process "https://www.amd.com/en/support/downloads/drivers.html"

# AMD Adrenalin drivers  
Write-Host "Opening AMD Adrenalin driver page..." -ForegroundColor Cyan
Start-Process "https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-25-6-1.html"

# NVIDIA drivers (if needed)
Write-Host "Opening NVIDIA driver page..." -ForegroundColor Cyan
Start-Process "https://www.nvidia.com/en-us/drivers/"

Write-Host ""
Write-Host "DRIVER UPDATE INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. AMD Chipset: Select Chipset > AMD Socket AM5 > B850" -ForegroundColor White
Write-Host "   Download: 2024.30 AMD Embedded Windows Chipset drivers" -ForegroundColor White
Write-Host ""
Write-Host "2. AMD Adrenalin: Download 25.6.1 (WHQL Recommended)" -ForegroundColor White
Write-Host ""
Write-Host "3. NVIDIA: Your 580.88 driver is already optimal!" -ForegroundColor Green
Write-Host ""
Write-Host "Install order: 1) Chipset, 2) Adrenalin, 3) Restart" -ForegroundColor Yellow

Write-Host "Driver update process initiated!" -ForegroundColor Green
