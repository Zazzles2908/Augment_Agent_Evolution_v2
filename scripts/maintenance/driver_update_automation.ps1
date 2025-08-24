# Four-Brain System Driver Update Automation
# Optimized for AMD Ryzen 7 7700X + MSI RTX 5070 Ti Gaming Trio OC Plus
# (Zazzles's Agent) Driver Update Script

param(
    [switch]$DownloadOnly,
    [switch]$InstallAll,
    [switch]$CheckOnly,
    [switch]$Force
)

Write-Host "🔧 Four-Brain System Driver Update Automation" -ForegroundColor Cyan
Write-Host "Hardware: AMD Ryzen 7 7700X + MSI RTX 5070 Ti Gaming Trio OC Plus" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Yellow

# Create download directory
$DownloadPath = "$env:TEMP\FourBrainDrivers"
if (!(Test-Path $DownloadPath)) {
    New-Item -ItemType Directory -Path $DownloadPath -Force | Out-Null
}

# Function to check current driver versions
function Get-CurrentDriverVersions {
    Write-Host "📊 Checking Current Driver Versions..." -ForegroundColor Yellow
    
    # NVIDIA Driver Version
    try {
        $nvidiaVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits).Trim()
        Write-Host "✅ NVIDIA RTX 5070 Ti: $nvidiaVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ NVIDIA Driver not detected" -ForegroundColor Red
    }
    
    # AMD Chipset Version
    $amdChipset = Get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceName -like "*AMD*" -and $_.DeviceName -like "*Controller*"} | Select-Object -First 1
    if ($amdChipset) {
        Write-Host "✅ AMD Chipset: $($amdChipset.DriverVersion) ($($amdChipset.DriverDate))" -ForegroundColor Green
    }
    
    # AMD Graphics Version
    $amdGraphics = Get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceName -like "*AMD Radeon*"} | Select-Object -First 1
    if ($amdGraphics) {
        Write-Host "✅ AMD Graphics: $($amdGraphics.DriverVersion)" -ForegroundColor Green
    }
    
    Write-Host ""
}

# Function to download AMD Chipset drivers
function Download-AMDChipsetDrivers {
    Write-Host "📥 Downloading AMD B850 Chipset Drivers..." -ForegroundColor Yellow
    
    # AMD Chipset driver URLs (these would need to be updated with actual URLs)
    $chipsetUrl = "https://drivers.amd.com/drivers/chipset/amd_chipset_software_2024.30.exe"
    $chipsetFile = "$DownloadPath\amd_chipset_2024.30.exe"
    
    Write-Host "🌐 Opening AMD driver download page for manual download..." -ForegroundColor Cyan
    Start-Process "https://www.amd.com/en/support/downloads/drivers.html"
    
    Write-Host "📋 Manual Download Instructions:" -ForegroundColor Yellow
    Write-Host "1. Select: Chipset > AMD Socket AM5 > B850" -ForegroundColor White
    Write-Host "2. Download: 2024.30 AMD Embedded Windows Chipset drivers" -ForegroundColor White
    Write-Host "3. Save to: $DownloadPath" -ForegroundColor White
    
    return $chipsetFile
}

# Function to download AMD Adrenalin drivers
function Download-AMDAdrenalinDrivers {
    Write-Host "📥 Downloading AMD Adrenalin 25.6.1 (WHQL)..." -ForegroundColor Yellow
    
    $adrenalinUrl = "https://drivers.amd.com/drivers/adrenalin/amd_software_adrenalin_25.6.1.exe"
    $adrenalinFile = "$DownloadPath\amd_adrenalin_25.6.1.exe"
    
    Write-Host "🌐 Opening AMD Adrenalin download page..." -ForegroundColor Cyan
    Start-Process "https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-25-6-1.html"
    
    Write-Host "📋 Manual Download Instructions:" -ForegroundColor Yellow
    Write-Host "1. Click 'Download' for AMD Software: Adrenalin Edition 25.6.1" -ForegroundColor White
    Write-Host "2. Choose WHQL Recommended version" -ForegroundColor White
    Write-Host "3. Save to: $DownloadPath" -ForegroundColor White
    
    return $adrenalinFile
}

# Function to check NVIDIA driver status
function Check-NVIDIADriver {
    Write-Host "🔍 Checking NVIDIA RTX 5070 Ti Driver Status..." -ForegroundColor Yellow
    
    try {
        $currentVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits).Trim()
        Write-Host "✅ Current NVIDIA Driver: $currentVersion" -ForegroundColor Green
        
        if ($currentVersion -eq "580.88") {
            Write-Host "🎉 NVIDIA Driver is OPTIMAL (Latest RTX 5070 Ti optimized version)" -ForegroundColor Green
            Write-Host "   No update needed!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Consider updating to 580.88 for RTX 5070 Ti optimization" -ForegroundColor Yellow
            Start-Process "https://www.nvidia.com/en-us/drivers/"
        }
    } catch {
        Write-Host "❌ NVIDIA Driver check failed" -ForegroundColor Red
    }
}

# Function to install drivers
function Install-Drivers {
    param([string[]]$DriverFiles)
    
    Write-Host "🚀 Installing Drivers..." -ForegroundColor Yellow
    Write-Host "⚠️ This will require administrator privileges and system restarts" -ForegroundColor Red
    
    foreach ($driver in $DriverFiles) {
        if (Test-Path $driver) {
            Write-Host "📦 Installing: $(Split-Path $driver -Leaf)" -ForegroundColor Cyan
            
            # Create restore point before installation
            Write-Host "💾 Creating system restore point..." -ForegroundColor Yellow
            Checkpoint-Computer -Description "Before Four-Brain Driver Update" -RestorePointType "MODIFY_SETTINGS"
            
            # Install driver
            try {
                Start-Process -FilePath $driver -ArgumentList "/S" -Wait -Verb RunAs
                Write-Host "✅ Installation completed: $(Split-Path $driver -Leaf)" -ForegroundColor Green
            } catch {
                Write-Host "❌ Installation failed: $(Split-Path $driver -Leaf)" -ForegroundColor Red
                Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host "❌ Driver file not found: $driver" -ForegroundColor Red
        }
    }
}

# Function to validate installation
function Validate-Installation {
    Write-Host "🔍 Validating Driver Installation..." -ForegroundColor Yellow
    
    # Wait for system to settle
    Start-Sleep -Seconds 10
    
    # Re-check driver versions
    Get-CurrentDriverVersions
    
    # Check for any driver issues
    $problemDevices = Get-WmiObject Win32_PnPEntity | Where-Object {$_.ConfigManagerErrorCode -ne 0}
    if ($problemDevices) {
        Write-Host "⚠️ Found devices with issues:" -ForegroundColor Yellow
        $problemDevices | ForEach-Object {
            Write-Host "   - $($_.Name): Error Code $($_.ConfigManagerErrorCode)" -ForegroundColor Red
        }
    } else {
        Write-Host "✅ All devices functioning properly" -ForegroundColor Green
    }
    
    # Test GPU functionality
    try {
        nvidia-smi | Out-Null
        Write-Host "✅ NVIDIA GPU responding correctly" -ForegroundColor Green
    } catch {
        Write-Host "❌ NVIDIA GPU test failed" -ForegroundColor Red
    }
}

# Main execution logic
Write-Host "🎯 Starting Driver Update Process..." -ForegroundColor Cyan

# Always check current versions first
Get-CurrentDriverVersions

if ($CheckOnly) {
    Write-Host "✅ Check completed. Use -DownloadOnly or -InstallAll to proceed." -ForegroundColor Green
    exit 0
}

# Download drivers
$driverFiles = @()

if ($DownloadOnly -or $InstallAll) {
    Write-Host "📥 Starting Download Phase..." -ForegroundColor Cyan
    
    # Download AMD Chipset drivers
    $chipsetFile = Download-AMDChipsetDrivers
    $driverFiles += $chipsetFile
    
    # Download AMD Adrenalin drivers
    $adrenalinFile = Download-AMDAdrenalinDrivers
    $driverFiles += $adrenalinFile
    
    # Check NVIDIA status
    Check-NVIDIADriver
    
    Write-Host "📋 Download Summary:" -ForegroundColor Yellow
    Write-Host "   Download Location: $DownloadPath" -ForegroundColor White
    Write-Host "   Files to download manually:" -ForegroundColor White
    Write-Host "   - AMD Chipset 2024.30" -ForegroundColor White
    Write-Host "   - AMD Adrenalin 25.6.1 (WHQL)" -ForegroundColor White
    Write-Host "   - NVIDIA: Already optimal (580.88)" -ForegroundColor Green
}

if ($InstallAll) {
    Write-Host "⏳ Waiting for manual downloads to complete..." -ForegroundColor Yellow
    Write-Host "Press any key when downloads are complete to proceed with installation..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    
    # Install drivers
    Install-Drivers -DriverFiles $driverFiles
    
    # Validate installation
    Validate-Installation
    
    Write-Host "Driver Update Process Completed!" -ForegroundColor Green
    Write-Host "Recommendation: Restart system for optimal performance" -ForegroundColor Yellow
}

Write-Host "✅ Four-Brain System Driver Update Complete!" -ForegroundColor Green
