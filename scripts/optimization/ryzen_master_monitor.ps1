# Ryzen Master Optimization Monitoring Script
# Monitors CPU performance and temperatures during optimization

Write-Host "Ryzen Master Optimization Monitor" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Yellow

# Function to get CPU information
function Get-CPUInfo {
    $cpu = Get-WmiObject Win32_Processor
    $temp = Get-WmiObject -Namespace "root/OpenHardwareMonitor" -Class Sensor -ErrorAction SilentlyContinue | Where-Object {$_.SensorType -eq "Temperature" -and $_.Name -like "*CPU*"}
    
    return @{
        Name = $cpu.Name
        CurrentClockSpeed = $cpu.CurrentClockSpeed
        MaxClockSpeed = $cpu.MaxClockSpeed
        LoadPercentage = $cpu.LoadPercentage
        Temperature = if($temp) { $temp.Value } else { "N/A" }
    }
}

# Function to check Ryzen Master status
function Check-RyzenMasterStatus {
    $rmProcess = Get-Process -Name "AMDRyzenMasterDriver*" -ErrorAction SilentlyContinue
    if ($rmProcess) {
        Write-Host "✅ Ryzen Master Driver: Running" -ForegroundColor Green
    } else {
        Write-Host "❌ Ryzen Master Driver: Not detected" -ForegroundColor Red
    }
    
    $rmService = Get-Service -Name "AMDRyzenMasterDriverV*" -ErrorAction SilentlyContinue
    if ($rmService -and $rmService.Status -eq "Running") {
        Write-Host "✅ Ryzen Master Service: Running" -ForegroundColor Green
    } else {
        Write-Host "❌ Ryzen Master Service: Not running" -ForegroundColor Red
    }
}

# Function to monitor CPU performance
function Monitor-CPUPerformance {
    param([int]$Duration = 60)
    
    Write-Host "Monitoring CPU for $Duration seconds..." -ForegroundColor Yellow
    
    $startTime = Get-Date
    $samples = @()
    
    while ((Get-Date) -lt $startTime.AddSeconds($Duration)) {
        $cpuInfo = Get-CPUInfo
        $sample = @{
            Time = Get-Date
            ClockSpeed = $cpuInfo.CurrentClockSpeed
            Load = $cpuInfo.LoadPercentage
            Temperature = $cpuInfo.Temperature
        }
        $samples += $sample
        
        Write-Host "$(Get-Date -Format 'HH:mm:ss') | Clock: $($cpuInfo.CurrentClockSpeed)MHz | Load: $($cpuInfo.LoadPercentage)% | Temp: $($cpuInfo.Temperature)°C" -ForegroundColor Cyan
        
        Start-Sleep -Seconds 5
    }
    
    return $samples
}

# Function to run CPU stress test
function Start-CPUStressTest {
    Write-Host "Starting CPU stress test..." -ForegroundColor Yellow
    Write-Host "This will load all CPU cores to test stability" -ForegroundColor Yellow
    
    # Simple PowerShell CPU stress test
    $jobs = @()
    for ($i = 0; $i -lt $env:NUMBER_OF_PROCESSORS; $i++) {
        $job = Start-Job -ScriptBlock {
            $endTime = (Get-Date).AddMinutes(5)
            while ((Get-Date) -lt $endTime) {
                $result = 1
                for ($j = 0; $j -lt 10000; $j++) {
                    $result = $result * 1.0001
                }
            }
        }
        $jobs += $job
    }
    
    Write-Host "Stress test running for 5 minutes..." -ForegroundColor Yellow
    Write-Host "Monitor temperatures in Ryzen Master!" -ForegroundColor Red
    
    # Monitor during stress test
    Monitor-CPUPerformance -Duration 300
    
    # Clean up jobs
    $jobs | Remove-Job -Force
    Write-Host "Stress test completed!" -ForegroundColor Green
}

# Function to check optimization status
function Check-OptimizationStatus {
    Write-Host "Checking optimization status..." -ForegroundColor Yellow
    
    # Check CPU base vs current speed
    $cpuInfo = Get-CPUInfo
    $baseSpeed = 3800  # 3.8 GHz base for 7700X
    $currentSpeed = $cpuInfo.CurrentClockSpeed
    
    if ($currentSpeed -gt $baseSpeed) {
        $boost = $currentSpeed - $baseSpeed
        Write-Host "✅ CPU Boost Active: +${boost}MHz above base" -ForegroundColor Green
    } else {
        Write-Host "⚠️ CPU at base speed or lower" -ForegroundColor Yellow
    }
    
    # Check power plan
    $powerPlan = powercfg /getactivescheme
    if ($powerPlan -like "*High performance*" -or $powerPlan -like "*Ultimate*") {
        Write-Host "✅ Power Plan: High Performance" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Power Plan: Not optimized for performance" -ForegroundColor Yellow
        Write-Host "   Recommendation: Set to High Performance" -ForegroundColor White
    }
    
    # Check CPU parking
    $cpuParking = powercfg /query scheme_current sub_processor CPMINCORES | Select-String "Current AC Power Setting Index"
    Write-Host "CPU Parking Status: $cpuParking" -ForegroundColor Cyan
}

# Function to show optimization recommendations
function Show-OptimizationRecommendations {
    Write-Host ""
    Write-Host "RYZEN MASTER OPTIMIZATION RECOMMENDATIONS:" -ForegroundColor Yellow
    Write-Host "===========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. CONTROL MODE SETUP:" -ForegroundColor Cyan
    Write-Host "   - Open Ryzen Master" -ForegroundColor White
    Write-Host "   - Change Control Mode to 'Precision Boost Overdrive'" -ForegroundColor White
    Write-Host "   - Verify PPT shows numerical values (not 0/1)" -ForegroundColor White
    Write-Host ""
    Write-Host "2. CONSERVATIVE PBO SETTINGS:" -ForegroundColor Cyan
    Write-Host "   - PPT: 125000 (125W)" -ForegroundColor White
    Write-Host "   - TDC: 85000 (85A)" -ForegroundColor White
    Write-Host "   - EDC: 125000 (125A)" -ForegroundColor White
    Write-Host "   - Max Boost: 0 MHz (start conservative)" -ForegroundColor White
    Write-Host ""
    Write-Host "3. MONITORING:" -ForegroundColor Cyan
    Write-Host "   - Watch temperatures (target <80°C)" -ForegroundColor White
    Write-Host "   - Test stability for 24 hours" -ForegroundColor White
    Write-Host "   - Run Four-Brain workloads" -ForegroundColor White
    Write-Host ""
    Write-Host "4. PROGRESSIVE TUNING:" -ForegroundColor Cyan
    Write-Host "   - Week 1: Conservative settings" -ForegroundColor White
    Write-Host "   - Week 2: Increase to 142000/95000/140000" -ForegroundColor White
    Write-Host "   - Week 3: Add Curve Optimizer (-10 to -20)" -ForegroundColor White
}

# Main menu
function Show-Menu {
    Write-Host ""
    Write-Host "RYZEN MASTER MONITORING MENU:" -ForegroundColor Green
    Write-Host "1. Check Ryzen Master Status" -ForegroundColor White
    Write-Host "2. Monitor CPU Performance (60 seconds)" -ForegroundColor White
    Write-Host "3. Run CPU Stress Test (5 minutes)" -ForegroundColor White
    Write-Host "4. Check Optimization Status" -ForegroundColor White
    Write-Host "5. Show Optimization Recommendations" -ForegroundColor White
    Write-Host "6. Exit" -ForegroundColor White
    Write-Host ""
}

# Main execution
do {
    Show-Menu
    $choice = Read-Host "Select option (1-6)"
    
    switch ($choice) {
        "1" { Check-RyzenMasterStatus }
        "2" { Monitor-CPUPerformance }
        "3" { Start-CPUStressTest }
        "4" { Check-OptimizationStatus }
        "5" { Show-OptimizationRecommendations }
        "6" { Write-Host "Exiting..." -ForegroundColor Green; break }
        default { Write-Host "Invalid option. Please select 1-6." -ForegroundColor Red }
    }
    
    if ($choice -ne "6") {
        Write-Host ""
        Write-Host "Press any key to continue..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
} while ($choice -ne "6")

Write-Host "Ryzen Master monitoring completed!" -ForegroundColor Green
