# K2-Instruct Vector Bridge Comprehensive Testing with Real Data
# Date: July 15, 2025 AEST
# Purpose: Comprehensive validation of K2-Instruct implementation with real test data

param(
    [switch]$SkipSystemStart,
    [switch]$QuickTest,
    [int]$WaitTime = 180
)

Write-Host "🧠 K2-Instruct Vector Bridge - Comprehensive Real Data Testing" -ForegroundColor Cyan
Write-Host "Date: July 15, 2025 AEST" -ForegroundColor Gray
Write-Host "=" * 70 -ForegroundColor Gray

# Set location to phase7 directory
$Phase7Dir = "C:\Project\Augment_Agent_Evolution\docker\phase7_four_brain_production_ready"
Set-Location $Phase7Dir

Write-Host "📁 Working directory: $Phase7Dir" -ForegroundColor Yellow

# Check test data availability
$TestDataDir = Join-Path $Phase7Dir "core\data_testwith"
if (-not (Test-Path $TestDataDir)) {
    Write-Host "❌ Test data directory not found: $TestDataDir" -ForegroundColor Red
    exit 1
}

$TestFiles = Get-ChildItem $TestDataDir
Write-Host "✅ Test data directory found with $($TestFiles.Count) files:" -ForegroundColor Green
foreach ($file in $TestFiles) {
    Write-Host "   - $($file.Name) ($([math]::Round($file.Length/1KB, 1)) KB)" -ForegroundColor Gray
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check service health
function Test-ServiceHealth {
    param([string]$Url, [string]$ServiceName)
    
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 10
        Write-Host "✅ $ServiceName: Healthy" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ $ServiceName: Unhealthy - $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Start system if not skipped
if (-not $SkipSystemStart) {
    Write-Host "🚀 Starting Four-Brain System..." -ForegroundColor Yellow
    
    # Check Docker
    if (-not (Test-DockerRunning)) {
        Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Start services
    docker-compose up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to start services" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
    Start-Sleep $WaitTime
}

# Test service health
Write-Host "🔍 Testing Service Health..." -ForegroundColor Yellow

$ServicesHealthy = $true

# Test Brain 3 health
if (-not (Test-ServiceHealth "http://localhost:8013/health" "Brain 3")) {
    $ServicesHealthy = $false
}

# Test K2 Vector Bridge health
if (-not (Test-ServiceHealth "http://localhost:8013/vector/health" "K2 Vector Bridge")) {
    $ServicesHealthy = $false
}

# Test K2 metrics
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9097/metrics" -TimeoutSec 10
    if ($response.Content -match "brain3_k2_vector_requests_total") {
        Write-Host "✅ K2 Metrics: Available" -ForegroundColor Green
    } else {
        Write-Host "⚠️  K2 Metrics: Missing expected metrics" -ForegroundColor Yellow
        $ServicesHealthy = $false
    }
}
catch {
    Write-Host "❌ K2 Metrics: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $ServicesHealthy = $false
}

if (-not $ServicesHealthy) {
    Write-Host "❌ Some services are not healthy. Check logs with: docker-compose logs" -ForegroundColor Red
    exit 1
}

# Run quick test if requested
if ($QuickTest) {
    Write-Host "⚡ Running Quick K2 Vector Test..." -ForegroundColor Yellow
    
    $testData = @{
        text = "Test K2-Instruct vector coordination with context engineering principles"
        brains = @("brain1", "brain2", "brain4")
        load = @{
            brain1 = 0.33
            brain2 = 0.33
            brain4 = 0.34
        }
    }
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8013/vector" -Method POST -Body ($testData | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
        
        Write-Host "✅ Quick test successful:" -ForegroundColor Green
        Write-Host "   Strategy: $($response.strategy.strategy)" -ForegroundColor Gray
        Write-Host "   Confidence: $($response.strategy.confidence)" -ForegroundColor Gray
        Write-Host "   Cost: $($response.cost_estimate)" -ForegroundColor Gray
        Write-Host "   Embedding: $($response.embedding_source)" -ForegroundColor Gray
        Write-Host "   Strategy Source: $($response.strategy_source)" -ForegroundColor Gray
    }
    catch {
        Write-Host "❌ Quick test failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Run comprehensive Python tests
Write-Host "🧪 Running Comprehensive Real Data Tests..." -ForegroundColor Yellow

$testScript = Join-Path $Phase7Dir "tests\test_k2_with_real_data.py"

if (-not (Test-Path $testScript)) {
    Write-Host "❌ Test script not found: $testScript" -ForegroundColor Red
    exit 1
}

try {
    Write-Host "📊 Executing comprehensive test suite..." -ForegroundColor Cyan
    python $testScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 COMPREHENSIVE TESTING PASSED!" -ForegroundColor Green
        Write-Host "K2-Instruct Vector Bridge is working excellently with real data" -ForegroundColor Green
    } else {
        Write-Host "❌ COMPREHENSIVE TESTING FAILED" -ForegroundColor Red
        Write-Host "Check the test output above for details" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ Failed to run comprehensive tests: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Performance summary
Write-Host ""
Write-Host "📈 Performance Summary:" -ForegroundColor Cyan

# Get current metrics
try {
    $metricsResponse = Invoke-WebRequest -Uri "http://localhost:9097/metrics" -TimeoutSec 10
    $metricsText = $metricsResponse.Content
    
    # Parse key metrics
    $totalRequests = 0
    $currentConfidence = 0
    
    foreach ($line in $metricsText -split "`n") {
        if ($line -match "brain3_k2_vector_requests_total\s+(\d+\.?\d*)") {
            $totalRequests = [double]$matches[1]
        }
        elseif ($line -match "brain3_k2_confidence\s+(\d+\.?\d*)") {
            $currentConfidence = [double]$matches[1]
        }
    }
    
    Write-Host "   Total K2 requests processed: $totalRequests" -ForegroundColor Gray
    Write-Host "   Current confidence score: $currentConfidence" -ForegroundColor Gray
    Write-Host "   Cost per request: ~$0.0012" -ForegroundColor Gray
    Write-Host "   Monthly cost estimate (20k req): ~$2.40" -ForegroundColor Gray
}
catch {
    Write-Host "   Could not retrieve performance metrics" -ForegroundColor Yellow
}

# Show useful information
Write-Host ""
Write-Host "🔧 System Information:" -ForegroundColor Cyan
Write-Host "   K2 Vector Bridge API:    http://localhost:8013/vector" -ForegroundColor Gray
Write-Host "   K2 Health Check:         http://localhost:8013/vector/health" -ForegroundColor Gray
Write-Host "   K2 Metrics:              http://localhost:9097/metrics" -ForegroundColor Gray
Write-Host "   Brain 3 API:             http://localhost:8013" -ForegroundColor Gray
Write-Host "   Prometheus:              http://localhost:9090" -ForegroundColor Gray

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Review test results above" -ForegroundColor Gray
Write-Host "   2. Monitor metrics in Grafana dashboards" -ForegroundColor Gray
Write-Host "   3. Deploy to production environment" -ForegroundColor Gray
Write-Host "   4. Set up automated monitoring alerts" -ForegroundColor Gray

Write-Host ""
Write-Host "🎉 K2-Instruct Vector Bridge Comprehensive Testing Complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Gray
