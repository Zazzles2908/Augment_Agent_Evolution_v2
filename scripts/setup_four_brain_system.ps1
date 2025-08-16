#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Four-Brain System Setup Script
    
.DESCRIPTION
    Comprehensive setup script for the Four-Brain System with 8B models, NVFP4 quantization,
    and orchestrator-based architecture. This script handles:
    
    - Environment configuration
    - Model downloads (8B models)
    - TensorRT engine building with NVFP4 quantization
    - Docker container deployment
    - System validation
    
.PARAMETER SkipModelDownload
    Skip downloading models (if already downloaded)
    
.PARAMETER SkipTensorRT
    Skip TensorRT engine building (if already built)
    
.PARAMETER ConfigOnly
    Only setup configuration files, don't start services
    
.EXAMPLE
    .\setup_four_brain_system.ps1
    .\setup_four_brain_system.ps1 -SkipModelDownload
    .\setup_four_brain_system.ps1 -ConfigOnly
#>

Param(
    [switch]$SkipModelDownload,
    [switch]$SkipTensorRT,
    [switch]$ConfigOnly,
    [string]$EnvFile = "containers/four-brain/.env"
)

# Color output functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Step { param($Message) Write-Host "🔧 $Message" -ForegroundColor Blue }

Write-Host @"
🚀 Four-Brain System Setup
========================
Brain 1: Embedding Service (Qwen3 8B NVFP4)
Brain 2: Reranking Service (Qwen3 8B NVFP4)  
Brain 3: Intelligence Service (HRM Manager)
Brain 4: Document Processor (Docling)
Orchestrator Hub: Central coordination
"@ -ForegroundColor Magenta

# Check prerequisites
Write-Step "Checking prerequisites..."

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Success "Docker found: $dockerVersion"
} catch {
    Write-Error "Docker not found. Please install Docker Desktop."
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker compose version
    Write-Success "Docker Compose found: $composeVersion"
} catch {
    Write-Error "Docker Compose not found. Please install Docker Compose."
    exit 1
}

# Check NVIDIA Docker
try {
    $nvidiaInfo = docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
    if ($nvidiaInfo) {
        Write-Success "NVIDIA GPU found: $nvidiaInfo"
    } else {
        Write-Warning "NVIDIA GPU not detected. Some features may not work."
    }
} catch {
    Write-Warning "NVIDIA Docker runtime not available. GPU acceleration disabled."
}

# Check available disk space
$drive = Get-PSDrive C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
if ($freeSpaceGB -lt 100) {
    Write-Warning "Low disk space: ${freeSpaceGB}GB free. Recommend at least 100GB for 8B models."
} else {
    Write-Success "Sufficient disk space: ${freeSpaceGB}GB free"
}

# Setup environment configuration
Write-Step "Setting up environment configuration..."

if (-not (Test-Path $EnvFile)) {
    if (Test-Path "containers/four-brain/.env.sample") {
        Copy-Item "containers/four-brain/.env.sample" $EnvFile
        Write-Success "Created .env file from template"
        Write-Warning "Please edit $EnvFile and fill in your API keys and configuration"
        Write-Info "Required: POSTGRES_PASSWORD, SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY"
        Write-Info "Optional: HF_TOKEN, K2_API_KEY, GLM_API_KEY"
        
        if (-not $ConfigOnly) {
            $continue = Read-Host "Continue with setup? (y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Info "Setup paused. Please configure $EnvFile and run again."
                exit 0
            }
        }
    } else {
        Write-Error ".env.sample not found. Cannot create environment configuration."
        exit 1
    }
} else {
    Write-Success "Environment file exists: $EnvFile"
}

if ($ConfigOnly) {
    Write-Success "Configuration setup complete. Use docker compose up to start services."
    exit 0
}

# Download models
if (-not $SkipModelDownload) {
    Write-Step "Downloading 8B models for Four-Brain system..."
    
    if (Test-Path "scripts/download_models.py") {
        try {
            python scripts/download_models.py
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Model download completed successfully"
            } else {
                Write-Error "Model download failed"
                exit 1
            }
        } catch {
            Write-Error "Failed to run model download script: $_"
            exit 1
        }
    } else {
        Write-Warning "Model download script not found. Skipping model download."
    }
} else {
    Write-Info "Skipping model download (--SkipModelDownload specified)"
}

# Build TensorRT engines
if (-not $SkipTensorRT) {
    Write-Step "Building TensorRT engines with NVFP4 quantization..."
    
    $models = @("embedding", "reranker", "hrm-h", "hrm-l")
    
    foreach ($model in $models) {
        Write-Info "Building TensorRT engine for $model..."
        
        if (Test-Path "scripts/build_trt_engine_8b_nvfp4.ps1") {
            try {
                $precision = if ($model -eq "hrm-h") { "fp16" } else { "nvfp4" }
                & "scripts/build_trt_engine_8b_nvfp4.ps1" -ModelType $model -Precision $precision
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "TensorRT engine built for $model"
                } else {
                    Write-Warning "TensorRT engine build failed for $model (continuing...)"
                }
            } catch {
                Write-Warning "Failed to build TensorRT engine for $model : $_"
            }
        } else {
            Write-Warning "TensorRT build script not found. Skipping engine building."
            break
        }
    }
} else {
    Write-Info "Skipping TensorRT engine building (--SkipTensorRT specified)"
}

# Start services
Write-Step "Starting Four-Brain system services..."

Set-Location "containers/four-brain/docker"

try {
    # Pull latest images
    Write-Info "Pulling latest Docker images..."
    docker compose pull
    
    # Build custom images
    Write-Info "Building Four-Brain services..."
    docker compose build
    
    # Start infrastructure services first
    Write-Info "Starting infrastructure services..."
    docker compose up -d postgres redis
    
    # Wait for infrastructure to be ready
    Write-Info "Waiting for infrastructure services to be ready..."
    Start-Sleep -Seconds 30
    
    # Start Triton
    Write-Info "Starting Triton Inference Server..."
    docker compose up -d triton
    
    # Wait for Triton to be ready
    Write-Info "Waiting for Triton to be ready..."
    Start-Sleep -Seconds 45
    
    # Start Four-Brain services
    Write-Info "Starting Four-Brain services..."
    docker compose up -d orchestrator-hub embedding-service reranker-service intelligence-service document-processor
    
    # Start monitoring and proxy
    Write-Info "Starting monitoring and proxy services..."
    docker compose up -d prometheus grafana nginx-proxy four-brain-dashboard
    
    Write-Success "Four-Brain system started successfully!"
    
} catch {
    Write-Error "Failed to start services: $_"
    exit 1
} finally {
    Set-Location $PSScriptRoot
}

# Validate system
Write-Step "Validating Four-Brain system..."

$services = @(
    @{Name="Postgres"; URL="http://localhost:5433"; Description="Database"},
    @{Name="Redis"; URL="http://localhost:6379"; Description="Cache"},
    @{Name="Triton"; URL="http://localhost:8000/v2/health/ready"; Description="Inference Server"},
    @{Name="Orchestrator"; URL="http://localhost:9018/health"; Description="Orchestrator Hub"},
    @{Name="Brain 1"; URL="http://localhost:8011/health"; Description="Embedding Service"},
    @{Name="Brain 2"; URL="http://localhost:8012/health"; Description="Reranker Service"},
    @{Name="Brain 3"; URL="http://localhost:8013/health"; Description="Intelligence Service"},
    @{Name="Brain 4"; URL="http://localhost:8014/health"; Description="Document Processor"},
    @{Name="Dashboard"; URL="http://localhost:3001"; Description="Four-Brain Dashboard"}
)

Write-Info "Checking service health..."
foreach ($service in $services) {
    try {
        if ($service.Name -eq "Postgres" -or $service.Name -eq "Redis") {
            # Skip HTTP check for database services
            Write-Success "$($service.Name): $($service.Description) - Running"
            continue
        }
        
        $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "$($service.Name): $($service.Description) - Healthy"
        } else {
            Write-Warning "$($service.Name): $($service.Description) - Status: $($response.StatusCode)"
        }
    } catch {
        Write-Warning "$($service.Name): $($service.Description) - Not responding"
    }
}

Write-Host @"

🎉 Four-Brain System Setup Complete!

Access Points:
- Dashboard: http://localhost:3001
- Orchestrator API: http://localhost:9018
- Brain 1 (Embedding): http://localhost:8011
- Brain 2 (Reranker): http://localhost:8012  
- Brain 3 (Intelligence): http://localhost:8013
- Brain 4 (Document): http://localhost:8014
- Triton Server: http://localhost:8000
- Grafana Monitoring: http://localhost:3000

Next Steps:
1. Configure API keys in $EnvFile
2. Test the system with sample requests
3. Monitor performance in Grafana
4. Check logs: docker compose logs -f

"@ -ForegroundColor Green
