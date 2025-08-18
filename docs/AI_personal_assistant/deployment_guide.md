# Deployment Guide — Windows 11 + Docker Desktop (GPU)

This guide provides step-by-step instructions tailored for your RTX 5070 Ti (16GB), WSL2, and CUDA 13 via containers.

## 0) Pre-Flight Ritual
```powershell
# 1 System health
docker ps --format "table {{.Names}}\t{{.Status}}"
supabase status
# 2 Logs tail
Get-ChildItem -Recurse -File -Filter *.log | Sort LastWriteTime -Descending | Select -First 1 | % { Get-Content $_.FullName -Tail 20 }
# 3 Inter-file scan
gci -Recurse -File | ? {$_.Name -match '\.(md|json|yaml|yml|toml|env)$'} | Sort LastWriteTime -Descending | Select -First 10
# 4 Git state
git status; git log --oneline -5
# 5 Agent metrics
$agentMetrics = @{ SuggestionAccuracy = "Calculate from acceptance rate"; ResponseTime = "Measure avg"; MemoryUsage = "Track VRAM"; IntegrationStatus = "Check MCP" }
$agentMetrics | ConvertTo-Json | Out-File "C:\AI_Cache\agent-metrics.json"
```

## 1) Install Prereqs
- Docker Desktop 4.44.1+ with WSL2 & GPU support
- NVIDIA GPU driver (supports CUDA 13.x)
- WSL2 distro (Ubuntu 24.04 recommended)
- Supabase CLI (optional local dev)

## 2) GPU Validation
```powershell
wsl --update
wsl --set-default-version 2
wsl -d Ubuntu nvidia-smi
```

## 3) Clone and Configure
```powershell
cd C:\Project
# repo is already present per your setup
Copy-Item .env.example .env -Force  # if available
```

## 4) Models and Engines
- Download models under `containers/four-brain/models/`
- Build TensorRT engines inside WSL2 using `trtexec` (see inference_stack.md)
- Place engines in `containers/four-brain/triton/model_repository/`

## 5) Start Stack
```powershell
cd containers\four-brain\docker
$env:COMPOSE_PROJECT_NAME = "four_brain"
docker compose up -d --build
```

## 6) Health Checks
```powershell
curl http://localhost:9098/health  # Orchestrator
curl http://localhost:8000/v2/health/ready  # Triton (mapped port)
```

## 7) Supabase (Optional Local)
```powershell
supabase init
supabase start
# Run SQL from database_architecture.md to create schema
```

## 8) Logs & Metrics
```powershell
docker compose ps
docker compose logs -f orchestrator
```

## 9) Shutdown
```powershell
docker compose down
```

## Troubleshooting
- Engines missing: Triton returns model not found → build and mount engines
- GPU unavailable: ensure NVIDIA Container Toolkit in WSL2; update drivers
- OOM: reduce batch size, single instance, unload idle models

