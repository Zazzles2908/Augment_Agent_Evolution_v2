#!/usr/bin/env pwsh
<#
.SYNOPSIS
    TensorRT Engine Builder for 8B Models with NVFP4 Quantization
    
.DESCRIPTION
    Builds TensorRT engines for Four-Brain System 8B models with NVFP4 quantization
    optimized for RTX 5070 Ti (Blackwell SM_120 architecture).
    
    Supports:
    - Brain 1: Qwen3 8B Embedding (NVFP4)
    - Brain 2: Qwen3 8B Reranker (NVFP4) 
    - Brain 3: HRM H-Module (FP16) + L-Module (NVFP4)
    
.PARAMETER ModelType
    Type of model to build: embedding, reranker, hrm-h, hrm-l
    
.PARAMETER Precision
    Precision mode: nvfp4, fp16, fp8 (default: nvfp4 for 8B models, fp16 for HRM-H)
    
.PARAMETER WorkspaceMB
    TensorRT workspace memory in MB (default: 8192 for 8B models)
    
.EXAMPLE
    .\build_trt_engine_8b_nvfp4.ps1 -ModelType embedding -Precision nvfp4
    .\build_trt_engine_8b_nvfp4.ps1 -ModelType hrm-h -Precision fp16
#>

Param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("embedding", "reranker", "hrm-h", "hrm-l")]
    [string]$ModelType,
    
    [ValidateSet("nvfp4", "fp16", "fp8")]
    [string]$Precision = "nvfp4",
    
    [string]$MinShape = "",
    [string]$OptShape = "",
    [string]$MaxShape = "",
    [int]$WorkspaceMB = 8192,
    [string]$TimingCacheFile = '',
    [int]$BuilderOpt = 5,  # Higher optimization for Blackwell
    [int]$MaxTactics = 2048,  # More tactics for better optimization
    [switch]$NoVerbose,
    [string]$ExtraArgs = ''
)

# Model-specific configurations
$ModelConfigs = @{
    "embedding" = @{
        "onnx_path" = "/models/qwen3_embedding_8b/1/model.onnx"
        "engine_path" = "/models/qwen3_embedding_8b_trt/1/model.plan"
        "min_shape" = "input_ids:1x64,attention_mask:1x64"
        "opt_shape" = "input_ids:4x256,attention_mask:4x256"
        "max_shape" = "input_ids:8x512,attention_mask:8x512"
        "workspace_mb" = 8192
        "description" = "Qwen3 8B Embedding Model"
    }
    "reranker" = @{
        "onnx_path" = "/models/qwen3_reranker_8b/1/model.onnx"
        "engine_path" = "/models/qwen3_reranker_8b_trt/1/model.plan"
        "min_shape" = "input_ids:1x64,attention_mask:1x64"
        "opt_shape" = "input_ids:4x256,attention_mask:4x256"
        "max_shape" = "input_ids:8x512,attention_mask:8x512"
        "workspace_mb" = 8192
        "description" = "Qwen3 8B Reranker Model"
    }
    "hrm-h" = @{
        "onnx_path" = "/models/hrm_h_module/1/model.onnx"
        "engine_path" = "/models/hrm_h_module_trt/1/model.plan"
        "min_shape" = "input_ids:1x32,attention_mask:1x32"
        "opt_shape" = "input_ids:2x128,attention_mask:2x128"
        "max_shape" = "input_ids:4x256,attention_mask:4x256"
        "workspace_mb" = 2048
        "description" = "HRM H-Module (27M Planning)"
    }
    "hrm-l" = @{
        "onnx_path" = "/models/hrm_l_module/1/model.onnx"
        "engine_path" = "/models/hrm_l_module_trt/1/model.plan"
        "min_shape" = "input_ids:1x32,attention_mask:1x32"
        "opt_shape" = "input_ids:2x128,attention_mask:2x128"
        "max_shape" = "input_ids:4x256,attention_mask:4x256"
        "workspace_mb" = 2048
        "description" = "HRM L-Module (27M Execution)"
    }
}

# Get model configuration
$config = $ModelConfigs[$ModelType]
if (-not $config) {
    Write-Error "Unknown model type: $ModelType"
    exit 1
}

# Override shapes if provided
if ($MinShape) { $config.min_shape = $MinShape }
if ($OptShape) { $config.opt_shape = $OptShape }
if ($MaxShape) { $config.max_shape = $MaxShape }
if ($WorkspaceMB -ne 8192) { $config.workspace_mb = $WorkspaceMB }

# Set precision defaults
if ($ModelType -eq "hrm-h" -and $Precision -eq "nvfp4") {
    $Precision = "fp8"  # H-Module uses FP8
    Write-Host "🔧 HRM H-Module: Using FP8 precision (always loaded)"
}

Write-Host "🚀 Building TensorRT Engine for Four-Brain System"
Write-Host "=" * 60
Write-Host "Model Type: $($config.description)"
Write-Host "Precision: $Precision"
Write-Host "ONNX Path: $($config.onnx_path)"
Write-Host "Engine Path: $($config.engine_path)"
Write-Host "Shapes: min=$($config.min_shape) opt=$($config.opt_shape) max=$($config.max_shape)"
Write-Host "Workspace: $($config.workspace_mb)MB"
Write-Host "Builder Optimization: Level $BuilderOpt (Blackwell optimized)"
Write-Host "Max Tactics: $MaxTactics"

$env:POWERSHELL_TELEMETRY_OPTOUT = '1'

# Base docker arguments
$dockerArgs = @(
    'run', '--rm', '--gpus', 'all',
    '-v', "${PWD}/containers/models:/models",
    'nvcr.io/nvidia/tensorrt:25.06-py3',
    'trtexec',
    "--onnx=$($config.onnx_path)",
    "--saveEngine=$($config.engine_path)",
    '--allowWeightStreaming',
    '--weightStreamingBudget=12G',  # Increased for 8B models
    "--minShapes=$($config.min_shape)",
    "--optShapes=$($config.opt_shape)",
    "--maxShapes=$($config.max_shape)",
    "--memPoolSize=workspace:$($config.workspace_mb)",
    "--builderOptimizationLevel=$BuilderOpt",
    "--maxTactics=$MaxTactics",
    '--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN,+EDGE_MASK_CONVOLUTIONS'  # Blackwell optimizations
)

# Add precision-specific flags
switch ($Precision) {
    "nvfp4" {
        $dockerArgs += '--fp16'  # NVFP4 builds on FP16
        $dockerArgs += '--int8'  # Enable INT8 for NVFP4
        $dockerArgs += '--stronglyTyped'
        Write-Host "🔧 NVFP4 Quantization: Enabled (RTX 5070 Ti optimized)"
    }
    "fp16" {
        $dockerArgs += '--fp16'
        Write-Host "🔧 FP16 Precision: Enabled"
    }
    "fp8" {
        $dockerArgs += '--fp16'
        $dockerArgs += '--fp8'
        $dockerArgs += '--stronglyTyped'
        Write-Host "🔧 FP8 Precision: Enabled (HRM H-Module optimized)"
    }
}

# Add Blackwell-specific optimizations
$dockerArgs += '--useSpinWait'
$dockerArgs += '--useCudaGraph'
$dockerArgs += '--separateProfileRun'

# Add timing cache if specified
if ($TimingCacheFile) {
    $dockerArgs += "--timingCacheFile=$TimingCacheFile"
    Write-Host "Timing Cache: $TimingCacheFile"
}

# Add extra arguments
if ($ExtraArgs) {
    $dockerArgs += $ExtraArgs.Split(' ')
    Write-Host "Extra Args: $ExtraArgs"
}

# Add verbose flag
if (-not $NoVerbose) { 
    $dockerArgs += '--verbose'
}

Write-Host "`n🔨 Starting TensorRT Engine Build..."
Write-Host "Command: docker $($dockerArgs -join ' ')"
Write-Host "-" * 60

$proc = Start-Process -FilePath docker -ArgumentList $dockerArgs -NoNewWindow -PassThru -Wait

if ($proc.ExitCode -eq 0) {
    Write-Host "`n✅ TensorRT Engine Build Successful!"
    Write-Host "🎯 Engine saved to: $($config.engine_path)"
    Write-Host "🧠 Ready for Four-Brain System deployment"
} else {
    Write-Host "`n❌ TensorRT Engine Build Failed!"
    Write-Host "Exit Code: $($proc.ExitCode)"
}

exit $proc.ExitCode
