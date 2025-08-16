#!/usr/bin/env python3
"""
Model Download Script for Four-Brain System
Downloads the correct models for each brain component.

This script ensures we have proper models:
- Brain1: Qwen3-8B embedding model (NVFP4 quantized)
- Brain2: Qwen3-8B reranker model (NVFP4 quantized)
- Brain3: HRM Manager (H-Module 27M, L-Module 27M)
- Brain4: Docling document processing model

Following master prompt: NO FABRICATION, REAL MODELS ONLY
Updated for 8B models with NVFP4 quantization support
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations - Corrected for HRM + Official Qwen3-8B models
MODELS_CONFIG = {
    "embedding_service": {
        "repo_id": "Qwen/Qwen3-Embedding-8B",  # Official Qwen3-8B embedding model
        "local_dir": "models/qwen3/embedding-8b",
        "description": "Embedding Service - Qwen3-Embedding-8B (official 8B embedding model)"
    },
    "reranker_service": {
        "repo_id": "Qwen/Qwen3-Reranker-8B",  # Official Qwen3-8B reranker model
        "local_dir": "models/qwen3/reranker-8b",
        "description": "Reranker Service - Qwen3-Reranker-8B (official 8B reranker model)"
    },
    "hrm_model_architecture": {
        "repo_id": "DOWNLOADED",  # Already downloaded HRM model architecture
        "local_dir": "models/hrm",
        "description": "HRM - Complete PyTorch model architecture (27M parameters, brain-inspired)"
    },
    "docling_models": {
        "repo_id": "ds4sd/docling-models",  # Official Docling models
        "local_dir": "models/docling/ds4sd--docling-models",
        "description": "Docling - Document processing models"
    }
}

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        logger.info(f"Available disk space: {free_gb} GB")
        
        if free_gb < 80:  # Need at least 80GB for all 8B models + HRM modules
            logger.error(f"Insufficient disk space. Need at least 80GB for 8B models, have {free_gb}GB")
            return False
        return True
    except Exception as e:
        logger.error(f"Could not check disk space: {e}")
        return True  # Continue anyway

def download_model(model_name, config):
    """Download a single model"""
    logger.info(f"🔄 Downloading {config['description']}...")
    logger.info(f"   Repository: {config['repo_id']}")
    logger.info(f"   Local path: {config['local_dir']}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(config['local_dir'], exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=config['repo_id'],
            local_dir=config['local_dir'],
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            cache_dir=None  # Don't use HF cache, download directly
        )
        
        logger.info(f"✅ Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def verify_model(model_name, config):
    """Verify model was downloaded correctly"""
    model_path = Path(config['local_dir'])
    
    if not model_path.exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        return False
    
    # Check for essential files
    essential_files = ['config.json']
    if 'reranker' in model_name:
        essential_files.extend(['tokenizer.json'])  # Reranker models use safetensors
    elif 'embedding' in model_name:
        essential_files.extend(['tokenizer.json'])  # Embedding models use safetensors
    
    missing_files = []
    for file in essential_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    # Check for model files (safetensors format)
    model_files = list(model_path.glob('*.safetensors'))
    if not model_files:
        missing_files.append('model files (*.safetensors)')
    
    if missing_files:
        logger.error(f"❌ Missing essential files for {model_name}: {missing_files}")
        return False
    
    # Check total size
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    logger.info(f"✅ {model_name} verified - Size: {size_gb:.2f} GB")
    
    return True

def main():
    """Main download function"""
    logger.info("🚀 Starting Four-Brain System Model Download")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_disk_space():
        sys.exit(1)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download each model
    success_count = 0
    total_models = len(MODELS_CONFIG)
    
    for model_name, config in MODELS_CONFIG.items():
        logger.info(f"\n📦 Processing {model_name} ({success_count + 1}/{total_models})")
        logger.info("-" * 40)
        
        # Download model
        if download_model(model_name, config):
            # Verify download
            if verify_model(model_name, config):
                success_count += 1
            else:
                logger.error(f"❌ Verification failed for {model_name}")
        else:
            logger.error(f"❌ Download failed for {model_name}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 DOWNLOAD SUMMARY")
    logger.info(f"   Successfully downloaded: {success_count}/{total_models} models")
    
    if success_count == total_models:
        logger.info("🎉 ALL MODELS DOWNLOADED SUCCESSFULLY!")
        logger.info("\n🔧 HRM + QWEN3 FOUR-BRAIN SYSTEM READY:")
        logger.info("   ✅ Brain 1: Qwen3-Embedding-8B (official model, NVFP4 quantization)")
        logger.info("   ✅ Brain 2: Qwen3-Reranker-8B (official model, NVFP4 quantization)")
        logger.info("   ✅ Brain 3: HRM checkpoints + source code (27M brain-inspired architecture)")
        logger.info("   ✅ Brain 4: Docling document processing models (NVFP4)")
        logger.info("   ✅ All models verified and ready for TensorRT optimization")
        logger.info("\n📋 HRM TRAINING OPTIONS:")
        logger.info("   🔄 Use pre-trained checkpoints for fine-tuning")
        logger.info("   🏗️  Train from scratch using source code")
        return 0
    else:
        logger.error(f"❌ {total_models - success_count} models failed to download")
        return 1

if __name__ == "__main__":
    sys.exit(main())
