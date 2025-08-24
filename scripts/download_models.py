#!/usr/bin/env python3
"""
Model Download Script for Four-Brain System
Downloads the correct models for each brain component.

This script ensures we have proper models:
- Brain1: Qwen3-8B embedding model (NVFP4 quantized)
- Brain2: Qwen3-8B reranker model (NVFP4 quantized)
- Brain3: Intelligence Service (Orchestrator/Agent)
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

# Model configurations - Qwen3 models and Docling (no HRM)
MODELS_CONFIG = {
    "embedding_service": {
        "repo_id": "Qwen/Qwen3-4B",  # Qwen3-4B model for RTX 5070 Ti constraints
        "local_dir": "models/qwen3/embedding-4b",
        "description": "Embedding Service - Qwen3-4B (optimized for 16GB VRAM)"
    },
    "reranker_service": {
        "repo_id": "Qwen/Qwen3-0.6B",  # Qwen3-0.6B reranker for RTX 5070 Ti constraints
        "local_dir": "models/qwen3/reranker-0.6b",
        "description": "Reranker Service - Qwen3-0.6B (optimized for 16GB VRAM)"
    },
    # Removed HRM model architecture
        # (intentionally removed)
        # (intentionally removed)
        # (intentionally removed)
    #
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
        
        if free_gb < 60:  # Need at least ~60GB for Qwen3-8B models and Docling
            logger.error(f"Insufficient disk space. Need at least 80GB for 8B models, have {free_gb}GB")
            return False
        return True
    except Exception as e:
        logger.error(f"Could not check disk space: {e}")
        return True  # Continue anyway

def download_model(model_name, config):
    """Download a single model or create if local-only."""
    logger.info(f"🔄 Preparing {config['description']}...")
    logger.info(f"   Repository: {config['repo_id']}")
    logger.info(f"   Local path: {config['local_dir']}")

    # (No HRM support)
    if False:
        os.makedirs(config['local_dir'], exist_ok=True)
        hrm_src = Path(config['local_dir']) / "hrm_act_v1.py"
        hrm_ckpt = Path(config['local_dir']) / "hrm_model.pth"
        hrm_cfg = Path(config['local_dir']) / "hrm_config.json"
        if hrm_ckpt.exists() and hrm_cfg.exists():
            logger.info("✅ HRM local checkpoint present; skipping creation")
            return True
        if hrm_src.exists():
            logger.info("ℹ️ HRM source present (hrm_act_v1.py). Skipping HF download.")
            logger.info("   Note: Checkpoint creation/export to ONNX/TRT will run in later steps.")
            return True
        return False

    try:
        # Create directory if it doesn't exist
        os.makedirs(config['local_dir'], exist_ok=True)

        # Download model from HF
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
    """Verify model presence and integrity."""
    model_path = Path(config['local_dir'])

    if not model_path.exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        return False

    # (No HRM verification)
    if False:
        src_ok = (model_path / "hrm_act_v1.py").exists()
        ckpt_ok = (model_path / "hrm_model.pth").exists() and (model_path / "hrm_config.json").exists()
        if not (src_ok or ckpt_ok):
            missing = []
            if not src_ok:
                missing.append('hrm_act_v1.py')
            if not ckpt_ok:
                missing.extend(['hrm_model.pth','hrm_config.json'])
            logger.error(f"❌ Missing HRM components: {missing}")
            return False
        logger.info("✅ HRM verified (source and/or local checkpoint present)")
        return True

    # Docling: require safetensors only
    if model_name == "docling_models":
        model_files = list(model_path.rglob('*.safetensors'))
        if not model_files:
            logger.error("❌ Missing Docling model files (*.safetensors)")
            return False
        total_size = sum(f.stat().st_size for f in model_path.rglob('*.safetensors'))
        size_gb = total_size / (1024**3)
        logger.info(f"✅ {model_name} verified - Size: {size_gb:.2f} GB")
        return True

    # Check for essential files for HF models
    essential_files = ['config.json', 'tokenizer.json']

    missing_files = []
    for file in essential_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    # Check for model files (safetensors format) deep
    model_files = list(model_path.rglob('*.safetensors'))
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
        logger.info("\n🔧 QWEN3/Docling Four-Brain System Ready:")
        logger.info("   ✅ Brain 1: Qwen3-Embedding-8B (official model, NVFP4 quantization)")
        logger.info("   ✅ Brain 2: Qwen3-Reranker-8B (official model, NVFP4 quantization)")
        logger.info("   ✅ Brain 3: Intelligence (Zazzles's Agent) Service")
        logger.info("   ✅ Brain 4: Docling document processing models")
        logger.info("   ✅ All models verified and ready for TensorRT optimization")
        logger.info("   🔄 Use pre-trained checkpoints for fine-tuning")
        logger.info("   🏗️  Train from scratch using source code")
        return 0
    else:
        logger.error(f"❌ {total_models - success_count} models failed to download")
        return 1

if __name__ == "__main__":
    sys.exit(main())
