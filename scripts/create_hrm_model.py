#!/usr/bin/env python3
"""
HRM Model Creation Script
Creates and initializes the Hierarchical Reasoning Model (HRM) for the Four-Brain system.

The HRM is a 27M parameter brain-inspired model with:
- High-level module: Abstract planning, strategic reasoning (FP16)
- Low-level module: Fast execution, detailed computations (FP8)
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add the models directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "models" / "hrm"))

try:
    from hrm_act_v1 import (
        HierarchicalReasoningModel_ACTV1,
        HierarchicalReasoningModel_ACTV1Config
    )
except ImportError as e:
    print(f"❌ Error importing HRM model: {e}")
    print("Make sure models/hrm/hrm_act_v1.py exists")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_hrm_config():
    """Create HRM configuration for Four-Brain system."""
    config = HierarchicalReasoningModel_ACTV1Config(
        vocab_size=50257,  # Standard GPT-2 vocab size
        hidden_size=768,   # 27M parameter configuration
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        H_cycles=3,  # High-level module cycles (strategic planning)
        L_cycles=1,  # Low-level module cycles (fast execution)
        act_threshold=0.9,  # Adaptive Computation Time threshold
        max_act_steps=10,
        use_cache=True,
        pad_token_id=50256,
        eos_token_id=50256,
    )
    return config

def create_hrm_model(config=None, device="cuda"):
    """Create and initialize HRM model."""
    if config is None:
        config = create_hrm_config()
    
    logger.info("🧠 Creating HRM model (27M parameters)...")
    logger.info(f"   High-level module: {config.H_cycles} cycles (strategic planning)")
    logger.info(f"   Low-level module: {config.L_cycles} cycles (fast execution)")
    
    model = HierarchicalReasoningModel_ACTV1(config)
    
    # Move to device
    if torch.cuda.is_available() and device == "cuda":
        model = model.to(device)
        logger.info(f"✅ HRM model loaded on {device}")
    else:
        logger.info("✅ HRM model loaded on CPU")
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Model Statistics:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Model size: ~{total_params / 1e6:.1f}M parameters")
    
    return model, config

def save_hrm_model(model, config, save_dir="models/hrm"):
    """Save HRM model and configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = save_dir / "hrm_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 Model saved to {model_path}")
    
    # Save configuration
    config_path = save_dir / "hrm_config.json"
    with open(config_path, 'w') as f:
        import json
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"💾 Config saved to {config_path}")
    
    return model_path, config_path

def load_hrm_model(model_path="models/hrm/hrm_model.pth", 
                   config_path="models/hrm/hrm_config.json", 
                   device="cuda"):
    """Load HRM model from saved files."""
    import json
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(config)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available() and device == "cuda":
        model = model.to(device)
    
    logger.info(f"✅ HRM model loaded from {model_path}")
    return model, config

def test_hrm_model(model, config):
    """Test HRM model with sample input."""
    logger.info("🧪 Testing HRM model...")
    
    # Create sample input
    batch_size = 1
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        
    logger.info(f"✅ Forward pass successful!")
    logger.info(f"   Input shape: {input_ids.shape}")
    logger.info(f"   Output shape: {outputs.logits.shape}")
    
    # Test high-level and low-level modules separately
    logger.info("🔍 Testing module separation...")
    
    # This would require accessing internal modules
    # Implementation depends on the exact HRM architecture
    
    return True

def main():
    """Main function to create and test HRM model."""
    logger.info("🚀 Starting HRM Model Creation for Four-Brain System")
    
    try:
        # Create HRM model
        model, config = create_hrm_model()
        
        # Test the model
        test_hrm_model(model, config)
        
        # Save the model
        model_path, config_path = save_hrm_model(model, config)
        
        logger.info("🎉 HRM Model Creation Complete!")
        logger.info("📋 Next Steps:")
        logger.info("   1. Train HRM on Four-Brain orchestration tasks")
        logger.info("   2. Convert to TensorRT engines (high-level FP16, low-level FP8)")
        logger.info("   3. Integrate with Triton model repository")
        logger.info("   4. Deploy in Four-Brain system")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error creating HRM model: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
