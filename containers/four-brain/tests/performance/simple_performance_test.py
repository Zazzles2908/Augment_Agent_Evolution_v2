#!/usr/bin/env python3
"""
Simple RTX 5070 Ti Performance Test
Tests basic model loading optimizations
"""

import os
import time
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

def test_baseline_loading():
    """Test baseline model loading (current method)."""
    print("🔄 Testing baseline model loading...")
    start_time = time.time()
    
    try:
        # Current method from the existing system
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=True  # Current setting
        )
        
        model_kwargs = {
            'cache_dir': "/workspace/models",
            'trust_remote_code': True,
            'local_files_only': True,
            'torch_dtype': torch.bfloat16,
            'device_map': 'auto',
            'quantization_config': quantization_config
        }
        
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/workspace/models/qwen3/embedding-4b",
            cache_dir="/workspace/models",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("🔄 Loading model with baseline settings...")
        model = AutoModel.from_pretrained("/workspace/models/qwen3/embedding-4b", **model_kwargs)
        
        loading_time = time.time() - start_time
        print(f"✅ Baseline loading completed in {loading_time:.2f} seconds")
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"💾 GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
        
        return {
            'success': True,
            'loading_time': loading_time,
            'model': model,
            'tokenizer': tokenizer
        }
        
    except Exception as e:
        loading_time = time.time() - start_time
        print(f"❌ Baseline loading failed after {loading_time:.2f}s: {e}")
        return {
            'success': False,
            'loading_time': loading_time,
            'error': str(e)
        }

def test_optimized_loading():
    """Test optimized model loading (RTX 5070 Ti optimizations)."""
    print("\n⚡ Testing optimized model loading...")
    start_time = time.time()
    
    try:
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Optimized quantization config (no CPU offloading)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False  # OPTIMIZATION: Keep on GPU
        )
        
        # Optimized model loading arguments
        model_kwargs = {
            'cache_dir': "/workspace/models",
            'trust_remote_code': True,
            'local_files_only': True,
            'torch_dtype': torch.bfloat16,  # Optimal for RTX 5070 Ti
            'device_map': 'auto',
            'quantization_config': quantization_config,
            'low_cpu_mem_usage': True,  # OPTIMIZATION: Reduce CPU usage
            'use_safetensors': True     # OPTIMIZATION: Faster loading
        }
        
        print("📝 Loading tokenizer (optimized)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/workspace/models/qwen3/embedding-4b",
            cache_dir="/workspace/models",
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True  # OPTIMIZATION: Use fast tokenizer
        )
        
        print("⚡ Loading model with RTX 5070 Ti optimizations...")
        model = AutoModel.from_pretrained("/workspace/models/qwen3/embedding-4b", **model_kwargs)
        
        loading_time = time.time() - start_time
        print(f"✅ Optimized loading completed in {loading_time:.2f} seconds")
        
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            utilization = (allocated / total) * 100
            print(f"💾 GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Utilization: {utilization:.1f}%")
        
        return {
            'success': True,
            'loading_time': loading_time,
            'model': model,
            'tokenizer': tokenizer
        }
        
    except Exception as e:
        loading_time = time.time() - start_time
        print(f"❌ Optimized loading failed after {loading_time:.2f}s: {e}")
        return {
            'success': False,
            'loading_time': loading_time,
            'error': str(e)
        }

def test_embedding_generation(model_data):
    """Test embedding generation performance."""
    if not model_data['success']:
        return None
    
    print("\n🧪 Testing embedding generation...")
    start_time = time.time()
    
    try:
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        test_texts = [
            "RTX 5070 Ti optimization test",
            "Performance validation for Four-Brain Architecture",
            "Model loading speed improvement test"
        ]
        
        # Tokenize
        inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Move to GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        generation_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings)} embeddings in {generation_time:.3f} seconds")
        print(f"📏 Embedding dimensions: {embeddings.shape[-1]}")
        
        return {
            'success': True,
            'generation_time': generation_time,
            'embedding_count': len(embeddings),
            'embedding_dim': embeddings.shape[-1]
        }
        
    except Exception as e:
        generation_time = time.time() - start_time
        print(f"❌ Embedding generation failed after {generation_time:.3f}s: {e}")
        return {
            'success': False,
            'generation_time': generation_time,
            'error': str(e)
        }

def main():
    print("🚀 RTX 5070 Ti Simple Performance Test")
    print("=" * 60)
    
    # Test baseline loading
    baseline_result = test_baseline_loading()
    
    # Clear memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test optimized loading
    optimized_result = test_optimized_loading()
    
    # Compare results
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if baseline_result['success'] and optimized_result['success']:
        baseline_time = baseline_result['loading_time']
        optimized_time = optimized_result['loading_time']
        improvement = baseline_time / optimized_time
        reduction = ((baseline_time - optimized_time) / baseline_time) * 100
        
        print(f"📊 Baseline Loading: {baseline_time:.2f} seconds")
        print(f"⚡ Optimized Loading: {optimized_time:.2f} seconds")
        print(f"🚀 Improvement Factor: {improvement:.2f}x faster")
        print(f"📉 Time Reduction: {reduction:.1f}%")
        
        # Test embedding generation with optimized model
        embedding_result = test_embedding_generation(optimized_result)
        
        # Success criteria
        target_time = 30  # seconds
        success_criteria = {
            'under_30_seconds': optimized_time < target_time,
            'improvement_over_2x': improvement > 2.0,
            'embeddings_work': embedding_result and embedding_result['success']
        }
        
        print(f"\n✅ SUCCESS CRITERIA")
        print("=" * 60)
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {criterion.replace('_', ' ').title()}")
        
        all_passed = all(success_criteria.values())
        
        if all_passed:
            print(f"\n🎉 SUCCESS: RTX 5070 Ti optimization working!")
            print(f"⚡ Loading time reduced to {optimized_time:.2f} seconds")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Some optimizations working")
            
    else:
        print("❌ Could not complete performance comparison")
        if not baseline_result['success']:
            print(f"   Baseline failed: {baseline_result.get('error', 'Unknown error')}")
        if not optimized_result['success']:
            print(f"   Optimized failed: {optimized_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
