#!/usr/bin/env python3
"""
CUDA and GPU Details Verification Script
Checks exact CUDA toolkit version, GPU architecture, and memory management capabilities
"""

import sys
import os

def check_cuda_details():
    print("=" * 60)
    print("🔍 CUDA AND GPU SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"✅ Device Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_capability = torch.cuda.get_device_capability(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                print(f"📱 Device {i}:")
                print(f"   Name: {device_name}")
                print(f"   Compute Capability: {device_capability[0]}.{device_capability[1]}")
                print(f"   Total Memory: {memory_total:.2f} GB")
                
                # Check if this is Blackwell (SM 120 = 12.0)
                if device_capability == (12, 0):
                    print(f"   🚀 BLACKWELL ARCHITECTURE DETECTED (SM_120)")
                else:
                    print(f"   ⚠️  Architecture: SM_{device_capability[0]}{device_capability[1]}")
        else:
            print("❌ CUDA not available")
    except ImportError:
        print("❌ PyTorch not available")
    
    # Check TensorRT
    print("\n" + "=" * 60)
    print("🔍 TENSORRT VERIFICATION")
    print("=" * 60)
    
    try:
        import tensorrt as trt
        print(f"✅ TensorRT Version: {trt.__version__}")
        
        # Check if it's the expected version for CUDA 13
        if trt.__version__.startswith("10.13"):
            print("✅ TensorRT 10.13.x - Compatible with CUDA 13")
        else:
            print(f"⚠️  TensorRT {trt.__version__} - May not be optimal for CUDA 13")
            
    except ImportError:
        print("❌ TensorRT not available")
    
    # Check CUDA Toolkit directly
    print("\n" + "=" * 60)
    print("🔍 CUDA TOOLKIT VERIFICATION")
    print("=" * 60)
    
    # Check nvcc version
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"✅ NVCC: {line.strip()}")
        else:
            print("⚠️  nvcc not found in PATH")
    except FileNotFoundError:
        print("⚠️  nvcc not found")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi GPU details:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        name, compute_cap, memory = parts[0], parts[1], parts[2]
                        print(f"   GPU: {name}")
                        print(f"   Compute Capability: {compute_cap}")
                        print(f"   Memory: {memory} MB")
                        
                        # Check for Blackwell
                        if compute_cap.startswith('12.0'):
                            print(f"   🚀 BLACKWELL ARCHITECTURE CONFIRMED (SM_120)")
        else:
            print("⚠️  nvidia-smi not available or failed")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found")
    
    # Check VRAM Management Capabilities
    print("\n" + "=" * 60)
    print("🔍 VRAM MANAGEMENT VERIFICATION")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Get initial memory stats
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_before = torch.cuda.memory_allocated(0)
            reserved_before = torch.cuda.memory_reserved(0)
            
            print(f"✅ Total GPU Memory: {total_memory / (1024**3):.2f} GB")
            print(f"✅ Initially Allocated: {allocated_before / (1024**2):.2f} MB")
            print(f"✅ Initially Reserved: {reserved_before / (1024**2):.2f} MB")
            
            # Test dynamic allocation
            print("\n🧪 Testing Dynamic Memory Allocation...")
            test_tensor = torch.randn(1000, 1000, device=device)
            
            allocated_after = torch.cuda.memory_allocated(0)
            reserved_after = torch.cuda.memory_reserved(0)
            
            print(f"✅ After Allocation: {allocated_after / (1024**2):.2f} MB")
            print(f"✅ After Reserved: {reserved_after / (1024**2):.2f} MB")
            print(f"✅ Dynamic Allocation: {(allocated_after - allocated_before) / (1024**2):.2f} MB")
            
            # Test memory cleanup
            del test_tensor
            torch.cuda.empty_cache()
            
            allocated_cleaned = torch.cuda.memory_allocated(0)
            reserved_cleaned = torch.cuda.memory_reserved(0)
            
            print(f"✅ After Cleanup: {allocated_cleaned / (1024**2):.2f} MB")
            print(f"✅ Memory Management: {'WORKING' if allocated_cleaned < allocated_after else 'ISSUE'}")
            
            # Test memory fraction setting
            print("\n🧪 Testing Memory Fraction Control...")
            try:
                torch.cuda.set_per_process_memory_fraction(0.5, 0)  # 50% of GPU memory
                print("✅ Memory Fraction Control: WORKING")
                torch.cuda.set_per_process_memory_fraction(1.0, 0)  # Reset to 100%
            except Exception as e:
                print(f"⚠️  Memory Fraction Control: {e}")
                
        else:
            print("❌ CUDA not available for VRAM testing")
    except Exception as e:
        print(f"❌ VRAM Management Test Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    check_cuda_details()
