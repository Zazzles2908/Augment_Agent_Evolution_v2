# Systematic Rebuild Plan - Phases 5-7 Implementation Guide

**Date:** August 19, 2025  
**Project:** Qwen3 & Docling Model Rebuild with Triton Integration  
**Prerequisites:** Phases 1-4 Complete, TensorRT 10.13.2 + Model Optimizer Ready

---

## IMPLEMENTATION OVERVIEW

```
Execution Flow:
Phase 4 (Complete) --> Phase 5 (Qwen3) --> Phase 6 (Docling) --> Phase 7 (Integration)
                           |                    |                      |
                       FP8 + NVFP4          NVFP4              Triton + Validation
```

---

## PHASE 5: QWEN3 MODEL REBUILD

### Phase 5A: Environment & Source Verification

**Task 5A.1: Verify Qwen3 ONNX Model Files**
```bash
# Check for Qwen3 source models
ls -la containers/four-brain/triton/model_repository/qwen3_*/
find . -name "*qwen3*.onnx" -type f
```

**Task 5A.2: Check Preserved TRT Engine Integrity**
```bash
# Verify backup integrity
ls -la qwen3_embedding_trt_backup.plan
file qwen3_embedding_trt_backup.plan
```

**Task 5A.3: Prepare Model Optimizer Scripts**
```bash
# Create Qwen3-specific compilation scripts
mkdir -p scripts/qwen3
cp scripts/hrm/compile_modelopt.py scripts/qwen3/
# Adapt for 8B model architecture
```

**Task 5A.4: Test VRAM Availability**
```bash
# Monitor baseline VRAM usage
docker run --rm --gpus all tensorrt-modelopt:latest nvidia-smi
```

### Phase 5B: Qwen3 8B Embedding (FP8)

**Task 5B.1: Create FP8 Quantization Script**
```bash
# Adapt compile_modelopt.py for Qwen3 8B embedding
# Configure FP8 precision settings
# Add memory optimization flags
```

**Task 5B.2: Compile with Memory Optimization**
```bash
# Primary compilation attempt
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 scripts/qwen3/compile_modelopt.py \
  --model qwen3_embedding_8b.onnx \
  --precision fp8 \
  --out models/qwen3_embedding_8b_fp8.pt \
  --batch-size 1 \
  --seq-len 512

# Fallback: CPU compilation if VRAM insufficient
docker run --rm -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 scripts/qwen3/compile_modelopt.py \
  --model qwen3_embedding_8b.onnx \
  --precision fp8 \
  --out models/qwen3_embedding_8b_fp8.pt \
  --device cpu
```

**Task 5B.3: Validate FP8 Precision**
```bash
# Check model file and quantization
ls -la models/qwen3_embedding_8b_fp8.pt
python3 -c "import torch; m=torch.load('models/qwen3_embedding_8b_fp8.pt'); print(f'Precision: {m[\"precision\"]}, Quantized: {m.get(\"quantized\", False)}')"
```

**Task 5B.4: Test Model Loading**
```bash
# Basic inference test
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 -c "
import torch
model_data = torch.load('models/qwen3_embedding_8b_fp8.pt')
print('Model loaded successfully')
print(f'Model size: {len(str(model_data))} bytes')
"
```

### Phase 5C: Qwen3 8B Reranker (NVFP4) + TRT Restoration

**Task 5C.1: Compile 8B Reranker (NVFP4)**
```bash
# Use proven NVFP4 pipeline from HRM
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 scripts/qwen3/compile_modelopt.py \
  --model qwen3_reranker_8b.onnx \
  --precision nvfp4 \
  --out models/qwen3_reranker_8b_nvfp4.pt \
  --batch-size 1 \
  --seq-len 512
```

**Task 5C.2: Restore Preserved TRT Engine**
```bash
# Restore working TRT engine
mkdir -p containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/
cp qwen3_embedding_trt_backup.plan containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/model.plan
```

**Task 5C.3: Verify Restored TRT Engine**
```bash
# Check file integrity
ls -la containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/model.plan
file containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/model.plan
```

**Task 5C.4: Validate NVFP4 Reranker**
```bash
# Verify NVFP4 quantization
ls -la models/qwen3_reranker_8b_nvfp4.pt
python3 -c "import torch; m=torch.load('models/qwen3_reranker_8b_nvfp4.pt'); print(f'Precision: {m[\"precision\"]}, Quantized: {m.get(\"quantized\", False)}')"
```

### Phase 5D: Qwen3 Models Validation

**Task 5D.1: Individual Model Loading Tests**
```bash
# Test each model individually
for model in qwen3_embedding_8b_fp8.pt qwen3_reranker_8b_nvfp4.pt; do
  echo "Testing $model..."
  docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
    python3 -c "import torch; torch.load('models/$model'); print('$model: OK')"
done
```

**Task 5D.2: Memory Usage Profiling**
```bash
# Monitor VRAM usage during model loading
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1 &
# Load models and monitor
```

**Task 5D.3: Basic Inference Validation**
```bash
# Create simple inference test
python3 scripts/test_qwen3_inference.py
```

**Task 5D.4: Prepare for Triton Integration**
```bash
# Verify Triton model repository structure
ls -la containers/four-brain/triton/model_repository/
```

---

## PHASE 6: DOCLING MODEL REBUILD

### Phase 6A: Docling Model Compilation

**Task 6A.1: Locate Docling ONNX Model**
```bash
# Find Docling model files
find . -name "*docling*.onnx" -type f
ls -la containers/four-brain/triton/model_repository/docling*/
```

**Task 6A.2: Adapt Compilation Script**
```bash
# Create Docling-specific script
mkdir -p scripts/docling
cp scripts/hrm/compile_modelopt.py scripts/docling/
# Adapt for Docling architecture
```

**Task 6A.3: Compile with NVFP4 Precision**
```bash
# Use proven NVFP4 pipeline
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 scripts/docling/compile_modelopt.py \
  --model docling.onnx \
  --precision nvfp4 \
  --out models/docling_nvfp4.pt
```

**Task 6A.4: Validate NVFP4 Quantization**
```bash
# Check quantization success
ls -la models/docling_nvfp4.pt
python3 -c "import torch; m=torch.load('models/docling_nvfp4.pt'); print(f'Precision: {m[\"precision\"]}, Quantized: {m.get(\"quantized\", False)}')"
```

### Phase 6B: Docling Validation

**Task 6B.1: Test Model Loading**
```bash
# Basic loading test
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace tensorrt-modelopt:latest \
  python3 -c "import torch; torch.load('models/docling_nvfp4.pt'); print('Docling model loaded successfully')"
```

**Task 6B.2: Memory Usage Profiling**
```bash
# Monitor VRAM during Docling loading
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

**Task 6B.3: Prepare Triton Configuration**
```bash
# Verify Triton config for Docling
ls -la containers/four-brain/triton/model_repository/docling*/config.pbtxt
```

---

## PHASE 7: INTEGRATION, VALIDATION & PERFORMANCE TESTING

### Phase 7A: Incremental Triton Integration

**Task 7A.1: Configure Triton Model Repository**
```bash
# Prepare model repository structure
docker compose -f containers/four-brain/docker/docker-compose.yml up -d triton
curl http://localhost:8000/v2/health/ready
```

**Task 7A.2: Load Current Models**
```bash
# Load current Triton models
curl -X POST http://localhost:8000/v2/repository/models/qwen3_4b_embedding/load
curl -X POST http://localhost:8000/v2/repository/models/qwen3_0_6b_reranking/load
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

**Task 7A.3: Verify Model Load State**
```bash
# Confirm both models are READY
curl -s http://localhost:8000/v2/models/qwen3_4b_embedding | jq
curl -s http://localhost:8000/v2/models/qwen3_0_6b_reranking | jq
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

**Task 7A.4: Add Docling Model**
```bash
# Load Docling model
curl -X POST http://localhost:8000/v2/repository/models/docling/load
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

**Task 7A.5: Monitor VRAM Usage**
```bash
# Continuous VRAM monitoring (<75% threshold = 12GB)
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk "{printf \"VRAM: %.1f/%.1f GB (%.1f%%)\n\", \$1/1024, \$2/1024, (\$1/\$2)*100}"'
```

### Phase 7B: Performance Validation

**Task 7B.1: Individual Model Inference Testing**
```bash
# Test each model endpoint
curl -X POST http://localhost:8000/v2/models/qwen3_4b_embedding/infer -d @test_data/qwen3_embed_input.json
curl -X POST http://localhost:8000/v2/models/qwen3_0_6b_reranking/infer -d @test_data/qwen3_rerank_input.json
curl -X POST http://localhost:8000/v2/models/docling/infer -d @test_data/docling_input.json
```

**Task 7B.2: Multi-Model Concurrent Testing**
```bash
# Concurrent inference testing
python3 scripts/test_concurrent_inference.py
```

**Task 7B.3: Performance Benchmarking**
```bash
# Response time validation (<500ms target)
python3 scripts/benchmark_response_times.py
```

**Task 7B.4: VRAM Optimization**
```bash
# Final VRAM optimization
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Phase 7C: Final System Validation

**Task 7C.1: Complete System Health Checks**
```bash
# Comprehensive health validation
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models
```

**Task 7C.2: Evidence Capture**
```bash
# Capture final system state
nvidia-smi > final_system_state.txt
docker ps >> final_system_state.txt
curl http://localhost:8000/v2/models >> final_system_state.txt
```

**Task 7C.3: Cleanup and Finalization**
```bash
# Clean up temporary files
rm -rf /tmp/tensorrt_*
docker system prune -f
```

**Task 7C.4: Final Validation Report**
```bash
# Generate completion report
python3 scripts/generate_final_report.py
```

---

## SUCCESS CRITERIA CHECKLIST

- [ ] All Qwen3 models compiled (FP8 embedding, NVFP4 reranker, preserved TRT)
- [ ] Docling model compiled (NVFP4)
- [ ] All models loaded in Triton successfully
- [ ] VRAM usage <75% (12GB) with all models active
- [ ] Response times <500ms for all endpoints
- [ ] System health checks passing
- [ ] Performance benchmarks within acceptable ranges

---

*Implementation Guide Complete - Ready for Execution*
