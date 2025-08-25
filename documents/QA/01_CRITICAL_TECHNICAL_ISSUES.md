# Critical Technical Issues - Detailed Analysis
**Date:** 2025-08-21  
**Priority:** BLOCKING  

## Issue #1: Embedding Dimension Mismatch
**Severity:** RESOLVED
**Status:** FIXED (Aligned to 2000-dim)

### Resolution
- Triton model `qwen3_4b_embedding` now outputs 2000-dim embeddings (TYPE_FP32) per config.pbtxt.
- Embedding service defaults to Triton model name `qwen3_4b_embedding`.
- Docs and examples updated to reflect 2000-dim vectors for Supabase pgvector.

### Evidence
- containers/four-brain/triton/model_repository/qwen3_4b_embedding/config.pbtxt shows `dims: [ -1, 2000 ]`.
- containers/four-brain/src/brains/embedding_service/config/settings.py sets default TRITON_MODEL_NAME to `qwen3_4b_embedding`.
- documents/augment_code/02_qwen3_embedding_request_examples.md updated to `qwen3_4b_embedding`.

---

## Issue #2: Missing Docling Integration
**Severity:** CRITICAL  
**Status:** CORE COMPONENT MISSING  

### Problem
Stack documentation emphasizes Docling for document extraction and chunking, but implementation only contains placeholder code.

### Evidence
```python
# In examples/end_to_end_demo.py:
# 1) Docling conversion placeholder (you can call your Docling pipeline here)
doc_text = Path(args.document).read_text(errors="ignore")[:4000]
```

### Impact
- Document processing pipeline incomplete
- Cannot handle complex document formats
- Chunking strategy not implemented
- Metadata extraction missing

### Fix Required
1. Implement actual Docling integration
2. Add document format support (PDF, DOCX, etc.)
3. Implement proper chunking strategy
4. Add metadata extraction and storage

---

## Issue #3: Legacy Model Cleanup Incomplete
**Severity:** HIGH  
**Status:** VIOLATES PHASE 0 REQUIREMENTS  

### Problem
Phase 0 requirements specify cleaning up legacy models, but multiple legacy artifacts remain.

### Evidence
Found in `containers/four-brain/triton/model_repository/`:
- `hrm_h_trt/`
- `hrm_high_fp16/`
- `hrm_l/`
- `hrm_l_trt/`
- `hrm_low_fp8/`
- `brain1_embedding_fp8/`
- `brain2_reranker_nvfp4/`

### Impact
- Violates documented Phase 0 cleanup requirements
- Increases storage requirements
- Creates confusion about which models to use
- May cause deployment conflicts

### Fix Required
1. Remove all hrm_* model directories
2. Remove all brain1_*, brain2_* model directories
3. Keep only: qwen3_4b_embedding, qwen3_0_6b_reranking, glm45_air
4. Update any references in scripts/configs

---

## Issue #4: CUDA Version Inconsistency
**Severity:** HIGH  
**Status:** ENVIRONMENT SETUP CONFLICT  

### Problem
Conflicting CUDA version requirements between documentation sources.

### Evidence
- **BUILD_INSTRUCTIONS.md:** "CUDA 13 (nvidia-smi shows CUDA Version: 13.x)"
- **Stack docs:** "CUDA 12.x, TensorRT 10.x"
- **Docker configs:** References to CUDA 13.0

### Impact
- Environment setup confusion
- Potential compatibility issues
- Build failures on different CUDA versions
- Inconsistent deployment environments

### Fix Required
1. Standardize on single CUDA version across all documentation
2. Update BUILD_INSTRUCTIONS.md to match stack requirements
3. Verify TensorRT compatibility with chosen CUDA version
4. Test build process on standardized environment

---

## Issue #5: Missing Monitoring Infrastructure
**Severity:** HIGH  
**Status:** OBSERVABILITY GAP  

### Problem
Stack documentation specifies comprehensive monitoring (Prometheus + Grafana + Loki + Alloy), but implementation is incomplete.

### Evidence
- Docker-compose includes monitoring services but configs missing
- No Prometheus configuration for Triton metrics scraping
- No Grafana dashboards for GPU/latency/throughput
- No Loki log aggregation setup

### Impact
- Cannot monitor system performance
- No visibility into GPU utilization
- Difficult to debug issues
- No alerting for failures

### Fix Required
1. Implement Prometheus configuration for Triton :8002/metrics
2. Create Grafana dashboards for GPU, latency, throughput
3. Configure Loki + Alloy for log aggregation
4. Set up alerts for failures and GPU pressure

---

## Issue #6: Security Vulnerabilities
**Severity:** HIGH  
**Status:** DATA PROTECTION RISK  

### Problem
Multiple security issues identified in current implementation.

### Evidence
1. **Plain text credentials in config files**
2. **No input validation in text processing**
3. **Insecure Redis configuration (no auth/encryption)**
4. **Missing rate limiting and access controls**

### Impact
- Credential exposure risk
- Injection attack vulnerability
- Unauthorized data access
- Service abuse potential

### Fix Required
1. Implement secure credential management
2. Add input validation and sanitization
3. Configure Redis authentication and encryption
4. Implement rate limiting and access controls

---

## Immediate Action Items
1. **Fix embedding dimensions** - Update Triton config and rebuild engines
2. **Implement Docling** - Replace placeholder with actual integration
3. **Clean legacy models** - Remove hrm_* and brain* directories
4. **Standardize CUDA** - Update all documentation to consistent version
5. **Security audit** - Address credential and validation issues

## Testing Requirements
Each fix must include:
- Unit tests for the specific component
- Integration tests for end-to-end pipeline
- Performance validation on target hardware
- Security validation for credential handling
