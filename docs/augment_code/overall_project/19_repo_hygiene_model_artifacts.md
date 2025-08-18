# Repository Hygiene — Triton Model Artifacts

Goal
- Keep config.pbtxt versioned; avoid committing weights (.onnx) and engines (.plan)

Action (draft; do not run unless approved)
```
# Create clean branch for audit cleanup
git checkout -b audit/cleanup-artifacts-2025-08-16

# Remove committed artifacts from index (preserve files locally)
git rm -r --cached "containers/four-brain/triton/model_repository/**/1/*"

# If specific tracked files persist, remove them explicitly (examples)
# git rm --cached containers/four-brain/triton/model_repository/qwen3_embedding_trt/1/model.plan
# git rm --cached containers/four-brain/triton/model_repository/qwen3_embedding/1/model.onnx

# Commit
git commit -m "audit: purge model artifacts from repo; keep configs only"
```

.gitignore check
- Ensure these patterns are present:
```
containers/four-brain/triton/model_repository/**/1/*
!.*/config.pbtxt
```

Notes
- Engines should be built in Ubuntu 24.04 (container/WSL) with TensorRT 10.13.x matching Triton image
- Prefer building inside the same Triton base image to avoid ABI mismatch
- Use local cache directories for large artifacts; never commit them

