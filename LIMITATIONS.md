Known limitations as of 2025-08-18

- TensorRT engines not bundled: You must export ONNX and build FP8/NVFP4 engines before Triton can serve models.
- Windows GPU path depends on WSL2 + NVIDIA Container Toolkit; if unavailable, use CPU or smaller models.
- Supabase local stack not auto-initialized; requires manual `supabase start` and schema migrations.
- API schemas are proposed; ensure they match actual service implementations or generate OpenAPI from code.
- VRAM constraints on 16GB GPU may require single-instance models and dynamic unloading in busy scenarios.

