Title: Docling Integration — On-Demand GPU, Doc→Embedding Pipeline

Goals
- Use Docling GPU on demand for parsing/segmenting documents
- Control GPU residency via ResourceManager (approx 2.0GB)
- Pipe parsed text→embedding→storage with metadata tracking

1) Loading Strategy
- Model name: docling (TensorRT or ONNX). Keep unloaded by default
- Load when ingestion tasks arrive; unload when idle and memory needed
- Prefer small batch parsing; back-pressure via queue lengths

2) Pipeline
- Input: document bytes + metadata
- Docling GPU → structured blocks (text/tables)
- Chunker → normalized text chunks
- Embedding via qwen3_4b_embedding
- Persist to Supabase (text, vector, metadata); cache in Redis

3) Metadata
- Track doc_id, source, mime, checksum, page ranges, parsing confidence
- Store parsing stats (time, tokens, errors) for monitoring

4) Failure Handling
- If Docling load fails → retry; else fall back to CPU or defer
- If VRAM pressure rises → unload docling_gpu first (evictable)

5) Monitoring
- Emit metrics: docs/min, avg parse time, GPU residency time, error rate

See also: 11_data_integration_supabase_redis.md and 09_resource_manager_design.md.

