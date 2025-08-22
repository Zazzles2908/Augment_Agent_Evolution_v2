# API Reference — Orchestrator & Brain Services

This reference standardizes endpoints for client and service integration.

## Orchestrator (FastAPI)
- GET /health → { status }
- POST /chat → { answer, citations[], steps[] }
- POST /memory/search → { results: [{ id, score, snippet, doc_id }] }
- POST /memory/upsert → { upserted: n }
- POST /documents/ingest → { job_id }
- GET /jobs/{id} → { status, result? }

### Request/Response Schemas (JSON)
```json
{
  "chat.request": {"user_id": "uuid", "message": "string", "context": {"household_shared": true}},
  "chat.response": {"answer": "string", "citations": [{"doc_id": "uuid", "chunk_id": "uuid"}], "steps": ["string"]}
}
```

## Embedding Service
- POST /embed → { embedding: number[2000] }

```json
{
  "embed.request": {"text": "string"},
  "embed.response": {"embedding": [0.01, 0.02]}
}
```

## Reranker Service
- POST /rerank → { scores: number[] }

```json
{
  "rerank.request": {"query": "string", "candidates": ["text", "text2"]},
  "rerank.response": {"scores": [0.88, 0.42]}
}
```

## Docling Processor
- POST /parse → { blocks: [...], metadata: {...} }

```json
{
  "parse.request": {"file_url": "string"},
  "parse.response": {"blocks": [{"type": "text", "content": "..."}], "metadata": {"pages": 10}}
}
```

## Triton Control (Orchestrator → Triton)
- POST /v2/repository/models/{name}/load
- POST /v2/repository/models/{name}/unload
- POST /v2/models/{name}/infer

## Error Codes
- 400 Validation error
- 502 Model not ready (load failed or unloaded)
- 503 GPU unavailable / OOM

## Notes
- Dimensions in responses must match configured models
- All endpoints must include `user_id` for RLS-aware operations

