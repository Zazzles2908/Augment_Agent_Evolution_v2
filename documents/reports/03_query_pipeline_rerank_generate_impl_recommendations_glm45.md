# GLM-4.5 Implementation Recommendations – 03_query_pipeline_rerank_generate.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Orchestration Flow (Sequence)
```
Query → [Embed Qwen3-4B FP8] → [Vector Search Supabase] → [Rerank Qwen3-0.6B NVFP4] → [Generate GLM-4.5 Air NVFP4] → Response
```
- Embed + search can be batched where applicable; collector pattern for async ops
- Early termination on empty search results; fallback to raw search if rerank fails

## 2. Model Invocation Patterns (Timeouts, Batching, Retries)
- Embedding: 5s timeout; 2 retries (exponential backoff); batch ≤32
- Vector search: 3s timeout; 1 retry; single-query RPC
- Reranking: 8s timeout; 2 retries; batch ≤10 query–candidate pairs
- Generation: 15s timeout; 1 retry; sequential for quality

## 3. Reranking Strategy & Scoring
- Take top-20 from vector search; normalise scores to [0,1]
- Filter below 0.3 relevance; combine vector (0.7) + rerank (0.3)
- Keep top-5 for generation context

## 4. Generation Prompts & Citation Format
Prompt template:
```
Context: {reranked_context}
Query: {user_query}

Instructions:
1. Answer using only the context
2. Include citations in [citation:CHUNK_ID]
3. If insufficient context, say so
4. ≤300 words

Answer:
```
- Footer format: Sources: [chunk_id_1], [chunk_id_2], [chunk_id_3]; include metadata if available

## 5. Caching & Idempotency
- Cache: embeddings 24h; search results 1h; final responses 30m
- Keys: hash(query)+versioned model IDs; dedupe at embedding stage
- Use idempotency keys for critical operations

## 6. Latency Targets & Backpressure
- Targets (approx.): embed <100ms; search <200ms; rerank <300ms; generate <800ms; e2e <1.5s (P95)
- Backpressure: request queue with priorities; circuit breakers per model; use cached responses under load; graceful degradation

