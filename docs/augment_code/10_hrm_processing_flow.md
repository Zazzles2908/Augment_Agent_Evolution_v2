Title: HRM Processing Flow — Four-Brain Orchestrator with H-Module Planning, L-Module Execution

Goals
- Direct four-brain coordination by HRM H-Module via Orchestrator Hub
- Execute specialized tasks through Brain 1-4 services with on-demand model loading
- Load models on demand via Orchestrator ResourceManager; unload when idle or under pressure

1) Four-Brain Architecture Roles
- **Brain 3 (Intelligence Service)** - HRM Manager:
  - H-Module (always loaded, FP16): Analyze task, choose required brains/models, set constraints
  - L-Module (NVFP4, on demand): Execute fast retrieval and light transformations
  - Orchestrate iterative refinement (decide next step or stop)
- **Brain 1 (Embedding Service)**: Qwen3 8B NVFP4 - semantic embedding generation
- **Brain 2 (Reranker Service)**: Qwen3 8B NVFP4 - retrieval result reranking
- **Brain 4 (Document Processor)**: Docling GPU - document parsing and extraction
- **Orchestrator Hub**: Central coordination, resource management, model loading/unloading

2) Task Analysis → Brain Selection (Orchestrator-Coordinated)
- Classify task type: retrieval, summarization, doc ingestion, QA
- Map to brain services and models, e.g.:
  - Retrieval: Brain 1 (qwen3_embedding_8b_trt) + Brain 2 (qwen3_reranker_8b_trt)
  - Document Ingestion: Brain 4 (docling_gpu) + Brain 1 (qwen3_embedding_8b_trt)
  - Quick lookup: Brain 3 L-Module (hrm_l_trt); Brain 2 optional if budget allows
  - Complex reasoning: Brain 3 H-Module + coordinated multi-brain processing
- Call Orchestrator.ensure_loaded(brain_models) via ResourceManager

3) Orchestrator-Coordinated Iterative Loop
```
while not done and within budget:
    plan = Brain3_H_Module.plan(context)
    needed_brains = plan.required_brain_services
    needed_models = plan.required_models
    Orchestrator.ensure_loaded(needed_models)

    # Coordinate multi-brain execution
    if plan.needs_embedding:
        embedding_result = Brain1.embed(plan.text)
    if plan.needs_reranking:
        rerank_result = Brain2.rerank(plan.candidates)
    if plan.needs_document_processing:
        doc_result = Brain4.process_document(plan.document)

    result = Brain3_L_Module.execute(plan, brain_results)
    Brain3_H_Module.observe(result)
    done = Brain3_H_Module.accept(result) or budget_exhausted()
```
- Evict: Orchestrator ResourceManager may unload least-recently used models between iterations

4) Four-Brain Data Interactions
- **Brain 1 (Embedding)**: Check Redis cache before embedding; if miss, run embed via Triton, store in Supabase pgvector and Redis
- **Brain 2 (Reranker)**: Use pgvector similarity search; reranker refines top-k candidates
- **Brain 4 (Docling)**: Process documents, extract metadata, coordinate with Brain 1 for embedding
- **Orchestrator**: Track usage counts per brain/model to inform prefetch and resource allocation
- **Redis Streams**: Inter-brain communication and task queuing

5) Error Handling & Degradation (Orchestrator-Managed)
- If brain service fails: retry with backoff, fall back to subset (e.g., skip Brain 2 reranker)
- If OOM imminent: Orchestrator reduces batch sizes, unloads idle models, switches to Brain 3 L-Module-only path
- If Triton unhealthy: pause model loading, alert, queue work for when healthy
- Brain service isolation: failure in one brain doesn't cascade to others

6) Telemetry (Multi-Brain Metrics)
- Log per-iteration decisions (selected brains, models, VRAM status, cache hit rate)
- Expose metrics: brain loads/unloads, inter-brain communication delays, failure counts per brain
- Orchestrator dashboard: real-time brain status, resource utilization, task routing

See also: 09_resource_manager_design.md and 11_data_integration_supabase_redis.md.

