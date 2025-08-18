# Personal AI Assistant System — High-Level Architecture

## Purpose
- Provide a production-ready, modular architecture for a two-user household assistant (you and your wife) that can be deployed locally on Windows 11 + Docker Desktop with GPU acceleration.
- Unify planning (HRM), knowledge (Supabase + pgvector), document intelligence (Docling), and retrieval quality (embedding + re-ranking) behind a single orchestrated interface.

## Core Principles
- Human-in-the-loop by design: assistants surface reasoning summaries; high-risk actions require explicit approval.
- Resource-aware: dynamic loading/unloading of heavy models via Triton to fit 16GB VRAM.
- Privacy-first: data remains local by default; optional cloud sync is opt-in with clear scopes.

## System Context Diagram (Mermaid)
```mermaid
flowchart LR
  Users[You & Wife]
  UI[Client(s): Desktop, Browser, Mobile]
  API[Orchestrator Hub (FastAPI)]
  HRM[HRM Core (H-Module/L-Module)]
  EMB[Embedding Service (Qwen3-8B Embeddings)]
  RER[Re-ranking Service (Qwen3-8B Reranker)]
  DOC[Docling Processor]
  TRT[Triton Inference Server (TensorRT Engines)]
  PG[(Supabase / Postgres + pgvector)]
  REDIS[(Redis Streams/Cache)]

  Users --> UI --> API
  API --> HRM
  HRM <--> EMB
  HRM <--> RER
  HRM <--> DOC
  API <--> TRT
  HRM <--> TRT
  HRM <--> REDIS
  HRM <--> PG
```

## Components
- Orchestrator Hub
  - Single entrypoint and brain router. Manages resource budgets, VRAM load, and workflow steps. Exposes REST for clients.
- HRM Core (Human Resource Manager)
  - H-Module (always-on, lightweight): decomposes goals, chooses which brains to use, sets constraints.
  - L-Module (on-demand, faster/quantized): executes short hops, light reasoning, and tool calls.
- Embedding Service
  - Qwen3-8B Embedding model optimized with TensorRT (FP8). Produces dense vectors stored in pgvector.
- Re-ranking Service
  - Qwen3-8B Reranker optimized with TensorRT (NVFP4). Improves ranking of retrieved candidates.
- Docling Processor
  - GPU-accelerated Docling for PDF → Markdown/JSON, metadata extraction, and chunking.
- Triton Inference Server
  - Hosts TensorRT engines for models with dynamic batching and explicit load/unload control.
- Data Layer
  - Supabase (Postgres + pgvector) for long-term semantic memory (documents, conversations, tasks).
  - Redis for working memory, ephemeral queues, and rate control.

## Logical Flow
1) A user asks a question/task through the client.
2) Orchestrator forwards to HRM. H-Module plans steps and selects brains.
3) HRM queries Supabase (semantic memory) using embeddings for retrieval.
4) Re-ranking service reorders retrieved candidates.
5) If documents are involved, Docling parses and extracts structured data.
6) HRM L-Module iterates (bounded) until the plan converges.
7) Orchestrator returns the final answer plus citations, actions, and next-step suggestions.

## Memory Model
- Working memory (Redis): task queues, tool call traces, HRM iteration state.
- Episodic memory (Supabase): conversation transcripts with metadata and access control per user.
- Semantic memory (Supabase + pgvector): document chunks and entity knowledge.
- Personalization layer: per-user embeddings, preferences, calendars, and routines with RLS policies (see database_architecture.md).

## Operational Characteristics
- VRAM-constrained: single-instance engines with dynamic batching; unload idle models.
- Observability: Prometheus/Grafana integration for GPU/CPU/memory and app metrics; Loki for logs.
- Safety gates: configurable HRM approval thresholds for actions (scheduler updates, reminders, purchases).

## Deployment Targets
- Local: Windows 11 + Docker Desktop (WSL2 backend) with NVIDIA GPU passthrough.
- Optional Edge: Mini-PC/NAS running Linux Docker for 24/7 home assistant.

## Multi-User (You & Wife)
- Separate Supabase roles and RLS policies; shared household collections with opt-in write permissions.
- Per-user profile vectors; shared context is tagged and filtered during retrieval.

## Interfaces
- REST endpoints (Orchestrator): /chat, /tasks, /memory/search, /memory/upsert, /documents/ingest
- Tool adapters: calendar, email, to-do, file system, Home Assistant (optional), with HRM approval gates.

## Risks & Mitigations
- Heavy models on 16GB VRAM → mitigate via FP8/NVFP4, single instance, unload, or switch to 4B variants.
- Schema drift → manage via migrations; test dimension matches embedding output before enabling writes.
- Windows GPU passthrough → validate WSL2 + NVIDIA Container Toolkit; provide fallbacks (CPU/smaller models).

## What’s Next
- See inference_stack.md for TensorRT/Triton setup
- See database_architecture.md for Supabase schema & pgvector
- See deployment_guide.md for Windows-first steps
- See api_reference.md for REST contracts

