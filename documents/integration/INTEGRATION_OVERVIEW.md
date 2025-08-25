# Integration Overview

This document explains how the Main System and the Sub‑systems integrate and communicate. It is the canonical high‑level guide for integration.

Main System (Local)
- Docling → Chunking → Triton (Qwen3‑4B embeddings, 2000‑dim) → Supabase (pgvector) → Triton (Qwen3‑0.6B reranker) → Triton (GLM‑4.5 Air) → Response
- Redis TTL caching for embeddings and reranked results
- TensorRT 10.13.x on CUDA 13, Ubuntu 24.04; Triton explicit model control

Sub‑systems
- Zen MCP Server: external assistant tools and project automation; 16 public tools, helpers hidden
- Supabase MCP Server (planned): encapsulates Supabase vector operations and agent message similarity behind MCP tools

Contracts & Data Model
- Embeddings: vector(2000) in pgvector (MRL embeddings)
- RPC: match_documents(query_embedding vector(2000), match_count int, similarity_threshold float)
- Tables: augment_agent.document_vectors (doc_id, chunk_id, title, text_excerpt, embedding, metadata)

Integration Flow
1) Ingestion uses Triton (Qwen3‑4B) to produce 2000‑d embeddings and upserts rows into Supabase
2) Query path obtains cached or fresh embedding; calls Supabase for vector similarity; reranks with Qwen3‑0.6B; generates with GLM‑4.5 Air
3) All Supabase interactions are centralized via Supabase MCP tools (once implemented)

Security & Performance
- Service role key is server‑side only; RLS enforced; PostgREST RPC used
- Redis used for cache; HNSW index on vector(2000) for fast ANN search
- End‑to‑end P95 target < 1.5s

Validation
- Auggie CLI self‑check for MCPs; integration smoke tests execute MCP tools for vector upsert & match

