# Database Architecture — Supabase + pgvector

This document defines the memory layer: long-term semantic memory and episodic memory for two users.

## Goals
- Store document chunks and conversation context with embeddings
- Support per-user privacy via RLS
- Provide fast similarity search using pgvector

## Schema (SQL)
```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Use dedicated schema
CREATE SCHEMA IF NOT EXISTS augment_agent;

-- Users (Supabase auth.users is preferred; mirror minimal profile in augment_agent.user_profiles)
CREATE TABLE IF NOT EXISTS augment_agent.user_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  display_name text,
  preferences jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Documents & Vectors (aligns with existing augment_agent.*)
CREATE TABLE IF NOT EXISTS augment_agent.documents (
  id uuid PRIMARY KEY,
  user_id uuid NOT NULL,
  filename TEXT NOT NULL,
  file_size BIGINT,
  mime_type TEXT,
  document_type TEXT,
  content_text TEXT,
  processing_status TEXT DEFAULT 'pending',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  upload_timestamp TIMESTAMPTZ DEFAULT NOW(),
  processing_timestamp TIMESTAMPTZ NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS augment_agent.document_vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id UUID NOT NULL REFERENCES augment_agent.documents(id) ON DELETE CASCADE,
  chunk_id TEXT NOT NULL,
  page_no INT,
  title TEXT,
  text_excerpt TEXT,
  embedding VECTOR(2000) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chats (sessions and messages)
CREATE TABLE IF NOT EXISTS augment_agent.chat_sessions (
  id uuid PRIMARY KEY,
  user_id uuid NOT NULL,
  session_title text,
  session_context jsonb DEFAULT '{}'::jsonb,
  message_count int DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  last_activity timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS augment_agent.chat_messages (
  id uuid PRIMARY KEY,
  session_id uuid NOT NULL REFERENCES augment_agent.chat_sessions(id) ON DELETE CASCADE,
  user_id uuid NOT NULL,
  message_type varchar(32) NOT NULL, -- 'user' | 'assistant' | 'system' | tool
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  task_id varchar(64),
  created_at timestamptz DEFAULT now()
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_docvec_embedding ON augment_agent.document_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_user ON augment_agent.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON augment_agent.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON augment_agent.chat_messages(session_id);
```

> Note: Supabase vector length is capped at 2000. This project targets 2000-dim embeddings via MRL truncation in the embedding service; do not exceed 2000.

## RLS Policies (Supabase)
```sql
-- Enable RLS on augment_agent tables
ALTER TABLE augment_agent.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE augment_agent.document_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE augment_agent.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE augment_agent.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE augment_agent.user_profiles ENABLE ROW LEVEL SECURITY;

-- Owner can read/write their rows
CREATE POLICY owner_rw_documents ON augment_agent.documents
  FOR ALL USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY owner_rw_docvecs ON augment_agent.document_vectors
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM augment_agent.documents d WHERE d.id = doc_id AND d.user_id = auth.uid()
    )
  ) WITH CHECK (
    EXISTS (
      SELECT 1 FROM augment_agent.documents d WHERE d.id = doc_id AND d.user_id = auth.uid()
    )
  );
CREATE POLICY owner_rw_sessions ON augment_agent.chat_sessions
  FOR ALL USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());
CREATE POLICY owner_rw_messages ON augment_agent.chat_messages
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM augment_agent.chat_sessions s WHERE s.id = session_id AND s.user_id = auth.uid()
    )
  ) WITH CHECK (
    EXISTS (
      SELECT 1 FROM augment_agent.chat_sessions s WHERE s.id = session_id AND s.user_id = auth.uid()
    )
  );
CREATE POLICY owner_rw_profiles ON augment_agent.user_profiles
  FOR ALL USING (user_id = auth.uid()) WITH CHECK (user_id = auth.uid());

-- Shared household (optional) via metadata flag on documents/vectors
CREATE POLICY household_read_docvecs ON augment_agent.document_vectors
  FOR SELECT USING (
    metadata ? 'household_shared' AND (metadata->>'household_shared')::boolean = true
  );
```

## Memory Patterns
- Upsert pipeline: Docling → chunk → embed → insert chunk + vector
- Conversation vectorization: optional for semantic search over chats
- Personalization vectors: store per-user preferences in a profile table or metadata

## Example Query
```sql
-- k nearest neighbors by cosine distance (per-user)
SELECT dv.id, dv.doc_id, 1 - (dv.embedding <=> $1::vector) AS score
FROM augment_agent.document_vectors dv
JOIN augment_agent.documents d ON d.id = dv.doc_id
WHERE d.user_id = $2
ORDER BY dv.embedding <-> $1::vector
LIMIT 10;
```

## Supabase Setup
- Configure `SUPABASE_URL` and `SUPABASE_ANON_KEY` in orchestrator/brain services
- Use PostgREST endpoints for CRUD, or direct Postgres connection from backend services

## Backups & Retention
- Nightly pg_dump backups
- PII minimization; encrypt sensitive metadata at rest

