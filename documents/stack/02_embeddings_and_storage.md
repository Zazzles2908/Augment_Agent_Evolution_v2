# Embeddings and Storage (Qwen3-4B + Supabase + Redis)

- Embedding model: Qwen3-4B served via Triton (FP8 plan), reranker/generator (NVFP4)
- Output dimension: 2000 (aligns with Supabase pgvector HNSW limit)
- Cache: Redis for deduping repeated texts
- Store: Supabase table with text, embedding (vector(2000)), and metadata

Supabase schema (example)

```sql
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(2000) not null,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);
create index if not exists documents_embedding_idx
  on documents using hnsw (embedding vector_cosine_ops);
```

Embedding flow
1) Check Redis cache by hash(content)
2) If miss, call Triton model qwen3_4b_embedding
3) Store embedding in Redis and upsert into Supabase



Supabase RPC (match_documents)

SQL (create once):
```
create or replace function match_documents(
  query_embedding vector(2000),
  match_count int default 10,
  similarity_threshold float default 0.3
)
returns table (id uuid, content text, metadata jsonb, similarity float)
language sql stable as $$
  select d.id, d.content, d.metadata,
         1 - (d.embedding <=> query_embedding) as similarity
  from documents d
  where 1 - (d.embedding <=> query_embedding) >= similarity_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;
```

Python example:
```
from supabase import create_client
from tritonclient.http import InferenceServerClient, InferInput

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
triton = InferenceServerClient(url="http://localhost:8000")

# Embed query with Triton
# Adjust inputs to your embedding model signature
inputs = [
  InferInput("input_ids", [1, 128], "INT64"),
  InferInput("attention_mask", [1, 128], "INT64"),
]
# TODO: Fill with tokenized data
result = triton.infer("qwen3_4b_embedding", inputs)
query_emb = result.as_numpy("embedding")[0]

rows = supabase.rpc("match_documents", {
  "query_embedding": query_emb.tolist(),
  "match_count": 10,
  "similarity_threshold": 0.3
}).execute()
print(rows)
```
