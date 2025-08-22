# Client Implementation — For AI Specialist Developers

This guide explains how to integrate clients (desktop/web/mobile) with the Orchestrator API and memory services.

## Personas
- User A (You): primary operator; full access to personal + shared collections
- User B (Wife): full access to personal; opt-in write to shared household collections

## Authentication
- Use Supabase Auth or your own JWT provider; include user_id in JWT claims for RLS

## Core Flows
- Chat: send user messages; receive responses and citations
- Tasks: create, monitor, and complete action items
- Memory: ingest documents, search memory, and fetch relevant chunks

## REST Endpoints (Proposed)
- POST /chat { user_id, message, context? }
- POST /tasks { user_id, description, due?, approvals_required? }
- GET /tasks?user_id=...
- POST /memory/upsert { user_id, chunks[] }
- POST /memory/search { user_id, query, top_k }
- POST /documents/ingest { user_id, file }

## Example: TypeScript Client
```ts
async function chat(message: string, userId: string) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, message }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}
```

## Document Ingestion
- Upload file → Docling parses → chunks → embeddings → store in pgvector
- Return ingestion job id; poll until complete

## Safety & Approvals
- Annotate requests with `requires_approval: true` where irreversible actions occur
- HRM surfaces a structured approval card for the user to accept/deny

## UI Guidance
- Show reasoning summaries with toggles (transparency)
- Provide sources/citations for factual answers
- Expose retrievable memories and allow pinning to shared household

## Telemetry
- Capture latency per endpoint, token counts, and approval rates
- Respect privacy; do not send content to external clouds unless allowed

## Example Error Handling
```ts
try {
  const result = await chat("Plan weekly meals", userId)
} catch (e) {
  // Suggest fallback: reduce model size or switch to CPU
}
```

