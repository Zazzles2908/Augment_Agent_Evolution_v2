# GLM-4.5 Implementation Recommendations – 01_document_ingestion_docling.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Ingestion Architecture (Flow)
```
Document Input → DoclingLoader → SemanticChunker → Metadata Extraction → Processed Output
```
- Accept PDF/Doc via file watcher or API
- Use DoclingLoader with DOC_CHUNKS export for structured extraction
- Apply SemanticChunker (500–1000 tokens, ~10% overlap)
- Enforce metadata schema: source, page, section, table_id, chunk_id
- Output structured JSON with content + metadata pairs

## 2. Chunking & Metadata Design
- Token size: 500–1000; adjust by doc type; overlap ≈10%
- Metadata example:
```json
{
  "source": "document.pdf",
  "page": 5,
  "section": "3.2 Financial Results",
  "table_id": "tbl_financial_2023",
  "chunk_id": "doc1_page5_sec3_2_chunk3",
  "token_count": 742
}
```

## 3. Table Handling Strategy
- Extract tables prior to semantic chunking
- Store table content separately with dedicated metadata
- Maintain table structure and positional context
- Create dedicated table chunks for QA precision
- Link table chunks to surrounding content via section metadata

## 4. Performance Engineering
- Pre-screen documents; flag >100MB for special handling
- Use streaming processing for large files
- Set memory limits and batch sizes
- Track progress and timings per stage
- Cache frequently accessed document structures

## 5. Validation & QA Checks
- Validate chunk size distribution (≥90% within target)
- Ensure required metadata fields present
- Spot-check 10% of extracted tables for accuracy
- Test retrieval with representative queries
- Monitor processing time per document type

## 6. Operational Runbook (Errors, Retries, Idempotency)
- Retries: up to 3 attempts with backoff
- Idempotency: unique doc ID + checksum; skip if unchanged
- Structured error logs with metadata context
- Circuit breaker for repeated failures
- Fallback strategies for problematic formats

