# Document Ingestion with Docling

- Use Docling to extract structure, tables, and layout from PDFs/Docs.
- Recommended: semantic chunking with metadata (source, page, section, table ids).

Python (example skeleton)

```python
from langchain_docling import DoclingLoader

loader = DoclingLoader(
    file_path="/path/to/document.pdf",
    export_type=ExportType.DOC_CHUNKS,
    chunker=SemanticChunker(),
)
chunks = loader.load()
processed = [{
    "content": c.page_content,
    "metadata": c.metadata,
} for c in chunks]
```

Status
- Current code uses a naive splitter; Docling integration is planned. Use this as the target API and keep metadata parity.

Notes
- Keep chunk sizes ~500-1000 tokens for robust retrieval and reranking.
- Capture table content separately if needed for precise QA.

