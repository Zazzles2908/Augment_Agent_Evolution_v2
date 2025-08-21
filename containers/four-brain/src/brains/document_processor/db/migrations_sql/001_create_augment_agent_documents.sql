-- Create schema and table for Brain-4 document storage
CREATE SCHEMA IF NOT EXISTS augment_agent;

CREATE TABLE IF NOT EXISTS augment_agent.documents (
  id UUID PRIMARY KEY,
  filename TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  mime_type TEXT NOT NULL,
  processing_status TEXT NOT NULL DEFAULT 'pending',
  metadata JSONB NOT NULL DEFAULT '{}',
  upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  processing_timestamp TIMESTAMPTZ NULL
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_documents_status ON augment_agent.documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON augment_agent.documents(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_documents_processed ON augment_agent.documents(processing_timestamp);

