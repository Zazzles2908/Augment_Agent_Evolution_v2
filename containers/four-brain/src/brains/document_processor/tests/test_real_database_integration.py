"""
Real Database Integration Tests - NO MOCKING
Tests actual PostgreSQL operations against augment_agent schema
AUTHENTIC IMPLEMENTATION - Zero fabrication policy
"""

import pytest
import asyncio
import asyncpg
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from brain4_docling.config.settings import Brain4Settings
from brain4_docling.integration.document_store import DocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealDatabaseIntegration:
    """Integration tests using real PostgreSQL database - NO MOCKING"""
    
    @pytest.fixture
    def settings(self):
        """Real database settings"""
        return Brain4Settings()
    
    @pytest.fixture
    async def real_db_connection(self, settings):
        """Real PostgreSQL connection to augment_agent schema"""
        conn = await asyncpg.connect(settings.database_url, timeout=10.0)
        
        # Verify connection to real database
        result = await conn.fetchval("SELECT current_database()")
        logger.info(f"Connected to real database: {result}")
        
        yield conn
        
        # Cleanup
        await conn.close()
    
    @pytest.fixture
    async def real_document_store(self, settings):
        """Real DocumentStore instance"""
        store = DocumentStore(settings)
        await store.initialize()
        
        yield store
        
        await store.close()
    
    @pytest.mark.asyncio
    async def test_real_database_schema_validation(self, real_db_connection):
        """Validate real augment_agent schema exists and has correct structure"""
        conn = real_db_connection
        
        # Verify augment_agent schema exists
        schema_exists = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'augment_agent')
        """)
        assert schema_exists is True, "augment_agent schema does not exist"
        
        # Verify documents table structure
        documents_columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'augment_agent' AND table_name = 'documents'
            ORDER BY ordinal_position
        """)
        
        expected_columns = {
            'id': 'uuid',
            'filename': 'character varying',
            'file_path': 'text',
            'file_size': 'bigint',
            'mime_type': 'character varying',
            'upload_timestamp': 'timestamp with time zone',
            'processing_status': 'character varying',
            'processing_timestamp': 'timestamp with time zone',
            'metadata': 'jsonb',
            'created_at': 'timestamp with time zone',
            'updated_at': 'timestamp with time zone'
        }
        
        actual_columns = {row['column_name']: row['data_type'] for row in documents_columns}
        
        for col_name, col_type in expected_columns.items():
            assert col_name in actual_columns, f"Missing column: {col_name}"
            assert actual_columns[col_name] == col_type, f"Wrong type for {col_name}: {actual_columns[col_name]} != {col_type}"
        
        logger.info("✅ Real database schema validation passed")
    
    @pytest.mark.asyncio
    async def test_real_document_insertion(self, real_db_connection):
        """Test real document insertion into augment_agent.documents"""
        conn = real_db_connection
        
        # Generate real test data
        document_id = str(uuid.uuid4())
        test_metadata = {
            "test_type": "real_database_integration",
            "timestamp": datetime.now().isoformat(),
            "authentic": True
        }
        
        # Perform real INSERT operation
        await conn.execute("""
            INSERT INTO augment_agent.documents 
            (id, filename, file_size, mime_type, processing_status, metadata, upload_timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 
        document_id,
        "test_document.pdf",
        1024,
        "application/pdf",
        "pending",
        json.dumps(test_metadata),
        datetime.now()
        )
        
        # Verify real data was inserted
        inserted_doc = await conn.fetchrow("""
            SELECT id, filename, file_size, mime_type, processing_status, metadata
            FROM augment_agent.documents 
            WHERE id = $1
        """, document_id)
        
        assert inserted_doc is not None, "Document not found after insertion"
        assert str(inserted_doc['id']) == document_id
        assert inserted_doc['filename'] == "test_document.pdf"
        assert inserted_doc['file_size'] == 1024
        assert inserted_doc['mime_type'] == "application/pdf"
        assert inserted_doc['processing_status'] == "pending"
        
        # Verify metadata JSON
        stored_metadata = json.loads(inserted_doc['metadata'])
        assert stored_metadata['test_type'] == "real_database_integration"
        assert stored_metadata['authentic'] is True
        
        # Cleanup test data
        await conn.execute("DELETE FROM augment_agent.documents WHERE id = $1", document_id)
        
        logger.info("✅ Real document insertion test passed")
    
    @pytest.mark.asyncio
    async def test_real_document_query_operations(self, real_db_connection):
        """Test real document query operations with actual data"""
        conn = real_db_connection
        
        # Insert multiple test documents
        test_docs = []
        for i in range(3):
            doc_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO augment_agent.documents 
                (id, filename, file_size, processing_status, upload_timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, doc_id, f"test_doc_{i}.pdf", 1000 + i, "pending", datetime.now())
            test_docs.append(doc_id)
        
        # Test real SELECT with pagination
        documents = await conn.fetch("""
            SELECT id, filename, file_size, processing_status
            FROM augment_agent.documents 
            WHERE id = ANY($1)
            ORDER BY filename
            LIMIT 2 OFFSET 0
        """, test_docs)
        
        assert len(documents) == 2
        assert documents[0]['filename'] == "test_doc_0.pdf"
        assert documents[1]['filename'] == "test_doc_1.pdf"
        
        # Test real COUNT operation
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM augment_agent.documents WHERE id = ANY($1)
        """, test_docs)
        assert count == 3
        
        # Test real UPDATE operation
        updated_doc_id = test_docs[0]
        await conn.execute("""
            UPDATE augment_agent.documents 
            SET processing_status = 'completed', processing_timestamp = $1
            WHERE id = $2
        """, datetime.now(), updated_doc_id)
        
        # Verify update
        updated_doc = await conn.fetchrow("""
            SELECT processing_status, processing_timestamp
            FROM augment_agent.documents WHERE id = $1
        """, updated_doc_id)
        
        assert updated_doc['processing_status'] == "completed"
        assert updated_doc['processing_timestamp'] is not None
        
        # Cleanup test data
        await conn.execute("DELETE FROM augment_agent.documents WHERE id = ANY($1)", test_docs)
        
        logger.info("✅ Real document query operations test passed")
    
    @pytest.mark.asyncio
    async def test_real_document_store_integration(self, real_document_store):
        """Test real DocumentStore operations without mocking"""
        store = real_document_store
        
        # Test real document storage
        test_document_data = {
            "source_path": "/test/real_document.pdf",
            "filename": "real_document.pdf",
            "document_type": "pdf",
            "file_size": 2048,
            "content": {"text": "Real document content for testing"},
            "chunks": [{"chunk_id": "chunk_0", "text": "Real content chunk"}],
            "processing_time": 1.5,
            "embeddings": [[0.1, 0.2, 0.3] * 667]  # 2000 dimensions
        }
        
        # Store document with real DocumentStore
        result = await store.store_document(test_document_data)
        assert result is True, "Document storage failed"
        
        logger.info("✅ Real DocumentStore integration test passed")
    
    @pytest.mark.asyncio
    async def test_real_database_error_handling(self, settings):
        """Test real database error handling without mocking"""
        # Test connection failure with invalid URL
        invalid_url = "postgresql://invalid:invalid@invalid:5432/invalid"
        
        with pytest.raises(asyncpg.exceptions.PostgresConnectionError):
            await asyncpg.connect(invalid_url, timeout=5.0)
        
        # Test real connection with valid URL
        conn = await asyncpg.connect(settings.database_url, timeout=10.0)
        
        # Test constraint violation (duplicate primary key)
        test_id = str(uuid.uuid4())
        
        # First insertion should succeed
        await conn.execute("""
            INSERT INTO augment_agent.documents (id, filename, file_size)
            VALUES ($1, $2, $3)
        """, test_id, "test.pdf", 1024)
        
        # Second insertion with same ID should fail
        with pytest.raises(asyncpg.exceptions.UniqueViolationError):
            await conn.execute("""
                INSERT INTO augment_agent.documents (id, filename, file_size)
                VALUES ($1, $2, $3)
            """, test_id, "test2.pdf", 2048)
        
        # Cleanup
        await conn.execute("DELETE FROM augment_agent.documents WHERE id = $1", test_id)
        await conn.close()
        
        logger.info("✅ Real database error handling test passed")
    
    @pytest.mark.asyncio
    async def test_real_transaction_handling(self, real_db_connection):
        """Test real database transaction handling"""
        conn = real_db_connection
        
        # Test successful transaction
        async with conn.transaction():
            doc_id = str(uuid.uuid4())
            await conn.execute("""
                INSERT INTO augment_agent.documents (id, filename, file_size)
                VALUES ($1, $2, $3)
            """, doc_id, "transaction_test.pdf", 1024)
            
            # Verify document exists within transaction
            exists = await conn.fetchval("""
                SELECT EXISTS(SELECT 1 FROM augment_agent.documents WHERE id = $1)
            """, doc_id)
            assert exists is True
        
        # Verify document still exists after transaction commit
        exists_after = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM augment_agent.documents WHERE id = $1)
        """, doc_id)
        assert exists_after is True
        
        # Test transaction rollback
        try:
            async with conn.transaction():
                doc_id2 = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO augment_agent.documents (id, filename, file_size)
                    VALUES ($1, $2, $3)
                """, doc_id2, "rollback_test.pdf", 2048)
                
                # Force rollback by raising exception
                raise Exception("Intentional rollback")
        except Exception:
            pass
        
        # Verify document was rolled back
        exists_rollback = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM augment_agent.documents WHERE id = $1)
        """, doc_id2)
        assert exists_rollback is False
        
        # Cleanup
        await conn.execute("DELETE FROM augment_agent.documents WHERE id = $1", doc_id)
        
        logger.info("✅ Real transaction handling test passed")

if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v", "-s"])
