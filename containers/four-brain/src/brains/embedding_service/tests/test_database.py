#!/usr/bin/env python3
"""
Test Database Connections for Three-Brain Architecture
Tests both Supabase REST API and direct PostgreSQL connections
"""

import asyncio
import logging
import sys
import os
from database_manager import get_database_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_supabase_connection():
    """Test Supabase REST API connection."""
    logger.info("🧪 Testing Supabase REST API connection...")
    
    try:
        db_manager = await get_database_manager()
        
        # FIXED: Test basic connection by querying knowledge table with correct column name
        result = db_manager.supabase_client.schema('augment_agent').table('knowledge').select('knowledge_id').limit(1).execute()
        
        if result.data is not None:
            logger.info(f"✅ Supabase connection successful - {len(result.data)} records found")
            return True
        else:
            logger.error("❌ Supabase connection failed - no data returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {e}")
        return False

async def test_postgresql_connection():
    """Test direct PostgreSQL connection."""
    logger.info("🧪 Testing direct PostgreSQL connection...")
    
    try:
        db_manager = await get_database_manager()
        
        # Test direct PostgreSQL connection
        async with db_manager.pg_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM augment_agent.knowledge")
            logger.info(f"✅ PostgreSQL connection successful - {count} records in knowledge table")
            return True
            
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection test failed: {e}")
        return False

async def test_vector_operations():
    """Test vector operations."""
    logger.info("🧪 Testing vector operations...")
    
    try:
        db_manager = await get_database_manager()
        
        # Get embedding statistics
        stats = await db_manager.get_embedding_stats()
        logger.info(f"📊 Embedding stats: {stats}")
        
        # Test vector similarity search with dummy embedding
        dummy_embedding = [0.1] * 2000  # 2000-dimensional dummy vector
        results = await db_manager.vector_similarity_search(
            dummy_embedding, 
            similarity_threshold=0.1, 
            max_results=5
        )
        
        logger.info(f"✅ Vector search test completed - {len(results)} results")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector operations test failed: {e}")
        return False

async def test_crud_operations():
    """Test basic CRUD operations."""
    logger.info("🧪 Testing CRUD operations...")
    
    try:
        db_manager = await get_database_manager()
        
        # Test creating a knowledge entry with correct data format
        test_data = {
            'title': 'Test Knowledge Entry',
            'content': {'text': 'This is a test entry for database validation', 'type': 'validation'},  # jsonb format
            'domain': 'testing',
            'knowledge_type': 'test',
            'confidence_level': 0.95,
            'tags': ['test', 'validation']
        }
        
        # Create entry
        created_entry = db_manager.create_knowledge_entry(test_data)
        if created_entry and 'id' in created_entry:
            entry_id = created_entry['id']
            logger.info(f"✅ Created test entry with ID: {entry_id}")
            
            # Read entry
            retrieved_entry = db_manager.get_knowledge_by_id(entry_id)
            if retrieved_entry:
                logger.info("✅ Successfully retrieved test entry")
                
                # Update entry
                update_data = {'confidence_level': 0.99}
                updated_entry = db_manager.update_knowledge_entry(entry_id, update_data)
                if updated_entry:
                    logger.info("✅ Successfully updated test entry")
                    return True
        
        logger.error("❌ CRUD operations test failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ CRUD operations test failed: {e}")
        return False

async def test_schema_validation():
    """Test schema and table existence."""
    logger.info("🧪 Testing schema validation...")
    
    try:
        db_manager = await get_database_manager()
        
        async with db_manager.pg_pool.acquire() as conn:
            # Check if augment_agent schema exists
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'augment_agent')"
            )
            
            if schema_exists:
                logger.info("✅ augment_agent schema exists")
                
                # Check tables in schema
                tables = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'augment_agent'"
                )
                
                table_names = [row['table_name'] for row in tables]
                logger.info(f"📋 Tables in augment_agent schema: {table_names}")
                
                # Check if knowledge table has embedding column
                embedding_column = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_schema = 'augment_agent' 
                        AND table_name = 'knowledge' 
                        AND column_name = 'embedding'
                    )
                    """
                )
                
                if embedding_column:
                    logger.info("✅ knowledge table has embedding column")
                else:
                    logger.warning("⚠️ knowledge table missing embedding column")
                
                return True
            else:
                logger.error("❌ augment_agent schema does not exist")
                return False
                
    except Exception as e:
        logger.error(f"❌ Schema validation failed: {e}")
        return False

async def main():
    """Run all database tests."""
    logger.info("🚀 Starting database connection tests...")
    
    # Load environment variables
    if os.path.exists('.env.enhanced'):
        from dotenv import load_dotenv
        load_dotenv('.env.enhanced')
        logger.info("📁 Loaded environment from .env.enhanced")
    
    tests = [
        ("Schema Validation", test_schema_validation),
        ("Supabase Connection", test_supabase_connection),
        ("PostgreSQL Connection", test_postgresql_connection),
        ("Vector Operations", test_vector_operations),
        ("CRUD Operations", test_crud_operations),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All database tests PASSED!")
        return True
    else:
        logger.error("💥 Some database tests FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
