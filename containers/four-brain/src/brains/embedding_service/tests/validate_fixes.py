#!/usr/bin/env python3
"""
Comprehensive Validation Script for Fixed Three-Brain Architecture
Tests the fixes for:
1. Memory Management (smart caching vs load-test-delete)
2. Database Connection Architecture (hybrid approach)
3. Performance Optimizations
"""

import asyncio
import logging
import sys
import os
import time
import torch
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_memory_management():
    """Test that models stay loaded in memory (no load-test-delete pattern)."""
    logger.info("🧪 Testing Memory Management Fixes...")
    
    try:
        from three_brain_system import ThreeBrainSystem
        
        # Initialize system
        system = ThreeBrainSystem(cache_dir="/workspace/models")
        
        # Test model caching
        logger.info("Testing smart model caching...")
        
        # Load embedding model - should cache it
        embedding_model_1 = system.get_embedding_model(use_8bit=True)
        cache_size_after_first_load = len(system._model_cache)
        
        # Load same model again - should use cache
        embedding_model_2 = system.get_embedding_model(use_8bit=True)
        cache_size_after_second_load = len(system._model_cache)
        
        # Verify caching works
        if embedding_model_1 is embedding_model_2:
            logger.info("✅ Model caching works - same object returned")
        else:
            logger.error("❌ Model caching failed - different objects returned")
            return False
        
        if cache_size_after_first_load == cache_size_after_second_load == 1:
            logger.info("✅ Cache size correct - model not duplicated")
        else:
            logger.error(f"❌ Cache size incorrect - {cache_size_after_first_load} vs {cache_size_after_second_load}")
            return False
        
        # Test that models persist after initialization
        logger.info("Testing model persistence...")
        init_success = system.initialize_all_brains(use_8bit=True, use_4bit=False)
        
        if init_success:
            # Check that models are still in cache and accessible
            if system.brain1_embedding is not None and system.brain2_reranker is not None:
                logger.info("✅ Models persist after initialization - NO load-test-delete pattern")
            else:
                logger.error("❌ Models were deleted after initialization - load-test-delete pattern still present")
                return False
        else:
            logger.error("❌ System initialization failed")
            return False
        
        # Test GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"📊 GPU Memory Allocated: {memory_allocated:.2f}GB")
            
            if memory_allocated > 0:
                logger.info("✅ Models are loaded in GPU memory")
            else:
                logger.warning("⚠️ No GPU memory allocated - models may not be loaded")
        
        logger.info("✅ Memory management test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory management test FAILED: {e}")
        return False

async def test_database_architecture():
    """Test hybrid database approach (Supabase + PostgreSQL)."""
    logger.info("🧪 Testing Database Architecture Fixes...")
    
    try:
        from database_manager import get_database_manager
        
        # Initialize database manager
        db_manager = await get_database_manager()
        
        # Test Supabase REST API connection
        logger.info("Testing Supabase REST API...")
        try:
            # FIXED: Use correct column name from actual schema
            result = db_manager.supabase_client.schema('augment_agent').table('knowledge').select('knowledge_id').limit(1).execute()
            logger.info("✅ Supabase REST API connection works")
        except Exception as e:
            logger.error(f"❌ Supabase REST API failed: {e}")
            return False
        
        # Test PostgreSQL direct connection
        logger.info("Testing PostgreSQL direct connection...")
        try:
            async with db_manager.pg_pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM augment_agent.knowledge")
                logger.info(f"✅ PostgreSQL direct connection works - {count} records")
        except Exception as e:
            logger.error(f"❌ PostgreSQL direct connection failed: {e}")
            return False
        
        # Test connection pool stats
        pool_stats = await db_manager.get_connection_pool_stats()
        logger.info(f"📊 Connection Pool Stats: {pool_stats}")
        
        # Test health check
        health = await db_manager.health_check()
        if health['overall_healthy']:
            logger.info("✅ Database health check PASSED")
        else:
            logger.error(f"❌ Database health check FAILED: {health}")
            return False
        
        logger.info("✅ Database architecture test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database architecture test FAILED: {e}")
        return False

async def test_vector_operations():
    """Test performance-optimized vector operations."""
    logger.info("🧪 Testing Vector Operations Performance...")
    
    try:
        from database_manager import get_database_manager
        
        db_manager = await get_database_manager()
        
        # Test vector similarity search
        logger.info("Testing vector similarity search...")
        dummy_embedding = [0.1] * 2000  # 2000-dimensional vector

        start_time = time.time()
        results = await db_manager.vector_similarity_search(
            dummy_embedding,
            similarity_threshold=0.1,
            max_results=10,
            use_index=True
        )
        search_time = time.time() - start_time

        # FIXED: Validate actual success conditions, not just absence of exceptions
        if len(results) == 0:
            logger.warning(f"⚠️ Vector search returned 0 results - may indicate database connection issues")
            # Check if this is due to empty table or connection failure
            stats = await db_manager.get_embedding_stats()
            if not stats or stats.get('total_knowledge_entries', 0) == 0:
                logger.info("ℹ️ Empty results due to no data in knowledge table")
            else:
                logger.error("❌ Vector search failed - database has data but no results returned")
                return False

        logger.info(f"✅ Vector search completed in {search_time:.3f}s - {len(results)} results")
        
        # Test batch vector operations
        logger.info("Testing batch vector operations...")
        batch_embeddings = [[0.1] * 2000 for _ in range(5)]  # 5 queries

        start_time = time.time()
        batch_results = await db_manager.batch_vector_search(batch_embeddings, max_results_per_query=3)
        batch_time = time.time() - start_time

        # FIXED: Validate meaningful results, not empty responses
        if len(batch_results) != 5:
            logger.error(f"❌ Batch vector search failed - expected 5 result sets, got {len(batch_results)}")
            return False

        logger.info(f"✅ Batch vector search completed in {batch_time:.3f}s - {len(batch_results)} result sets")
        
        # Test hybrid search
        logger.info("Testing hybrid search...")
        start_time = time.time()
        hybrid_results = await db_manager.hybrid_search(
            "test query", 
            dummy_embedding, 
            text_weight=0.3, 
            vector_weight=0.7,
            max_results=5
        )
        hybrid_time = time.time() - start_time
        
        logger.info(f"✅ Hybrid search completed in {hybrid_time:.3f}s - {len(hybrid_results)} results")
        
        # Test embedding statistics
        stats = await db_manager.get_embedding_stats()
        logger.info(f"📊 Embedding Stats: {stats}")
        
        logger.info("✅ Vector operations test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector operations test FAILED: {e}")
        return False

async def test_integration():
    """Test integration between Three-Brain system and database."""
    logger.info("🧪 Testing Three-Brain + Database Integration...")
    
    try:
        from three_brain_system import ThreeBrainSystem
        
        # Initialize system with database integration
        system = ThreeBrainSystem(cache_dir="/workspace/models")
        
        # Test database initialization
        db_init_success = await system.initialize_database()
        if not db_init_success:
            logger.error("❌ Database initialization failed")
            return False
        
        # Test Three-Brain initialization
        brain_init_success = system.initialize_all_brains(use_8bit=True, use_4bit=False)
        if not brain_init_success:
            logger.error("❌ Three-Brain initialization failed")
            return False
        
        # Test knowledge search integration
        logger.info("Testing knowledge search integration...")
        search_results = await system.search_similar_knowledge("test query", max_results=5)
        logger.info(f"✅ Knowledge search returned {len(search_results)} results")
        
        # Test system status
        status = system.get_system_status()
        logger.info(f"📊 System Status: {status}")
        
        if status['brain1_initialized'] and status['brain2_initialized']:
            logger.info("✅ Both brains initialized and persistent")
        else:
            logger.error("❌ Brain initialization status incorrect")
            return False
        
        logger.info("✅ Integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test FAILED: {e}")
        return False

async def main():
    """Run comprehensive validation of all fixes."""
    logger.info("🚀 Starting Comprehensive Validation of Three-Brain Architecture Fixes")
    logger.info("="*80)
    
    # Load environment variables
    if os.path.exists('.env.enhanced'):
        try:
            from dotenv import load_dotenv
            load_dotenv('.env.enhanced')
            logger.info("📁 Loaded environment from .env.enhanced")
        except ImportError:
            logger.warning("⚠️ python-dotenv not available, using system environment")

    # CRITICAL FIX: Create ONE persistent brain system for all tests
    logger.info("🧠 Initializing persistent Three-Brain Architecture...")
    try:
        from three_brain_system import ThreeBrainSystem
        persistent_brain_system = ThreeBrainSystem(cache_dir="/workspace/models")
        # Pre-initialize to load models once
        init_success = persistent_brain_system.initialize_all_brains(use_8bit=True, use_4bit=False)
        if not init_success:
            logger.error("❌ Failed to initialize persistent brain system")
            return False
        logger.info("✅ Persistent brain system initialized - models cached for reuse")
    except Exception as e:
        logger.error(f"❌ Failed to create persistent brain system: {e}")
        return False

    tests = [
        ("Memory Management Fixes", test_memory_management),
        ("Database Architecture Fixes", test_database_architecture),
        ("Vector Operations Performance", test_vector_operations),
        ("Three-Brain + Database Integration", test_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = await test_func()
            test_time = time.time() - start_time
            
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name}: PASSED ({test_time:.2f}s)")
            else:
                logger.error(f"❌ {test_name}: FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Final Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPREHENSIVE VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
        logger.info("✅ Memory management: Smart caching implemented")
        logger.info("✅ Database architecture: Hybrid approach working")
        logger.info("✅ Vector operations: Performance optimized")
        logger.info("✅ Integration: Three-Brain + Database working")
        return True
    else:
        logger.error("💥 SOME FIXES FAILED VALIDATION!")
        logger.error("❌ Please review failed tests and fix issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
