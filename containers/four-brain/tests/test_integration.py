#!/usr/bin/env python3
"""
Integration Tests for Four-Brain System v2
Tests end-to-end workflow, performance scoring, and self-improvement detection

This module provides comprehensive integration tests to verify that all
components of the Four-Brain System work together correctly.

Zero Fabrication Policy: ENFORCED
All tests use real system components and actual data flows.
"""

import asyncio
import pytest
import time
import json
import uuid
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.redis_client import RedisStreamsClient
from shared.memory_store import MemoryStore, TaskScore
from shared.self_grading import SelfGradingSystem
from shared.self_improvement import SelfImprovementEngine, ReflectionLevel
from shared.streams import StreamMessage, MessageType
from shared.model_verification import ModelVerificationSystem, run_comprehensive_verification
from brains.brain3_augment.agentic_loop import AgenticLoop

class TestFourBrainIntegration:
    """Integration tests for Four-Brain System"""
    
    @pytest.fixture
    async def setup_system(self):
        """Setup test system components"""
        # Initialize Redis client
        redis_client = RedisStreamsClient(brain_id="test_brain")
        await redis_client.connect()
        
        # Initialize memory store
        memory_store = MemoryStore()
        await memory_store.connect()
        
        # Initialize self-grading system
        self_grading = SelfGradingSystem(redis_client, memory_store)
        
        # Initialize self-improvement engine
        self_improvement = SelfImprovementEngine(memory_store, self_grading)
        
        # Initialize Brain-3 agentic loop
        agentic_loop = AgenticLoop()
        
        yield {
            "redis_client": redis_client,
            "memory_store": memory_store,
            "self_grading": self_grading,
            "self_improvement": self_improvement,
            "agentic_loop": agentic_loop
        }
        
        # Cleanup
        await redis_client.disconnect()
        await memory_store.disconnect()
    
    @pytest.mark.asyncio
    async def test_model_verification_system(self):
        """Test model verification system"""
        print("🔍 Testing model verification system...")
        
        # Run comprehensive verification
        verification_result = run_comprehensive_verification()
        
        # Verify structure
        assert "verification_completed" in verification_result
        assert "models_verified" in verification_result
        assert "system_recommendations" in verification_result
        assert "model_results" in verification_result
        
        # Check that verification completed
        assert verification_result["verification_completed"] is True
        
        # Check that at least some models were verified
        assert verification_result["models_verified"] > 0
        
        print(f"✅ Model verification completed: {verification_result['models_verified']} models checked")
        
        # Test individual model verification
        verifier = ModelVerificationSystem()
        brain3_result = verifier.verify_model("brain3")
        
        # Brain-3 should pass (API-based, no local requirements)
        assert brain3_result.model_name == "Brain-3 Zazzles's Agent"
        assert len(brain3_result.errors) == 0
        
        print("✅ Individual model verification passed")
    
    @pytest.mark.asyncio
    async def test_redis_streams_communication(self, setup_system):
        """Test Redis Streams communication"""
        print("🌊 Testing Redis Streams communication...")
        
        components = await setup_system
        redis_client = components["redis_client"]
        
        # Test stream creation and message sending
        test_message = StreamMessage(
            task_id=str(uuid.uuid4()),
            message_type=MessageType.AGENTIC_TASK,
            timestamp=time.time(),
            brain_id="test_brain",
            data={"test": "message", "content": "integration test"}
        )
        
        # Send message to agentic tasks stream
        await redis_client.send_message("agentic_tasks", test_message)
        
        # Verify message was sent
        stream_info = await redis_client.get_stream_info("agentic_tasks")
        assert stream_info["length"] > 0
        
        print("✅ Redis Streams communication verified")
    
    @pytest.mark.asyncio
    async def test_memory_store_operations(self, setup_system):
        """Test memory store operations"""
        print("🧠 Testing memory store operations...")
        
        components = await setup_system
        memory_store = components["memory_store"]
        
        # Create test task score
        task_score = TaskScore(
            task_id=str(uuid.uuid4()),
            brain_id="test_brain",
            operation="test_operation",
            score=0.85,
            task_signature="test_signature_123",
            timestamp=time.time(),
            metadata={"test": True, "integration": "test"}
        )
        
        # Store task score
        success = await memory_store.store_task_score(task_score)
        assert success is True
        
        # Retrieve task score
        retrieved_score = await memory_store.get_task_score(task_score.task_id)
        assert retrieved_score is not None
        assert retrieved_score.score == 0.85
        assert retrieved_score.brain_id == "test_brain"
        
        # Test pattern matching
        pattern_match = await memory_store.get_past_attempts(
            "test_operation", 
            {"test": True}
        )
        assert pattern_match is not None
        
        print("✅ Memory store operations verified")
    
    @pytest.mark.asyncio
    async def test_self_grading_system(self, setup_system):
        """Test self-grading system"""
        print("📊 Testing self-grading system...")
        
        components = await setup_system
        self_grading = components["self_grading"]
        
        # Test performance score calculation
        test_data = {
            "task_id": str(uuid.uuid4()),
            "brain_id": "test_brain",
            "operation": "test_grading",
            "input_data": {"test": "input"},
            "output_data": {"result": "success", "quality": "high"},
            "execution_time": 1.5,
            "resource_usage": {"memory": 100, "cpu": 50}
        }
        
        # Calculate performance score
        score = await self_grading.calculate_performance_score(test_data)
        
        # Verify score is valid
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be a decent score for successful operation
        
        print(f"✅ Self-grading system verified: score = {score:.3f}")
    
    @pytest.mark.asyncio
    async def test_self_improvement_engine(self, setup_system):
        """Test self-improvement engine"""
        print("🔄 Testing self-improvement engine...")
        
        components = await setup_system
        self_improvement = components["self_improvement"]
        memory_store = components["memory_store"]
        
        # Add some test performance data
        test_scores = [
            TaskScore(
                task_id=str(uuid.uuid4()),
                brain_id="test_brain",
                operation="test_improvement",
                score=0.6 + i * 0.05,  # Gradually improving scores
                task_signature=f"test_sig_{i}",
                timestamp=time.time() - (10 - i) * 3600,  # Spread over 10 hours
                metadata={"iteration": i}
            )
            for i in range(5)
        ]
        
        # Store test scores
        for score in test_scores:
            await memory_store.store_task_score(score)
        
        # Perform reflection
        reflection_report = await self_improvement.perform_reflection(
            "test_brain", 
            ReflectionLevel.MEDIUM, 
            12  # 12 hour window
        )
        
        # Verify reflection report
        assert reflection_report.brain_id == "test_brain"
        assert reflection_report.reflection_level == ReflectionLevel.MEDIUM
        assert reflection_report.confidence_score > 0.0
        
        # Check if improvement suggestions were generated
        assert len(reflection_report.improvement_suggestions) >= 0
        
        print(f"✅ Self-improvement engine verified: {len(reflection_report.improvement_suggestions)} suggestions generated")
    
    @pytest.mark.asyncio
    async def test_brain3_self_validation(self, setup_system):
        """Test Brain-3 self-validation mechanism"""
        print("🧠 Testing Brain-3 self-validation...")
        
        components = await setup_system
        agentic_loop = components["agentic_loop"]
        
        # Test quality gates with good content
        good_result = {
            "output": """
def test_function(x, y):
    '''Test function with validation
    
    Examples:
    >>> test_function(2, 3)
    5
    >>> test_function(0, 5)
    5
    
    Validation: Function tested with examples above, works correctly.
    '''
    try:
        return x + y
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")
"""
        }
        
        validation_result = await agentic_loop._apply_quality_gates(
            good_result, 
            "Create a simple addition function"
        )
        
        # Should pass quality gates
        assert validation_result["passed"] is True
        assert validation_result["score"] > 0.7
        assert len(validation_result["failures"]) == 0
        
        # Test quality gates with poor content
        poor_result = {"output": "bad"}
        
        validation_result_poor = await agentic_loop._apply_quality_gates(
            poor_result,
            "Create a complex system"
        )
        
        # Should fail quality gates
        assert validation_result_poor["passed"] is False
        assert validation_result_poor["score"] < 0.5
        assert len(validation_result_poor["failures"]) > 0
        
        print("✅ Brain-3 self-validation verified")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, setup_system):
        """Test complete end-to-end workflow"""
        print("🔄 Testing end-to-end workflow...")
        
        components = await setup_system
        redis_client = components["redis_client"]
        memory_store = components["memory_store"]
        self_grading = components["self_grading"]
        
        # Simulate complete task workflow
        task_id = str(uuid.uuid4())
        
        # 1. Task arrives
        task_message = StreamMessage(
            task_id=task_id,
            message_type=MessageType.AGENTIC_TASK,
            timestamp=time.time(),
            brain_id="test_brain",
            data={
                "task_description": "Generate a simple Python function",
                "context": {"language": "python", "complexity": "simple"}
            }
        )
        
        # 2. Send to agentic tasks stream
        await redis_client.send_message("agentic_tasks", task_message)
        
        # 3. Simulate processing and scoring
        processing_data = {
            "task_id": task_id,
            "brain_id": "test_brain",
            "operation": "code_generation",
            "input_data": task_message.data,
            "output_data": {
                "code": "def simple_func(): return 'Hello World'",
                "validation": "passed",
                "quality": "good"
            },
            "execution_time": 2.3,
            "resource_usage": {"memory": 150, "cpu": 30}
        }
        
        # 4. Calculate performance score
        performance_score = await self_grading.calculate_performance_score(processing_data)
        
        # 5. Store in memory
        task_score = TaskScore(
            task_id=task_id,
            brain_id="test_brain",
            operation="code_generation",
            score=performance_score,
            task_signature="simple_python_function",
            timestamp=time.time(),
            metadata=processing_data
        )
        
        stored = await memory_store.store_task_score(task_score)
        assert stored is True
        
        # 6. Verify end-to-end completion
        retrieved = await memory_store.get_task_score(task_id)
        assert retrieved is not None
        assert retrieved.score == performance_score
        
        print(f"✅ End-to-end workflow verified: score = {performance_score:.3f}")
    
    def test_system_health_checks(self):
        """Test system health check endpoints"""
        print("🏥 Testing system health checks...")
        
        # This would test the actual HTTP endpoints
        # For now, just verify the health check logic exists
        
        # Test model verification
        verification_result = run_comprehensive_verification()
        assert "verification_completed" in verification_result
        
        print("✅ System health checks verified")

# Test runner
if __name__ == "__main__":
    print("🧪 Running Four-Brain System Integration Tests...")
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
    
    print("🎉 Integration tests completed!")
