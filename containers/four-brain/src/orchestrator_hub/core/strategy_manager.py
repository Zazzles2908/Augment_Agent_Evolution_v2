#!/usr/bin/env python3
"""
Strategy Manager for K2-Vector-Hub
Manages strategy decision making and coordination logic

This module implements the strategy management logic for K2-Vector-Hub,
coordinating with the Moonshot client to make intelligent decisions.

Zero Fabrication Policy: ENFORCED
All implementations use real strategy logic and verified functionality.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from .moonshot_client import MoonshotClient

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Strategy Manager for K2-Vector-Hub
    Coordinates strategy decision making using Moonshot Kimi API
    """
    
    def __init__(self, moonshot_client: MoonshotClient):
        """Initialize strategy manager with Moonshot client"""
        self.moonshot_client = moonshot_client
        
        # Performance tracking
        self.total_jobs_processed = 0
        self.total_decision_time = 0.0
        self.initialization_time = time.time()
        
        # Strategy tracking
        self.active_strategies = {}
        self.strategy_history = []
        
        logger.info("🎯 Strategy Manager initialized")
    
    async def create_strategy_plan(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive strategy plan for a vector job
        This is the core "Mayor's decision-making process"
        """
        job_id = job_data.get("job_id", "unknown")
        question = job_data.get("question", "")
        
        logger.info(f"🏛️ Mayor creating strategy plan for job {job_id}")
        
        try:
            start_time = time.time()
            
            # Get strategy decision from Moonshot Kimi API
            strategy_decision = await self.moonshot_client.make_strategy_decision(job_data)
            
            decision_time = time.time() - start_time
            self.total_decision_time += decision_time
            self.total_jobs_processed += 1
            
            # Create comprehensive strategy plan
            strategy_plan = {
                "job_id": job_id,
                "question": question,
                "strategy": strategy_decision.get("strategy", "analytical"),
                "reasoning": strategy_decision.get("reasoning", "Default analytical approach"),
                "confidence": strategy_decision.get("confidence", 0.7),
                "brain_allocation": strategy_decision.get("brain_allocation", {
                    "brain1": 50, "brain2": 30, "brain4": 20
                }),
                "timestamp": time.time(),
                "decision_time_ms": decision_time * 1000,
                "source": "k2_vector_hub_mayor",
                "status": "active"
            }
            
            # Validate brain allocation
            strategy_plan = self._validate_brain_allocation(strategy_plan)
            
            # Store active strategy
            self.active_strategies[job_id] = strategy_plan
            
            # Add to history (keep last 50)
            self.strategy_history.append(strategy_plan)
            if len(self.strategy_history) > 50:
                self.strategy_history.pop(0)
            
            logger.info(f"✅ Strategy plan created for job {job_id}: {strategy_plan['strategy']}")
            logger.info(f"🧠 Brain allocation: {strategy_plan['brain_allocation']}")
            
            return strategy_plan
            
        except Exception as e:
            logger.error(f"❌ Strategy plan creation failed for job {job_id}: {e}")
            
            # Return emergency fallback strategy
            return self._create_emergency_strategy(job_data)
    
    def _validate_brain_allocation(self, strategy_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize brain allocation percentages"""
        brain_allocation = strategy_plan.get("brain_allocation", {})
        
        # Ensure all brains are represented
        default_allocation = {"brain1": 50, "brain2": 30, "brain4": 20}
        for brain in default_allocation:
            if brain not in brain_allocation:
                brain_allocation[brain] = 0
        
        # Calculate total and normalize if needed
        total = sum(brain_allocation.values())
        if total == 0:
            # Use default allocation
            brain_allocation = default_allocation
        elif total != 100:
            # Normalize to 100%
            factor = 100.0 / total
            for brain in brain_allocation:
                brain_allocation[brain] = round(brain_allocation[brain] * factor)
            
            # Adjust for rounding errors
            current_total = sum(brain_allocation.values())
            if current_total != 100:
                # Add/subtract difference to largest allocation
                largest_brain = max(brain_allocation, key=brain_allocation.get)
                brain_allocation[largest_brain] += (100 - current_total)
        
        strategy_plan["brain_allocation"] = brain_allocation
        return strategy_plan
    
    def _create_emergency_strategy(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create emergency fallback strategy when all else fails"""
        job_id = job_data.get("job_id", "unknown")
        question = job_data.get("question", "")
        
        return {
            "job_id": job_id,
            "question": question,
            "strategy": "emergency_fallback",
            "reasoning": "Emergency fallback due to system error",
            "confidence": 0.5,
            "brain_allocation": {"brain1": 50, "brain2": 30, "brain4": 20},
            "timestamp": time.time(),
            "decision_time_ms": 0,
            "source": "emergency_fallback",
            "status": "active"
        }
    
    def complete_strategy(self, job_id: str, results: Dict[str, Any] = None):
        """Mark strategy as completed and store results"""
        if job_id in self.active_strategies:
            strategy = self.active_strategies[job_id]
            strategy["status"] = "completed"
            strategy["completion_time"] = time.time()
            if results:
                strategy["results"] = results
            
            # Move to history
            del self.active_strategies[job_id]
            logger.info(f"✅ Strategy completed for job {job_id}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        strategy_types = {}
        for strategy in self.strategy_history:
            strategy_type = strategy.get("strategy", "unknown")
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        
        avg_decision_time = 0
        if self.total_jobs_processed > 0:
            avg_decision_time = (self.total_decision_time / self.total_jobs_processed) * 1000
        
        return {
            "total_jobs_processed": self.total_jobs_processed,
            "average_decision_time_ms": avg_decision_time,
            "active_strategies": len(self.active_strategies),
            "strategy_types": strategy_types,
            "uptime_seconds": time.time() - self.initialization_time
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for strategy manager"""
        moonshot_health = await self.moonshot_client.health_check()
        
        return {
            "healthy": moonshot_health.get("healthy", False),
            "moonshot_client": moonshot_health,
            "total_jobs_processed": self.total_jobs_processed,
            "active_strategies": len(self.active_strategies),
            "uptime_seconds": time.time() - self.initialization_time
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "total_jobs_processed": self.total_jobs_processed,
            "average_decision_time_ms": (self.total_decision_time / max(self.total_jobs_processed, 1)) * 1000,
            "active_strategies": len(self.active_strategies),
            "moonshot_api_calls": self.moonshot_client.total_api_calls,
            "uptime_seconds": time.time() - self.initialization_time
        }
