#!/usr/bin/env python3
"""
Moonshot Kimi API Client for K2-Vector-Hub
Real integration with Moonshot's Kimi API for strategy decision making

This module implements the Moonshot Kimi API client as specified in fix_containers.md
for making intelligent strategy decisions in the Four-Brain Architecture.

Zero Fabrication Policy: ENFORCED
All implementations use real Moonshot API calls and verified functionality.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp
import json

logger = logging.getLogger(__name__)

class MoonshotClient:
    """
    Moonshot Kimi API Client for strategy decision making
    Implements real API integration for K2-Vector-Hub
    """
    
    def __init__(self):
        """Initialize Moonshot client with environment configuration"""
        self.api_key = os.getenv("K2_API_KEY")
        self.api_url = os.getenv("K2_API_URL", "https://api.moonshot.cn/v1/chat/completions")
        self.model = os.getenv("K2_MODEL", "kimi-thinking-preview")
        self.timeout = int(os.getenv("K2_TIMEOUT", "30"))
        self.max_tokens = int(os.getenv("K2_MAX_TOKENS", "3000"))
        
        # Performance tracking
        self.total_api_calls = 0
        self.total_processing_time = 0.0
        self.initialization_time = time.time()
        
        # Connection state
        self.initialized = False
        
        logger.info(f"🌙 Moonshot client initialized with model: {self.model}")
    
    async def initialize(self) -> bool:
        """Initialize and test Moonshot API connection"""
        if not self.api_key:
            logger.error("❌ K2_API_KEY not set - Moonshot client cannot initialize")
            return False
        
        try:
            # Test API connection with a simple request
            test_response = await self._make_api_call(
                "Test connection",
                system_prompt="Respond with 'OK' to confirm API connectivity."
            )
            
            if test_response and "OK" in test_response.get("content", ""):
                self.initialized = True
                logger.info("✅ Moonshot API connection verified")
                return True
            else:
                logger.warning("⚠️ Moonshot API test response unexpected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Moonshot API initialization failed: {e}")
            return False
    
    async def make_strategy_decision(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make strategy decision using Moonshot Kimi API
        Implements the "Mayor's decision-making process" from fix_containers.md
        """
        if not self.initialized:
            raise RuntimeError("Moonshot client not initialized")
        
        question = job_data.get("question", "")
        user_context = job_data.get("user_context", {})
        job_id = job_data.get("job_id", "unknown")
        
        # Create strategy prompt for Kimi
        strategy_prompt = self._create_strategy_prompt(question, user_context)
        
        system_prompt = """You are the Mayor of a Four-Brain AI City. Your job is to decide how to allocate work across four specialized brains:

Brain 1 (Embedding): Converts text to vectors using Qwen3-4B
Brain 2 (Reranker): Scores and ranks document relevance using Qwen3-Reranker-4B  
Brain 3 (Augment): Orchestrates and manages the overall workflow
Brain 4 (Docling): Processes documents and PDFs into text

For each user question, decide the optimal allocation percentages that sum to 100%.
Respond with a JSON object containing:
- strategy: brief strategy name (e.g., "analytical", "document_heavy", "search_focused")
- reasoning: explanation of your decision
- confidence: confidence score 0.0-1.0
- brain_allocation: {"brain1": X, "brain2": Y, "brain4": Z} where X+Y+Z=100

Example allocations:
- Text analysis: {"brain1": 60, "brain2": 30, "brain4": 10}
- Document processing: {"brain1": 30, "brain2": 20, "brain4": 50}
- Search/ranking: {"brain1": 40, "brain2": 50, "brain4": 10}"""
        
        try:
            start_time = time.time()
            
            # Make API call to Moonshot Kimi
            response = await self._make_api_call(strategy_prompt, system_prompt)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            if response:
                # Parse strategy decision from response
                strategy_decision = self._parse_strategy_response(response, job_data)
                
                logger.info(f"🎯 Strategy decision for job {job_id}: {strategy_decision.get('strategy')}")
                logger.info(f"⚡ Decision time: {processing_time:.3f}s")
                
                return strategy_decision
            else:
                raise Exception("No response from Moonshot API")
                
        except Exception as e:
            logger.error(f"❌ Strategy decision failed for job {job_id}: {e}")
            # Return fallback strategy
            return self._create_fallback_strategy(job_data)
    
    def _create_strategy_prompt(self, question: str, user_context: Dict[str, Any]) -> str:
        """Create strategy prompt for Moonshot Kimi API"""
        context_info = ""
        if user_context:
            context_info = f"\nUser Context: {json.dumps(user_context, indent=2)}"
        
        return f"""User Question: {question}{context_info}

Please analyze this request and determine the optimal allocation of work across the Four-Brain system."""
    
    async def _make_api_call(self, user_prompt: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Make actual API call to Moonshot Kimi"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.total_api_calls += 1
                        
                        # Extract content from response
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            return {"content": content, "usage": result.get("usage", {})}
                        else:
                            logger.warning("⚠️ Unexpected Moonshot API response format")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Moonshot API error {response.status}: {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"❌ Moonshot API timeout after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"❌ Moonshot API call failed: {e}")
            return None
    
    def _parse_strategy_response(self, response: Dict[str, Any], job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse strategy decision from Moonshot API response"""
        content = response.get("content", "")
        
        try:
            # Try to extract JSON from response
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                strategy_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["strategy", "reasoning", "confidence", "brain_allocation"]
                if all(field in strategy_data for field in required_fields):
                    # Add job metadata
                    strategy_data["job_id"] = job_data.get("job_id")
                    strategy_data["question"] = job_data.get("question")
                    strategy_data["timestamp"] = time.time()
                    strategy_data["source"] = "moonshot_kimi"
                    
                    return strategy_data
                else:
                    logger.warning("⚠️ Strategy response missing required fields")
                    return self._create_fallback_strategy(job_data)
            else:
                logger.warning("⚠️ No JSON found in strategy response")
                return self._create_fallback_strategy(job_data)
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse strategy JSON: {e}")
            return self._create_fallback_strategy(job_data)
    
    def _create_fallback_strategy(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback strategy when Moonshot API fails"""
        question = job_data.get("question", "").lower()
        
        # Simple heuristic-based fallback
        if any(word in question for word in ["pdf", "document", "file", "upload"]):
            # Document-heavy strategy
            brain_allocation = {"brain1": 30, "brain2": 20, "brain4": 50}
            strategy = "document_heavy"
        elif any(word in question for word in ["search", "find", "rank", "best"]):
            # Search-focused strategy
            brain_allocation = {"brain1": 40, "brain2": 50, "brain4": 10}
            strategy = "search_focused"
        else:
            # Balanced analytical strategy
            brain_allocation = {"brain1": 50, "brain2": 30, "brain4": 20}
            strategy = "analytical"
        
        return {
            "job_id": job_data.get("job_id"),
            "question": job_data.get("question"),
            "strategy": strategy,
            "reasoning": "Fallback strategy due to API unavailability",
            "confidence": 0.6,
            "brain_allocation": brain_allocation,
            "timestamp": time.time(),
            "source": "fallback_heuristic"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Moonshot client"""
        return {
            "healthy": self.initialized,
            "api_key_configured": bool(self.api_key),
            "total_api_calls": self.total_api_calls,
            "average_processing_time_ms": (self.total_processing_time / max(self.total_api_calls, 1)) * 1000,
            "uptime_seconds": time.time() - self.initialization_time
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "initialized": self.initialized,
            "api_url": self.api_url,
            "model": self.model,
            "total_api_calls": self.total_api_calls,
            "average_processing_time_ms": (self.total_processing_time / max(self.total_api_calls, 1)) * 1000,
            "uptime_seconds": time.time() - self.initialization_time
        }
