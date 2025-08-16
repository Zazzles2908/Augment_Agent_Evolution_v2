"""
GLM Client for Z.AI Platform Integration
Implements GLM-4.5 API client with thinking mode and code generation capabilities
"""

import os
import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class GLMClient:
    """
    GLM-4.5 API Client for Z.AI Platform
    
    Supports:
    - Chat completions with thinking mode
    - Code generation and verification
    - Async operations with proper timeout handling
    - Agent API integration
    """
    
    def __init__(self):
        """Initialize GLM client with environment configuration"""
        self.api_key = os.getenv("GLM_API_KEY")
        self.api_url = os.getenv("GLM_API_URL", "https://api.z.ai/api/paas/v4/chat/completions")
        self.agent_api_url = os.getenv("GLM_AGENT_API_URL", "https://api.z.ai/api/v1/agents")
        
        # Model configuration
        self.model = os.getenv("GLM_MODEL", "glm-4-flash")
        self.model_secondary = os.getenv("GLM_MODEL_SECONDARY", "glm-4-flash")
        
        # Request parameters
        self.max_tokens = int(os.getenv("GLM_MAX_TOKENS", "8192"))
        self.temperature = float(os.getenv("GLM_TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("GLM_TOP_P", "0.8"))
        self.timeout = int(os.getenv("GLM_TIMEOUT", "60"))
        
        # Thinking mode configuration
        self.thinking_enabled = os.getenv("GLM_THINKING_ENABLED", "true").lower() == "true"
        
        # Async configuration
        self.async_timeout = int(os.getenv("GLM_ASYNC_RESULT_TIMEOUT", "300"))
        self.poll_interval = int(os.getenv("GLM_ASYNC_POLL_INTERVAL", "2"))
        
        # Validation
        if not self.api_key:
            raise ValueError("GLM_API_KEY environment variable is required")
        
        logger.info(f"🤖 GLM Client initialized - Model: {self.model}, Thinking: {self.thinking_enabled}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Language": "en-US,en"
        }
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Create chat completion using GLM-4.5
        
        Args:
            messages: List of message objects with role and content
            system_prompt: Optional system prompt to prepend
            thinking_mode: Override default thinking mode setting
            model: Override default model
            stream: Enable streaming response
            
        Returns:
            Chat completion response
        """
        try:
            # Prepare messages
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            formatted_messages.extend(messages)
            
            # Prepare request payload
            payload = {
                "model": model or self.model,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": stream
            }
            
            # Add thinking mode configuration
            thinking_enabled = thinking_mode if thinking_mode is not None else self.thinking_enabled
            payload["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled"
            }
            
            logger.debug(f"🔄 GLM API request: {self.model}, thinking: {thinking_enabled}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"GLM API error {response.status}: {error_text}")
                    
                    if stream:
                        return await self._handle_streaming_response(response)
                    else:
                        result = await response.json()
                        logger.debug(f"✅ GLM API response received")
                        return result
                        
        except asyncio.TimeoutError:
            logger.error(f"❌ GLM API timeout after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"❌ GLM API error: {e}")
            raise
    
    async def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from GLM API"""
        content_parts = []
        
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data == '[DONE]':
                    break
                
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content_parts.append(delta['content'])
                except json.JSONDecodeError:
                    continue
        
        # Reconstruct complete response
        complete_content = ''.join(content_parts)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": complete_content
                },
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": len(complete_content.split())}  # Approximate
        }
    
    async def generate_code(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        requirements: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code using GLM-4.5 with enhanced prompting
        
        Args:
            prompt: Code generation prompt
            system_prompt: System prompt for code generation context
            requirements: Specific requirements for the code
            context: Additional context about the project/system
            
        Returns:
            Generated code with metadata
        """
        # Default system prompt for code generation
        if not system_prompt:
            system_prompt = """
You are an expert Python developer specializing in AI systems and containerized applications.
Generate clean, efficient, and secure code that follows best practices.
Include appropriate error handling, type hints, and comprehensive comments.
Focus on real-world development needs, avoiding templated or generic outputs.
Ensure code is production-ready and follows PEP 8 standards.
"""
        
        # Enhanced prompt with context
        enhanced_prompt = f"""
{f"Context: {context}" if context else ""}
{f"Requirements: {requirements}" if requirements else ""}

Task: {prompt}

Please provide:
1. Complete, working code
2. Brief explanation of the approach
3. Any important considerations or limitations
"""
        
        messages = [{"role": "user", "content": enhanced_prompt.strip()}]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                thinking_mode=True  # Enable thinking for complex code generation
            )
            
            content = response['choices'][0]['message']['content']
            
            return {
                "success": True,
                "code": content,
                "model_used": self.model,
                "thinking_enabled": True,
                "timestamp": datetime.utcnow().isoformat(),
                "usage": response.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def verify_code(
        self, 
        code: str, 
        requirements: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify generated code against requirements using GLM-4.5
        
        Args:
            code: Code to verify
            requirements: Original requirements
            context: Additional context for verification
            
        Returns:
            Verification result with detailed assessment
        """
        verification_prompt = f"""
Please verify this code against the given requirements and provide a detailed assessment.

{f"Context: {context}" if context else ""}

Requirements:
{requirements}

Code to verify:
```python
{code}
```

Please check for:
1. Correctness: Does the code implement the requirements correctly?
2. Security: Are there any security vulnerabilities or concerns?
3. Performance: Are there obvious performance issues or inefficiencies?
4. Best Practices: Does the code follow Python best practices and PEP 8?
5. Error Handling: Is error handling appropriate and comprehensive?
6. Maintainability: Is the code well-structured and maintainable?

Provide a structured assessment with specific issues found and recommendations.
"""
        
        messages = [{"role": "user", "content": verification_prompt}]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                thinking_mode=True  # Enable thinking for thorough analysis
            )
            
            assessment = response['choices'][0]['message']['content']
            
            # Parse assessment into structured format
            verification_result = self._parse_verification_result(assessment)
            verification_result.update({
                "raw_assessment": assessment,
                "timestamp": datetime.utcnow().isoformat(),
                "usage": response.get('usage', {})
            })
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Code verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _parse_verification_result(self, assessment: str) -> Dict[str, Any]:
        """
        Parse verification assessment into structured format
        
        Args:
            assessment: Raw assessment text from GLM
            
        Returns:
            Structured verification result
        """
        lines = assessment.lower().split('\n')
        
        result = {
            "success": True,
            "overall_score": 0.0,
            "categories": {
                "correctness": {"score": 1.0, "issues": []},
                "security": {"score": 1.0, "issues": []},
                "performance": {"score": 1.0, "issues": []},
                "best_practices": {"score": 1.0, "issues": []},
                "error_handling": {"score": 1.0, "issues": []},
                "maintainability": {"score": 1.0, "issues": []}
            },
            "recommendations": [],
            "critical_issues": []
        }
        
        # Simple parsing logic - can be enhanced with more sophisticated NLP
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for issues in different categories
            if any(keyword in line for keyword in ["incorrect", "wrong", "bug", "error in logic"]):
                result["categories"]["correctness"]["score"] = 0.5
                result["categories"]["correctness"]["issues"].append(line)
                
            elif any(keyword in line for keyword in ["security", "vulnerable", "unsafe", "injection"]):
                result["categories"]["security"]["score"] = 0.5
                result["categories"]["security"]["issues"].append(line)
                result["critical_issues"].append(line)
                
            elif any(keyword in line for keyword in ["performance", "slow", "inefficient", "memory"]):
                result["categories"]["performance"]["score"] = 0.7
                result["categories"]["performance"]["issues"].append(line)
                
            elif any(keyword in line for keyword in ["best practice", "pep 8", "style", "convention"]):
                result["categories"]["best_practices"]["score"] = 0.8
                result["categories"]["best_practices"]["issues"].append(line)
                
            elif any(keyword in line for keyword in ["error handling", "exception", "try-catch"]):
                result["categories"]["error_handling"]["score"] = 0.7
                result["categories"]["error_handling"]["issues"].append(line)
                
            elif "recommend" in line:
                result["recommendations"].append(line)
        
        # Calculate overall score
        scores = [cat["score"] for cat in result["categories"].values()]
        result["overall_score"] = sum(scores) / len(scores)
        
        # Determine if verification passed
        result["success"] = result["overall_score"] >= 0.7 and len(result["critical_issues"]) == 0
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on GLM API
        
        Returns:
            Health check result
        """
        try:
            test_messages = [{"role": "user", "content": "Hello, please respond with 'OK' to confirm the API is working."}]
            
            start_time = time.time()
            response = await self.chat_completion(
                messages=test_messages,
                thinking_mode=False  # Simple test, no thinking needed
            )
            response_time = time.time() - start_time
            
            content = response['choices'][0]['message']['content'].strip().lower()
            is_healthy = 'ok' in content or 'working' in content
            
            return {
                "healthy": is_healthy,
                "response_time_ms": round(response_time * 1000, 2),
                "model": self.model,
                "api_url": self.api_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ GLM health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
