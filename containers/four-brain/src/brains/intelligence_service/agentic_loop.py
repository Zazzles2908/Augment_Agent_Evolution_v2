#!/usr/bin/env python3
"""
Agentic Loop for Brain-3 Augment with qwen-code CLI Integration
Implements Redis Streams processing with Kimi K2 model via qwen-code CLI

This module provides the agentic loop for Brain-3, integrating with the qwen-code CLI
powered by the Kimi K2 (128K context) model for enhanced code generation and planning.

Zero Fabrication Policy: ENFORCED
All implementations use real qwen-code CLI and Supabase integration.
"""

import os
import asyncio
import subprocess
import json
import time
import uuid
import signal
from typing import Dict, Any, Optional, List
import structlog
import psycopg2
from psycopg2.extras import RealDictCursor

import sys
sys.path.append('/workspace/src')
from shared.redis_client import RedisStreamsClient
from shared.streams import StreamNames, AgenticTask, StreamMessage
from shared.memory_store import MemoryStore, TaskScore

logger = structlog.get_logger(__name__)

class AgenticLoop:
    """Agentic loop for Brain-3 with qwen-code CLI integration"""

    def __init__(self):
        """Initialize agentic loop"""
        self.brain_id = "brain3"
        self.redis_client: Optional[RedisStreamsClient] = None
        self.memory_store: Optional[MemoryStore] = None

        # Supabase connection settings
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.postgres_conn_str = os.getenv("POSTGRES_CONNECTION_STRING")

        # qwen-code CLI settings
        self.k2_api_key = os.getenv("K2_API_KEY")
        self.qwen_code_timeout = 300  # 5 minutes timeout

        # Statistics
        self.tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0

        # Running state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.graceful_shutdown_timeout = 30.0  # 30 seconds for graceful shutdown

        logger.info("Agentic loop initialized", brain_id=self.brain_id)

    async def start(self):
        """Start the agentic loop with graceful shutdown support"""
        try:
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Initialize connections
            await self._initialize_connections()

            # Start processing loop
            self.is_running = True
            logger.info("🚀 Agentic loop started", brain_id=self.brain_id)

            await self._process_loop()

        except Exception as e:
            logger.error("Failed to start agentic loop", error=str(e))
            raise
        finally:
            await self.stop()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self._graceful_shutdown())

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("✅ Signal handlers registered for graceful shutdown")

    async def _graceful_shutdown(self):
        """Perform graceful shutdown to prevent Redis stream lag"""
        try:
            logger.info("🔄 Starting graceful shutdown sequence...")

            # Signal shutdown event
            self.shutdown_event.set()

            # Stop accepting new tasks
            self.is_running = False

            # Wait for current tasks to complete (with timeout)
            logger.info(f"⏳ Waiting up to {self.graceful_shutdown_timeout}s for tasks to complete...")

            # Wait for Redis streams to finish processing
            if self.redis_client:
                try:
                    await asyncio.wait_for(
                        self.redis_client.wait_for_pending_messages(),
                        timeout=self.graceful_shutdown_timeout
                    )
                    logger.info("✅ All Redis stream messages processed")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Graceful shutdown timeout - forcing shutdown")

            await self.stop()

        except Exception as e:
            logger.error(f"❌ Error during graceful shutdown: {str(e)}")
            await self.stop()

    async def stop(self):
        """Stop the agentic loop"""
        logger.info("🛑 Stopping agentic loop...", brain_id=self.brain_id)

        self.is_running = False

        # Disconnect Redis client gracefully
        if self.redis_client:
            try:
                # Stop consuming new messages
                await self.redis_client.stop_consuming()
                # Disconnect
                await self.redis_client.disconnect()
                logger.info("✅ Redis client disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting Redis: {str(e)}")

        # Disconnect memory store
        if self.memory_store:
            try:
                await self.memory_store.disconnect()
                logger.info("✅ Memory store disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting memory store: {str(e)}")

        logger.info("✅ Agentic loop stopped", brain_id=self.brain_id)

    async def _initialize_connections(self):
        """Initialize Redis and memory store connections"""
        # Initialize Redis Streams client
        self.redis_client = RedisStreamsClient(brain_id=self.brain_id)
        await self.redis_client.connect()

        # Register message handler
        self.redis_client.register_handler(
            StreamNames.AGENTIC_TASKS,
            self._handle_agentic_task
        )

        # Initialize memory store
        self.memory_store = MemoryStore(
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key
        )
        await self.memory_store.connect()

        # Start consuming messages
        await self.redis_client.start_consuming()

        logger.info("✅ Agentic loop connections initialized")

    async def _process_loop(self):
        """Main processing loop with graceful shutdown support"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check for shutdown signal
                if self.shutdown_event.is_set():
                    logger.info("🛑 Shutdown signal received, exiting process loop")
                    break

                # Health check
                if not await self.redis_client.health_check():
                    logger.warning("Redis connection unhealthy, attempting reconnect")
                    await self.redis_client.connect()

                # Sleep for a short interval (with shutdown check)
                try:
                    await asyncio.wait_for(asyncio.sleep(1), timeout=1.0)
                except asyncio.TimeoutError:
                    pass  # Continue loop

            except Exception as e:
                logger.error("Error in processing loop", error=str(e))
                # Check for shutdown during error recovery
                if not self.shutdown_event.is_set():
                    await asyncio.sleep(5)  # Wait before retrying
                else:
                    break

    async def _handle_agentic_task(self, message: StreamMessage):
        """Handle an agentic task from Redis Streams"""
        start_time = time.time()
        task_id = message.task_id

        try:
            logger.info("Processing agentic task", task_id=task_id)

            # Extract task data
            task_data = message.data
            task_description = task_data.get('task_description', '')
            context = task_data.get('context', {})

            # Check for past attempts
            past_attempts = await self.memory_store.get_past_attempts(
                "agentic_task",
                {"description": task_description, "context": context}
            )

            # Execute qwen-code CLI
            result = await self._execute_qwen_code(task_description, context, past_attempts)

            # Calculate performance score
            score = self._calculate_performance_score(result, past_attempts)

            # Store result in memory
            await self._store_task_result(task_id, task_description, result, score)

            # Send result to next stream
            await self._send_result(task_id, result, score)

            # Update statistics
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.successful_tasks += 1
            self.total_processing_time += processing_time

            logger.info("Agentic task completed",
                       task_id=task_id, score=score,
                       processing_time=processing_time)

        except Exception as e:
            # Handle failure
            processing_time = time.time() - start_time
            self.tasks_processed += 1
            self.failed_tasks += 1
            self.total_processing_time += processing_time

            logger.error("Agentic task failed",
                        task_id=task_id, error=str(e),
                        processing_time=processing_time)

            # Store failure in memory
            await self._store_task_failure(task_id, str(e))

    async def _execute_qwen_code(self, task_description: str, context: Dict[str, Any],
                               past_attempts) -> Dict[str, Any]:
        """Execute qwen-code CLI for task processing"""
        try:
            # Prepare prompt with context and past attempts
            prompt = self._prepare_prompt(task_description, context, past_attempts)

            # Execute qwen-code CLI
            cmd = [
                "qwen-code",
                "--prompt", prompt,
                "--max-tokens", "4000",
                "--temperature", "0.7"
            ]

            if self.k2_api_key:
                cmd.extend(["--api-key", self.k2_api_key])

            logger.debug("Executing qwen-code CLI", cmd_length=len(cmd))

            # Run subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.qwen_code_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"qwen-code CLI timeout after {self.qwen_code_timeout}s")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"qwen-code CLI failed: {error_msg}")

            # Parse result
            output = stdout.decode().strip()

            # Try to parse as JSON, fallback to text
            try:
                result = json.loads(output)
            except json.JSONDecodeError:
                result = {"output": output, "format": "text"}

            # Apply self-validation quality gates
            validation_result = await self._apply_quality_gates(result, task_description)

            if not validation_result["passed"]:
                logger.warning("Quality gates failed",
                             task_description=task_description,
                             failures=validation_result["failures"])

                # Return error if validation fails
                return {
                    "status": "validation_failed",
                    "error": f"Quality gates failed: {', '.join(validation_result['failures'])}",
                    "result": result,
                    "validation": validation_result,
                    "metadata": {
                        "model": "kimi-k2",
                        "processing_method": "qwen-code-cli",
                        "prompt_length": len(prompt),
                        "validation_applied": True
                    }
                }

            return {
                "status": "success",
                "result": result,
                "validation": validation_result,
                "metadata": {
                    "model": "kimi-k2",
                    "processing_method": "qwen-code-cli",
                    "prompt_length": len(prompt),
                    "validation_applied": True
                }
            }

        except Exception as e:
            logger.error("qwen-code CLI execution failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "model": "kimi-k2",
                    "processing_method": "qwen-code-cli"
                }
            }

    def _prepare_prompt(self, task_description: str, context: Dict[str, Any],
                       past_attempts) -> str:
        """Prepare prompt for qwen-code CLI with enhanced self-validation instructions"""
        prompt_parts = [
            "You are an advanced AI assistant with code generation and planning capabilities.",
            "CRITICAL: You MUST follow the self-validation protocol below for ALL code generation.",
            "",
            "SELF-VALIDATION PROTOCOL:",
            "1. ANALYZE: Understand the task requirements completely",
            "2. DESIGN: Plan the solution architecture and approach",
            "3. IMPLEMENT: Write the initial code implementation",
            "4. VALIDATE: Perform comprehensive self-validation:",
            "   a) Syntax check: Ensure code is syntactically correct",
            "   b) Logic check: Verify the logic flow makes sense",
            "   c) Edge case analysis: Consider boundary conditions and error cases",
            "   d) Example simulation: Mentally execute code with test inputs",
            "   e) Output verification: Confirm outputs match expected results",
            "5. REFINE: If validation fails, fix issues and re-validate",
            "6. DOCUMENT: Include validation results and test examples",
            "7. QUALITY GATE: Only return code that passes ALL validation checks",
            "",
            "VALIDATION REQUIREMENTS:",
            "- Code must be syntactically correct Python",
            "- Logic must handle edge cases and errors gracefully",
            "- Include at least 2 working examples in docstring",
            "- Provide validation_results section with test outcomes",
            "- If any validation step fails, you MUST fix and re-validate",
            "",
            f"Task: {task_description}",
            ""
        ]

        # Add context if available
        if context:
            prompt_parts.extend([
                "Context:",
                json.dumps(context, indent=2),
                ""
            ])

        # Add past attempts for learning
        if past_attempts.similar_tasks:
            prompt_parts.extend([
                "Past attempts for similar tasks:",
                f"Average score: {past_attempts.average_score:.3f}",
                f"Best score: {past_attempts.best_score:.3f}",
                f"Attempts: {past_attempts.attempt_count}",
                ""
            ])

        prompt_parts.extend([
            "Please provide a comprehensive solution with:",
            "- Clear explanation of approach",
            "- Implementation details",
            "- Expected outcomes",
            "- Self-validation results"
        ])

        return "\n".join(prompt_parts)

    def _calculate_performance_score(self, result: Dict[str, Any], past_attempts) -> float:
        """Calculate performance score for the task result with validation consideration"""
        if result.get("status") == "error":
            return 0.1  # Low score for errors

        if result.get("status") == "validation_failed":
            return 0.2  # Low score for validation failures

        # Base score for successful completion
        base_score = 0.7

        # Validation score integration
        if "validation" in result:
            validation = result["validation"]
            if validation.get("passed", False):
                # Use validation score as a component
                validation_score = validation.get("score", 0.7)
                base_score = (base_score + validation_score) / 2

                # Bonus for comprehensive validation
                checks_performed = len(validation.get("checks_performed", []))
                if checks_performed >= 4:
                    base_score += 0.1
            else:
                # Penalty for failed validation
                base_score *= 0.5

        # Bonus for comprehensive results
        if "result" in result and isinstance(result["result"], dict):
            if len(result["result"]) > 3:  # Multiple components
                base_score += 0.05

        # Bonus for self-validation metadata
        if "validation_applied" in result.get("metadata", {}):
            base_score += 0.05

        # Compare with past attempts
        if past_attempts.similar_tasks and past_attempts.average_score > 0:
            if base_score > past_attempts.average_score:
                base_score += 0.1  # Improvement bonus

        return min(1.0, base_score)

    async def _apply_quality_gates(self, result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Apply quality gates to validate generated code/content"""
        validation_result = {
            "passed": True,
            "failures": [],
            "checks_performed": [],
            "score": 1.0
        }

        try:
            # Extract content for validation
            content = ""
            if isinstance(result, dict):
                if "output" in result:
                    content = result["output"]
                elif "code" in result:
                    content = result["code"]
                elif "solution" in result:
                    content = result["solution"]
                else:
                    content = str(result)
            else:
                content = str(result)

            # Quality Gate 1: Content completeness
            validation_result["checks_performed"].append("content_completeness")
            if len(content.strip()) < 10:
                validation_result["failures"].append("Content too short or empty")
                validation_result["passed"] = False

            # Quality Gate 2: Python code syntax validation (if content contains code)
            if "def " in content or "class " in content or "import " in content:
                validation_result["checks_performed"].append("python_syntax")
                syntax_valid = self._validate_python_syntax(content)
                if not syntax_valid:
                    validation_result["failures"].append("Python syntax errors detected")
                    validation_result["passed"] = False

            # Quality Gate 3: Self-validation evidence
            validation_result["checks_performed"].append("self_validation_evidence")
            validation_keywords = [
                "validation", "test", "example", "verified", "checked",
                "simulation", "output", "result", "works", "passes"
            ]
            has_validation_evidence = any(keyword in content.lower() for keyword in validation_keywords)
            if not has_validation_evidence:
                validation_result["failures"].append("No self-validation evidence found")
                validation_result["passed"] = False

            # Quality Gate 4: Task relevance
            validation_result["checks_performed"].append("task_relevance")
            task_keywords = task_description.lower().split()[:5]  # First 5 words
            relevance_score = sum(1 for keyword in task_keywords if keyword in content.lower()) / max(len(task_keywords), 1)
            if relevance_score < 0.3:
                validation_result["failures"].append("Content not relevant to task")
                validation_result["passed"] = False

            # Quality Gate 5: Error handling presence (for code)
            if "def " in content:
                validation_result["checks_performed"].append("error_handling")
                has_error_handling = any(keyword in content.lower() for keyword in ["try:", "except", "raise", "error", "exception"])
                if not has_error_handling:
                    validation_result["failures"].append("No error handling detected in code")
                    # This is a warning, not a failure
                    validation_result["score"] *= 0.9

            # Calculate final score
            if validation_result["passed"]:
                validation_result["score"] = max(0.7, validation_result["score"])  # Minimum passing score
            else:
                validation_result["score"] = 0.3  # Failed validation score

            logger.debug("Quality gates validation completed",
                        passed=validation_result["passed"],
                        checks=len(validation_result["checks_performed"]),
                        failures=len(validation_result["failures"]))

            return validation_result

        except Exception as e:
            logger.error("Quality gates validation failed", error=str(e))
            return {
                "passed": False,
                "failures": [f"Validation error: {str(e)}"],
                "checks_performed": ["validation_error"],
                "score": 0.1
            }

    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python code syntax"""
        try:
            import ast

            # Extract Python code blocks if content contains markdown
            code_blocks = []
            if "```python" in code:
                lines = code.split('\n')
                in_code_block = False
                current_block = []

                for line in lines:
                    if line.strip().startswith("```python"):
                        in_code_block = True
                        current_block = []
                    elif line.strip() == "```" and in_code_block:
                        in_code_block = False
                        if current_block:
                            code_blocks.append('\n'.join(current_block))
                    elif in_code_block:
                        current_block.append(line)
            else:
                # Assume entire content is code
                code_blocks = [code]

            # Validate each code block
            for block in code_blocks:
                if block.strip():
                    ast.parse(block)

            return True

        except SyntaxError as e:
            logger.debug("Python syntax validation failed", error=str(e))
            return False
        except Exception as e:
            logger.debug("Python syntax validation error", error=str(e))
            return True  # Give benefit of doubt for non-syntax errors

    async def _store_task_result(self, task_id: str, task_description: str,
                               result: Dict[str, Any], score: float):
        """Store task result in agentic_memory table"""
        try:
            if not self.postgres_conn_str:
                logger.warning("No PostgreSQL connection string, skipping memory storage")
                return

            conn = psycopg2.connect(self.postgres_conn_str)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agentic_memory (task_id, prompt, outcome_score, metadata)
                VALUES (%s, %s, %s, %s)
            """, (
                task_id,
                task_description,
                score,
                json.dumps(result)
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.debug("Task result stored in agentic_memory", task_id=task_id, score=score)

        except Exception as e:
            logger.error("Failed to store task result", task_id=task_id, error=str(e))

    async def _store_task_failure(self, task_id: str, error: str):
        """Store task failure in agentic_memory table"""
        await self._store_task_result(task_id, f"FAILED: {error}", {}, 0.0)

    async def _send_result(self, task_id: str, result: Dict[str, Any], score: float):
        """Send result to the next stream in the pipeline"""
        try:
            result_message = StreamMessage(
                task_id=task_id,
                message_type="agentic_result",
                timestamp=time.time(),
                brain_id=self.brain_id,
                data={
                    "result": result,
                    "score": score,
                    "brain_id": self.brain_id
                }
            )

            await self.redis_client.send_message(
                StreamNames.AGENTIC_RESULTS,
                result_message
            )

            logger.debug("Result sent to agentic_results stream", task_id=task_id)

        except Exception as e:
            logger.error("Failed to send result", task_id=task_id, error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get agentic loop statistics"""
        avg_processing_time = (
            self.total_processing_time / self.tasks_processed
            if self.tasks_processed > 0 else 0.0
        )

        success_rate = (
            self.successful_tasks / self.tasks_processed
            if self.tasks_processed > 0 else 0.0
        )

        return {
            "brain_id": self.brain_id,
            "is_running": self.is_running,
            "tasks_processed": self.tasks_processed,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time
        }

# Global agentic loop instance
_agentic_loop: Optional[AgenticLoop] = None

def get_agentic_loop() -> AgenticLoop:
    """Get or create the global agentic loop instance"""
    global _agentic_loop
    if _agentic_loop is None:
        _agentic_loop = AgenticLoop()
    return _agentic_loop

if __name__ == "__main__":
    # Run the agentic loop
    loop = get_agentic_loop()
    asyncio.run(loop.start())