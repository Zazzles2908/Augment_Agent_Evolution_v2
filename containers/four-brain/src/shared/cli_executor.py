#!/usr/bin/env python3
"""
CLI Tool Executor for Four-Brain Architecture
Enables CLI tool execution within containers and tool creation capabilities

This module provides CLI tool execution functionality for the Four-Brain System,
enabling Brain-3 to execute command-line tools and create new tools dynamically
for enhanced problem-solving capabilities.

Zero Fabrication Policy: ENFORCED
All CLI executions use real subprocess calls with proper security controls.
"""

import os
import subprocess
import asyncio
import tempfile
import shutil
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class ToolType(Enum):
    """Types of CLI tools"""
    SYSTEM = "system"
    PYTHON = "python"
    SHELL = "shell"
    CUSTOM = "custom"

class ExecutionMode(Enum):
    """Execution modes for CLI tools"""
    SYNC = "sync"
    ASYNC = "async"
    BACKGROUND = "background"

@dataclass
class ToolDefinition:
    """Definition of a CLI tool"""
    name: str
    tool_type: ToolType
    command_template: str
    description: str
    parameters: Dict[str, Any]
    timeout: int = 30
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = None
    security_level: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}

@dataclass
class ExecutionResult:
    """Result of CLI tool execution"""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    tool_name: str
    command_executed: str
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'return_code': self.return_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'execution_time': self.execution_time,
            'tool_name': self.tool_name,
            'command_executed': self.command_executed,
            'error_message': self.error_message
        }

class CLIExecutor:
    """CLI tool executor with security controls"""
    
    def __init__(self, workspace_dir: str = "/workspace", 
                 max_concurrent_executions: int = 5):
        """Initialize CLI executor"""
        self.workspace_dir = workspace_dir
        self.max_concurrent_executions = max_concurrent_executions
        
        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}
        
        # Security settings
        self.allowed_commands = {
            "low": ["echo", "cat", "ls", "pwd", "date", "whoami"],
            "medium": ["python", "pip", "git", "curl", "wget", "grep", "find", "sort"],
            "high": ["docker", "systemctl", "sudo", "chmod", "chown"]
        }
        
        # Execution tracking
        self.active_executions = 0
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # Initialize built-in tools
        self._initialize_builtin_tools()
        
        logger.info("CLI executor initialized", workspace_dir=workspace_dir)
    
    def _initialize_builtin_tools(self):
        """Initialize built-in CLI tools"""
        
        # Python script executor
        self.register_tool(ToolDefinition(
            name="python_script",
            tool_type=ToolType.PYTHON,
            command_template="python -c \"{script}\"",
            description="Execute Python code snippets",
            parameters={"script": {"type": "string", "required": True}},
            timeout=60,
            security_level="medium"
        ))
        
        # File operations
        self.register_tool(ToolDefinition(
            name="list_files",
            tool_type=ToolType.SYSTEM,
            command_template="ls -la {path}",
            description="List files in directory",
            parameters={"path": {"type": "string", "default": "."}},
            timeout=10,
            security_level="low"
        ))
        
        # Text processing
        self.register_tool(ToolDefinition(
            name="grep_search",
            tool_type=ToolType.SYSTEM,
            command_template="grep -r \"{pattern}\" {path}",
            description="Search for text patterns in files",
            parameters={
                "pattern": {"type": "string", "required": True},
                "path": {"type": "string", "default": "."}
            },
            timeout=30,
            security_level="medium"
        ))
        
        # Git operations
        self.register_tool(ToolDefinition(
            name="git_status",
            tool_type=ToolType.SYSTEM,
            command_template="git status",
            description="Get git repository status",
            parameters={},
            timeout=15,
            security_level="medium"
        ))
        
        # System information
        self.register_tool(ToolDefinition(
            name="system_info",
            tool_type=ToolType.SYSTEM,
            command_template="uname -a && df -h && free -h",
            description="Get system information",
            parameters={},
            timeout=10,
            security_level="low"
        ))
    
    def register_tool(self, tool: ToolDefinition):
        """Register a new CLI tool"""
        self.tools[tool.name] = tool
        logger.info("Tool registered", tool_name=tool.name, tool_type=tool.tool_type.value)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [
            {
                'name': tool.name,
                'type': tool.tool_type.value,
                'description': tool.description,
                'parameters': tool.parameters,
                'security_level': tool.security_level
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any],
                          mode: ExecutionMode = ExecutionMode.SYNC) -> ExecutionResult:
        """Execute a CLI tool with given parameters"""
        
        if self.active_executions >= self.max_concurrent_executions:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="Maximum concurrent executions reached",
                execution_time=0.0,
                tool_name=tool_name,
                command_executed="",
                error_message="Execution limit exceeded"
            )
        
        if tool_name not in self.tools:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Tool '{tool_name}' not found",
                execution_time=0.0,
                tool_name=tool_name,
                command_executed="",
                error_message="Tool not found"
            )
        
        tool = self.tools[tool_name]
        
        try:
            # Validate parameters
            validation_result = self._validate_parameters(tool, parameters)
            if not validation_result[0]:
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=validation_result[1],
                    execution_time=0.0,
                    tool_name=tool_name,
                    command_executed="",
                    error_message="Parameter validation failed"
                )
            
            # Build command
            command = self._build_command(tool, parameters)
            
            # Security check
            if not self._security_check(tool, command):
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr="Security check failed",
                    execution_time=0.0,
                    tool_name=tool_name,
                    command_executed=command,
                    error_message="Security violation"
                )
            
            # Execute command
            self.active_executions += 1
            self.total_executions += 1
            
            start_time = time.time()
            
            if mode == ExecutionMode.ASYNC:
                result = await self._execute_async(tool, command)
            else:
                result = await self._execute_sync(tool, command)
            
            execution_time = time.time() - start_time
            
            # Update statistics
            if result.success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            result.execution_time = execution_time
            result.tool_name = tool_name
            result.command_executed = command
            
            logger.info("Tool executed", tool_name=tool_name, 
                       success=result.success, execution_time=execution_time)
            
            return result
            
        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            self.failed_executions += 1
            
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                tool_name=tool_name,
                command_executed="",
                error_message=str(e)
            )
        
        finally:
            self.active_executions -= 1
    
    def _validate_parameters(self, tool: ToolDefinition, 
                           parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool parameters"""
        for param_name, param_config in tool.parameters.items():
            if param_config.get("required", False) and param_name not in parameters:
                return False, f"Required parameter '{param_name}' missing"
            
            if param_name in parameters:
                param_value = parameters[param_name]
                param_type = param_config.get("type", "string")
                
                # Type validation
                if param_type == "string" and not isinstance(param_value, str):
                    return False, f"Parameter '{param_name}' must be a string"
                elif param_type == "int" and not isinstance(param_value, int):
                    return False, f"Parameter '{param_name}' must be an integer"
                elif param_type == "float" and not isinstance(param_value, (int, float)):
                    return False, f"Parameter '{param_name}' must be a number"
        
        return True, "Valid"
    
    def _build_command(self, tool: ToolDefinition, parameters: Dict[str, Any]) -> str:
        """Build command string from template and parameters"""
        # Add default values for missing parameters
        final_params = {}
        for param_name, param_config in tool.parameters.items():
            if param_name in parameters:
                final_params[param_name] = parameters[param_name]
            elif "default" in param_config:
                final_params[param_name] = param_config["default"]
        
        # Format command template
        try:
            command = tool.command_template.format(**final_params)
            return command
        except KeyError as e:
            raise ValueError(f"Missing parameter for command template: {e}")
    
    def _security_check(self, tool: ToolDefinition, command: str) -> bool:
        """Perform security check on command"""
        # Get allowed commands for security level
        allowed = self.allowed_commands.get(tool.security_level, [])
        
        # Extract base command
        base_command = command.split()[0] if command.split() else ""
        
        # Check if base command is allowed
        if tool.security_level != "high" and base_command not in allowed:
            logger.warning("Security check failed", command=base_command, 
                          security_level=tool.security_level)
            return False
        
        # Additional security checks
        dangerous_patterns = [";", "&&", "||", "|", ">", ">>", "<", "$(", "`"]
        if any(pattern in command for pattern in dangerous_patterns):
            logger.warning("Dangerous pattern detected in command", command=command)
            return False
        
        return True
    
    async def _execute_sync(self, tool: ToolDefinition, command: str) -> ExecutionResult:
        """Execute command synchronously"""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(tool.environment_vars)
            
            # Set working directory
            cwd = tool.working_directory or self.workspace_dir
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=tool.timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    return_code=process.returncode,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    execution_time=0.0,  # Will be set by caller
                    tool_name="",  # Will be set by caller
                    command_executed=""  # Will be set by caller
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return ExecutionResult(
                    success=False,
                    return_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {tool.timeout} seconds",
                    execution_time=0.0,
                    tool_name="",
                    command_executed="",
                    error_message="Timeout"
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                tool_name="",
                command_executed="",
                error_message=str(e)
            )
    
    async def _execute_async(self, tool: ToolDefinition, command: str) -> ExecutionResult:
        """Execute command asynchronously (same as sync for now)"""
        return await self._execute_sync(tool, command)
    
    def create_custom_tool(self, name: str, command_template: str, 
                          description: str, parameters: Dict[str, Any],
                          security_level: str = "medium") -> bool:
        """Create a custom CLI tool"""
        try:
            tool = ToolDefinition(
                name=name,
                tool_type=ToolType.CUSTOM,
                command_template=command_template,
                description=description,
                parameters=parameters,
                security_level=security_level
            )
            
            self.register_tool(tool)
            logger.info("Custom tool created", tool_name=name)
            return True
            
        except Exception as e:
            logger.error("Failed to create custom tool", tool_name=name, error=str(e))
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CLI executor statistics"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.successful_executions / max(self.total_executions, 1),
            "active_executions": self.active_executions,
            "max_concurrent_executions": self.max_concurrent_executions,
            "registered_tools": len(self.tools),
            "available_tools": list(self.tools.keys())
        }

# Global CLI executor instance
_cli_executor: Optional[CLIExecutor] = None

def get_cli_executor(workspace_dir: str = "/workspace") -> CLIExecutor:
    """Get or create the global CLI executor instance"""
    global _cli_executor
    if _cli_executor is None:
        _cli_executor = CLIExecutor(workspace_dir)
    return _cli_executor
