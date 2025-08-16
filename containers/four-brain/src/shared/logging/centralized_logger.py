"""
Centralized Logger - Unified Logging System
Provides centralized logging configuration and management for all Four-Brain components

This module creates a unified logging system that replaces scattered logging
configurations with a centralized, consistent, and comprehensive logging solution.

Created: 2025-07-29 AEST
Purpose: Centralized logging for all Four-Brain components
Module Size: 150 lines (modular design)
"""

import logging
import logging.handlers
import os
import sys
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import threading
from enum import Enum

# Custom log levels
TRACE_LEVEL = 5
SUCCESS_LEVEL = 25

# Add custom levels to logging
logging.addLevelName(TRACE_LEVEL, "TRACE")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class LogLevel(Enum):
    """Standardized log levels"""
    TRACE = TRACE_LEVEL
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    SUCCESS = SUCCESS_LEVEL
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    """Standardized log formats"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


class CentralizedLogger:
    """
    Centralized Logger System
    
    Provides unified logging configuration, formatting, and management
    for all Four-Brain components with consistent output and aggregation.
    """
    
    def __init__(self, brain_id: str, log_level: LogLevel = LogLevel.INFO):
        """Initialize centralized logger"""
        self.brain_id = brain_id
        self.log_level = log_level
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Logging configuration
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Log formats
        self.formats = {
            LogFormat.SIMPLE: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            LogFormat.DETAILED: "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            LogFormat.JSON: None,  # Custom JSON formatter
            LogFormat.STRUCTURED: "%(asctime)s | %(brain_id)s | %(name)s | %(levelname)s | %(message)s"
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize root logger
        self._setup_root_logger()
        
        print(f"🔍 Centralized Logger initialized for {brain_id}")
    
    def _setup_root_logger(self):
        """Setup root logger configuration"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level.value)
        console_formatter = self._create_formatter(LogFormat.STRUCTURED)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        all_logs_file = self.log_dir / f"{self.brain_id}_all.log"
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_formatter = self._create_formatter(LogFormat.DETAILED)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_logs_file = self.log_dir / f"{self.brain_id}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_logs_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = self._create_formatter(LogFormat.DETAILED)
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
        
        # JSON handler for structured logging
        json_logs_file = self.log_dir / f"{self.brain_id}_structured.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_logs_file, maxBytes=10*1024*1024, backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = self._create_json_formatter()
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
    
    def _create_formatter(self, format_type: LogFormat) -> logging.Formatter:
        """Create formatter based on format type"""
        format_string = self.formats[format_type]
        brain_id = self.brain_id  # Capture brain_id in closure

        class BrainFormatter(logging.Formatter):
            def format(self, record):
                # Add brain_id to record
                record.brain_id = brain_id
                return super().format(record)

        return BrainFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging"""
        brain_id = self.brain_id  # Capture brain_id in closure

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "brain_id": brain_id,
                    "logger_name": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "thread": record.thread,
                    "process": record.process
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                                  'filename', 'module', 'lineno', 'funcName', 'created', 
                                  'msecs', 'relativeCreated', 'thread', 'threadName', 
                                  'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                        log_entry[f"extra_{key}"] = value
                
                return json.dumps(log_entry)
        
        return JSONFormatter()
    
    def get_logger(self, name: str, component: str = None) -> logging.Logger:
        """Get or create a logger for a specific component"""
        logger_name = f"{self.brain_id}.{name}"
        if component:
            logger_name += f".{component}"
        
        with self._lock:
            if logger_name not in self.loggers:
                logger = logging.getLogger(logger_name)
                
                # Add custom methods
                def trace(message, *args, **kwargs):
                    logger.log(TRACE_LEVEL, message, *args, **kwargs)
                
                def success(message, *args, **kwargs):
                    logger.log(SUCCESS_LEVEL, message, *args, **kwargs)
                
                logger.trace = trace
                logger.success = success
                
                self.loggers[logger_name] = logger
            
            return self.loggers[logger_name]
    
    def log_operation_start(self, operation_name: str, details: Dict[str, Any] = None) -> str:
        """Log operation start with unique ID"""
        operation_id = f"op_{int(time.time() * 1000000)}"
        logger = self.get_logger("operations")
        
        logger.info(
            f"🚀 Operation started: {operation_name}",
            extra={
                "operation_id": operation_id,
                "operation_name": operation_name,
                "operation_status": "started",
                "details": details or {}
            }
        )
        
        return operation_id
    
    def log_operation_end(self, operation_id: str, operation_name: str, 
                         duration: float, success: bool = True, 
                         error_message: str = None, result: Dict[str, Any] = None):
        """Log operation completion"""
        logger = self.get_logger("operations")
        
        if success:
            logger.success(
                f"✅ Operation completed: {operation_name} ({duration:.2f}s)",
                extra={
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "operation_status": "completed",
                    "duration_seconds": duration,
                    "success": True,
                    "result": result or {}
                }
            )
        else:
            logger.error(
                f"❌ Operation failed: {operation_name} ({duration:.2f}s) - {error_message}",
                extra={
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "operation_status": "failed",
                    "duration_seconds": duration,
                    "success": False,
                    "error_message": error_message,
                    "result": result or {}
                }
            )
    
    def log_inter_brain_communication(self, source_brain: str, target_brain: str, 
                                    message_type: str, message_id: str, 
                                    success: bool = True, error_message: str = None):
        """Log inter-brain communication"""
        logger = self.get_logger("communication")
        
        if success:
            logger.info(
                f"📤 Message sent: {source_brain} → {target_brain} ({message_type})",
                extra={
                    "message_id": message_id,
                    "source_brain": source_brain,
                    "target_brain": target_brain,
                    "message_type": message_type,
                    "communication_status": "sent",
                    "success": True
                }
            )
        else:
            logger.error(
                f"❌ Message failed: {source_brain} → {target_brain} ({message_type}) - {error_message}",
                extra={
                    "message_id": message_id,
                    "source_brain": source_brain,
                    "target_brain": target_brain,
                    "message_type": message_type,
                    "communication_status": "failed",
                    "success": False,
                    "error_message": error_message
                }
            )
    
    def log_database_operation(self, operation_type: str, table_name: str, 
                             duration: float, success: bool = True, 
                             error_message: str = None, affected_rows: int = None):
        """Log database operations"""
        logger = self.get_logger("database")
        
        if success:
            logger.info(
                f"🗄️ Database {operation_type}: {table_name} ({duration:.3f}s)",
                extra={
                    "operation_type": operation_type,
                    "table_name": table_name,
                    "duration_seconds": duration,
                    "success": True,
                    "affected_rows": affected_rows
                }
            )
        else:
            logger.error(
                f"❌ Database {operation_type} failed: {table_name} ({duration:.3f}s) - {error_message}",
                extra={
                    "operation_type": operation_type,
                    "table_name": table_name,
                    "duration_seconds": duration,
                    "success": False,
                    "error_message": error_message
                }
            )
    
    def log_system_event(self, event_type: str, message: str, severity: LogLevel = LogLevel.INFO, 
                        metadata: Dict[str, Any] = None):
        """Log system events"""
        logger = self.get_logger("system")
        
        logger.log(
            severity.value,
            f"🔧 System event: {message}",
            extra={
                "event_type": event_type,
                "severity": severity.name,
                "metadata": metadata or {}
            }
        )
    
    def set_log_level(self, level: LogLevel):
        """Set logging level for all loggers"""
        self.log_level = level
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level.value)
        
        # Update console handler
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level.value)
    
    def get_log_files(self) -> List[Path]:
        """Get list of all log files"""
        return list(self.log_dir.glob(f"{self.brain_id}_*.log*"))
    
    def get_logger_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "brain_id": self.brain_id,
            "log_level": self.log_level.name,
            "active_loggers": len(self.loggers),
            "logger_names": list(self.loggers.keys()),
            "log_directory": str(self.log_dir),
            "log_files": [str(f) for f in self.get_log_files()]
        }


# Global logger instance
_global_logger: Optional[CentralizedLogger] = None


def create_centralized_logger(brain_id: str, log_level: LogLevel = LogLevel.INFO) -> CentralizedLogger:
    """Factory function to create centralized logger"""
    return CentralizedLogger(brain_id, log_level)


def get_global_logger(brain_id: str = None) -> CentralizedLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        if brain_id is None:
            brain_id = "unknown_brain"
        _global_logger = CentralizedLogger(brain_id)
    
    return _global_logger


# Convenience functions for easy integration
def get_logger(name: str, component: str = None) -> logging.Logger:
    """Convenience function to get logger"""
    global_logger = get_global_logger()
    return global_logger.get_logger(name, component)
