"""
Log Aggregator - Centralized Log Collection and Processing
Collects and processes logs from all Four-Brain components

This module provides centralized log aggregation, processing, and forwarding
to create a unified view of all system activities across all brains.

Created: 2025-07-29 AEST
Purpose: Centralized log collection and processing for all Four-Brain components
Module Size: 150 lines (modular design)
"""

import asyncio
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import re
import gzip
import shutil

logger = logging.getLogger(__name__)


class LogEntry:
    """Structured log entry"""
    def __init__(self, timestamp: float, brain_id: str, level: str, 
                 logger_name: str, message: str, metadata: Dict[str, Any] = None):
        self.timestamp = timestamp
        self.brain_id = brain_id
        self.level = level
        self.logger_name = logger_name
        self.message = message
        self.metadata = metadata or {}
        self.entry_id = f"{brain_id}_{int(timestamp * 1000000)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "brain_id": self.brain_id,
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class LogAggregator:
    """
    Centralized Log Aggregator
    
    Collects logs from all Four-Brain components, processes them,
    and provides unified access to aggregated log data.
    """
    
    def __init__(self, aggregator_id: str = "log_aggregator"):
        """Initialize log aggregator"""
        self.aggregator_id = aggregator_id
        self.enabled = True
        
        # Log storage
        self.log_entries: deque = deque(maxlen=50000)  # Keep last 50k entries
        self.logs_by_brain: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.logs_by_level: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Processing configuration
        self.log_sources = {}  # brain_id -> log_file_paths
        self.processors = []  # List of log processors
        self.filters = []  # List of log filters
        
        # Aggregation statistics
        self.stats = {
            "total_entries": 0,
            "entries_by_brain": defaultdict(int),
            "entries_by_level": defaultdict(int),
            "start_time": time.time(),
            "last_processed": None
        }
        
        # File monitoring
        self.monitored_files = {}  # file_path -> last_position
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"📊 Log Aggregator initialized: {aggregator_id}")
    
    def add_log_source(self, brain_id: str, log_file_paths: List[Path]):
        """Add log source for a brain"""
        with self._lock:
            self.log_sources[brain_id] = log_file_paths
            
            # Initialize file monitoring
            for file_path in log_file_paths:
                if file_path.exists():
                    self.monitored_files[str(file_path)] = file_path.stat().st_size
                else:
                    self.monitored_files[str(file_path)] = 0
        
        logger.info(f"📁 Added log source for {brain_id}: {len(log_file_paths)} files")
    
    def add_processor(self, processor: Callable[[LogEntry], LogEntry]):
        """Add log processor function"""
        self.processors.append(processor)
        logger.info(f"⚙️ Added log processor: {processor.__name__}")
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """Add log filter function"""
        self.filters.append(filter_func)
        logger.info(f"🔍 Added log filter: {filter_func.__name__}")
    
    def start_monitoring(self):
        """Start monitoring log files for new entries"""
        if self.monitoring_active:
            logger.warning("⚠️ Log monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🔄 Log monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring log files"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Log monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_log_files()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"❌ Log monitoring error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _check_log_files(self):
        """Check log files for new content"""
        for file_path_str, last_position in list(self.monitored_files.items()):
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                continue
            
            current_size = file_path.stat().st_size
            
            if current_size > last_position:
                # File has new content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        
                        for line in new_lines:
                            self._process_log_line(line.strip(), file_path)
                        
                        self.monitored_files[file_path_str] = current_size
                        
                except Exception as e:
                    logger.error(f"❌ Error reading log file {file_path}: {e}")
            elif current_size < last_position:
                # File was rotated or truncated
                self.monitored_files[file_path_str] = 0
    
    def _process_log_line(self, line: str, file_path: Path):
        """Process a single log line"""
        if not line.strip():
            return
        
        try:
            # Try to parse as JSON first (structured logs)
            if line.startswith('{'):
                log_data = json.loads(line)
                entry = LogEntry(
                    timestamp=datetime.fromisoformat(log_data.get('timestamp', datetime.now().isoformat())).timestamp(),
                    brain_id=log_data.get('brain_id', 'unknown'),
                    level=log_data.get('level', 'INFO'),
                    logger_name=log_data.get('logger_name', 'unknown'),
                    message=log_data.get('message', ''),
                    metadata=log_data
                )
            else:
                # Parse standard log format
                entry = self._parse_standard_log_line(line, file_path)
            
            if entry:
                self._add_log_entry(entry)
                
        except Exception as e:
            logger.debug(f"Failed to parse log line: {line[:100]}... - {e}")
    
    def _parse_standard_log_line(self, line: str, file_path: Path) -> Optional[LogEntry]:
        """Parse standard log format"""
        # Pattern for standard log format: timestamp - logger - level - message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\s]*-\s*([^-]+)\s*-\s*(\w+)\s*-\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, logger_name, level, message = match.groups()
            
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                timestamp = time.time()
            
            # Extract brain_id from file path or logger name
            brain_id = self._extract_brain_id(file_path, logger_name)
            
            return LogEntry(
                timestamp=timestamp,
                brain_id=brain_id,
                level=level.strip(),
                logger_name=logger_name.strip(),
                message=message.strip(),
                metadata={"source_file": str(file_path)}
            )
        
        return None
    
    def _extract_brain_id(self, file_path: Path, logger_name: str) -> str:
        """Extract brain ID from file path or logger name"""
        # Try to extract from filename
        filename = file_path.stem
        for brain_id in ['brain1', 'brain2', 'brain3', 'brain4', 'k2-hub']:
            if brain_id in filename.lower():
                return brain_id
        
        # Try to extract from logger name
        if '.' in logger_name:
            parts = logger_name.split('.')
            if len(parts) > 0:
                return parts[0]
        
        return "unknown"
    
    def _add_log_entry(self, entry: LogEntry):
        """Add log entry to aggregated storage"""
        # Apply filters
        for filter_func in self.filters:
            if not filter_func(entry):
                return  # Entry filtered out
        
        # Apply processors
        for processor in self.processors:
            entry = processor(entry)
        
        with self._lock:
            # Add to main storage
            self.log_entries.append(entry)
            
            # Add to categorized storage
            self.logs_by_brain[entry.brain_id].append(entry)
            self.logs_by_level[entry.level].append(entry)
            
            # Update statistics
            self.stats["total_entries"] += 1
            self.stats["entries_by_brain"][entry.brain_id] += 1
            self.stats["entries_by_level"][entry.level] += 1
            self.stats["last_processed"] = time.time()
    
    def add_log_entry_direct(self, entry: LogEntry):
        """Add log entry directly (for programmatic logging)"""
        self._add_log_entry(entry)
    
    def get_recent_logs(self, limit: int = 100, brain_id: str = None, 
                       level: str = None, since: float = None) -> List[Dict[str, Any]]:
        """Get recent log entries with optional filtering"""
        with self._lock:
            entries = list(self.log_entries)
        
        # Apply filters
        if brain_id:
            entries = [e for e in entries if e.brain_id == brain_id]
        
        if level:
            entries = [e for e in entries if e.level == level]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        entries = entries[:limit]
        
        return [entry.to_dict() for entry in entries]
    
    def get_logs_by_pattern(self, pattern: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs matching a regex pattern"""
        regex = re.compile(pattern, re.IGNORECASE)
        
        with self._lock:
            matching_entries = [
                entry for entry in self.log_entries
                if regex.search(entry.message)
            ]
        
        # Sort by timestamp (newest first) and limit
        matching_entries.sort(key=lambda x: x.timestamp, reverse=True)
        matching_entries = matching_entries[:limit]
        
        return [entry.to_dict() for entry in matching_entries]
    
    def get_error_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of errors in time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self._lock:
            error_entries = [
                entry for entry in self.log_entries
                if entry.level in ['ERROR', 'CRITICAL'] and entry.timestamp >= cutoff_time
            ]
        
        # Group by brain and error type
        errors_by_brain = defaultdict(int)
        error_patterns = defaultdict(int)
        
        for entry in error_entries:
            errors_by_brain[entry.brain_id] += 1
            
            # Extract error pattern (first 50 chars)
            pattern = entry.message[:50] + "..." if len(entry.message) > 50 else entry.message
            error_patterns[pattern] += 1
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_errors": len(error_entries),
            "errors_by_brain": dict(errors_by_brain),
            "common_error_patterns": dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def export_logs(self, output_file: Path, format: str = "jsonl", 
                   since: float = None, brain_id: str = None):
        """Export logs to file"""
        entries = self.get_recent_logs(limit=None, brain_id=brain_id, since=since)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if format == "jsonl":
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')
            elif format == "json":
                json.dump(entries, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"📤 Exported {len(entries)} log entries to {output_file}")
    
    def get_aggregator_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        with self._lock:
            return {
                "aggregator_id": self.aggregator_id,
                "enabled": self.enabled,
                "monitoring_active": self.monitoring_active,
                "total_entries": self.stats["total_entries"],
                "entries_by_brain": dict(self.stats["entries_by_brain"]),
                "entries_by_level": dict(self.stats["entries_by_level"]),
                "uptime_seconds": time.time() - self.stats["start_time"],
                "last_processed": self.stats["last_processed"],
                "monitored_files": len(self.monitored_files),
                "processors": len(self.processors),
                "filters": len(self.filters)
            }


# Factory function for easy creation
def create_log_aggregator(aggregator_id: str = "log_aggregator") -> LogAggregator:
    """Factory function to create log aggregator"""
    return LogAggregator(aggregator_id)
