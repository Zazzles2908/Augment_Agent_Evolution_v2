"""
Log Retention Manager - Log Rotation and Cleanup
Manages log file rotation, archival, and cleanup policies

This module provides comprehensive log retention management including
automatic rotation, compression, archival, and cleanup based on policies.

Created: 2025-07-29 AEST
Purpose: Log rotation and cleanup management for Four-Brain system
Module Size: 150 lines (modular design)
"""

import os
import gzip
import shutil
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import glob
import stat

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Log retention policy configuration"""
    policy_id: str
    max_file_size_mb: int
    max_files_per_brain: int
    max_age_days: int
    compress_after_days: int
    archive_after_days: int
    delete_after_days: int
    enabled: bool


@dataclass
class RetentionStats:
    """Retention operation statistics"""
    files_rotated: int
    files_compressed: int
    files_archived: int
    files_deleted: int
    space_freed_mb: float
    operation_duration: float
    last_run: float


class LogRetentionManager:
    """
    Log Retention Manager
    
    Provides comprehensive log retention management including automatic
    rotation, compression, archival, and cleanup based on configurable policies.
    """
    
    def __init__(self, retention_manager_id: str = "log_retention_manager"):
        """Initialize log retention manager"""
        self.manager_id = retention_manager_id
        self.enabled = True
        
        # Retention configuration
        self.log_directory = Path("logs")
        self.archive_directory = Path("logs/archive")
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        
        # Retention policies
        self.policies: Dict[str, RetentionPolicy] = {}
        self.default_policy = RetentionPolicy(
            policy_id="default",
            max_file_size_mb=100,
            max_files_per_brain=10,
            max_age_days=30,
            compress_after_days=7,
            archive_after_days=14,
            delete_after_days=30,
            enabled=True
        )
        
        # Statistics
        self.stats = RetentionStats(
            files_rotated=0,
            files_compressed=0,
            files_archived=0,
            files_deleted=0,
            space_freed_mb=0.0,
            operation_duration=0.0,
            last_run=0.0
        )
        
        # Automatic cleanup
        self.auto_cleanup_enabled = True
        self.cleanup_interval = 3600  # 1 hour
        self.cleanup_thread = None
        self.cleanup_running = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"🗂️ Log Retention Manager initialized: {retention_manager_id}")
    
    def add_retention_policy(self, brain_id: str, policy: RetentionPolicy):
        """Add retention policy for a specific brain"""
        with self._lock:
            self.policies[brain_id] = policy
        
        logger.info(f"📋 Retention policy added for {brain_id}: {policy.policy_id}")
    
    def get_retention_policy(self, brain_id: str) -> RetentionPolicy:
        """Get retention policy for a brain"""
        with self._lock:
            return self.policies.get(brain_id, self.default_policy)
    
    def start_auto_cleanup(self):
        """Start automatic cleanup thread"""
        if self.cleanup_running:
            logger.warning("⚠️ Auto cleanup already running")
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("🔄 Auto cleanup started")
    
    def stop_auto_cleanup(self):
        """Stop automatic cleanup thread"""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=10)
        
        logger.info("🛑 Auto cleanup stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop"""
        while self.cleanup_running:
            try:
                self.run_retention_cleanup()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"❌ Auto cleanup error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def run_retention_cleanup(self) -> RetentionStats:
        """Run comprehensive retention cleanup"""
        cleanup_start = time.time()
        
        operation_stats = RetentionStats(
            files_rotated=0,
            files_compressed=0,
            files_archived=0,
            files_deleted=0,
            space_freed_mb=0.0,
            operation_duration=0.0,
            last_run=cleanup_start
        )
        
        try:
            # Get all log files
            log_files = self._discover_log_files()
            
            # Group by brain
            files_by_brain = self._group_files_by_brain(log_files)
            
            # Process each brain's files
            for brain_id, files in files_by_brain.items():
                policy = self.get_retention_policy(brain_id)
                
                if not policy.enabled:
                    continue
                
                brain_stats = self._process_brain_files(brain_id, files, policy)
                
                # Aggregate statistics
                operation_stats.files_rotated += brain_stats.files_rotated
                operation_stats.files_compressed += brain_stats.files_compressed
                operation_stats.files_archived += brain_stats.files_archived
                operation_stats.files_deleted += brain_stats.files_deleted
                operation_stats.space_freed_mb += brain_stats.space_freed_mb
            
            operation_stats.operation_duration = time.time() - cleanup_start
            
            # Update global statistics
            with self._lock:
                self.stats.files_rotated += operation_stats.files_rotated
                self.stats.files_compressed += operation_stats.files_compressed
                self.stats.files_archived += operation_stats.files_archived
                self.stats.files_deleted += operation_stats.files_deleted
                self.stats.space_freed_mb += operation_stats.space_freed_mb
                self.stats.last_run = cleanup_start
            
            logger.info(
                f"🧹 Retention cleanup completed: "
                f"rotated={operation_stats.files_rotated}, "
                f"compressed={operation_stats.files_compressed}, "
                f"archived={operation_stats.files_archived}, "
                f"deleted={operation_stats.files_deleted}, "
                f"freed={operation_stats.space_freed_mb:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"❌ Retention cleanup failed: {e}")
            operation_stats.operation_duration = time.time() - cleanup_start
        
        return operation_stats
    
    def _discover_log_files(self) -> List[Path]:
        """Discover all log files"""
        log_files = []
        
        # Find all .log files
        for pattern in ["*.log", "*.log.*"]:
            log_files.extend(self.log_directory.glob(pattern))
        
        # Find compressed files
        for pattern in ["*.log.gz", "*.log.*.gz"]:
            log_files.extend(self.log_directory.glob(pattern))
        
        return log_files
    
    def _group_files_by_brain(self, log_files: List[Path]) -> Dict[str, List[Path]]:
        """Group log files by brain ID"""
        files_by_brain = {}
        
        for file_path in log_files:
            # Extract brain ID from filename
            brain_id = self._extract_brain_id_from_filename(file_path.name)
            
            if brain_id not in files_by_brain:
                files_by_brain[brain_id] = []
            
            files_by_brain[brain_id].append(file_path)
        
        return files_by_brain
    
    def _extract_brain_id_from_filename(self, filename: str) -> str:
        """Extract brain ID from log filename"""
        # Common patterns: brain1_all.log, brain2_errors.log.1, etc.
        for brain_id in ['brain1', 'brain2', 'brain3', 'brain4', 'k2-hub', 'system']:
            if filename.startswith(brain_id):
                return brain_id
        
        return "unknown"
    
    def _process_brain_files(self, brain_id: str, files: List[Path], 
                           policy: RetentionPolicy) -> RetentionStats:
        """Process files for a specific brain"""
        brain_stats = RetentionStats(
            files_rotated=0,
            files_compressed=0,
            files_archived=0,
            files_deleted=0,
            space_freed_mb=0.0,
            operation_duration=0.0,
            last_run=time.time()
        )
        
        current_time = time.time()
        
        # Sort files by modification time (newest first)
        files_with_stats = []
        for file_path in files:
            try:
                file_stat = file_path.stat()
                files_with_stats.append((file_path, file_stat))
            except OSError:
                continue  # File might have been deleted
        
        files_with_stats.sort(key=lambda x: x[1].st_mtime, reverse=True)
        
        # Process files based on policy
        for i, (file_path, file_stat) in enumerate(files_with_stats):
            file_age_days = (current_time - file_stat.st_mtime) / 86400
            file_size_mb = file_stat.st_size / (1024 * 1024)
            
            # Delete old files
            if file_age_days > policy.delete_after_days:
                if self._delete_file(file_path):
                    brain_stats.files_deleted += 1
                    brain_stats.space_freed_mb += file_size_mb
                continue
            
            # Archive old files
            if file_age_days > policy.archive_after_days:
                if self._archive_file(file_path):
                    brain_stats.files_archived += 1
                continue
            
            # Compress old files
            if file_age_days > policy.compress_after_days and not file_path.name.endswith('.gz'):
                if self._compress_file(file_path):
                    brain_stats.files_compressed += 1
                    brain_stats.space_freed_mb += file_size_mb * 0.7  # Estimate compression savings
                continue
            
            # Rotate large files
            if file_size_mb > policy.max_file_size_mb and not file_path.name.endswith('.gz'):
                if self._rotate_file(file_path):
                    brain_stats.files_rotated += 1
                continue
            
            # Remove excess files (keep only max_files_per_brain)
            if i >= policy.max_files_per_brain:
                if self._delete_file(file_path):
                    brain_stats.files_deleted += 1
                    brain_stats.space_freed_mb += file_size_mb
        
        return brain_stats
    
    def _rotate_file(self, file_path: Path) -> bool:
        """Rotate a log file"""
        try:
            # Find next rotation number
            rotation_num = 1
            while True:
                rotated_path = file_path.with_suffix(f"{file_path.suffix}.{rotation_num}")
                if not rotated_path.exists():
                    break
                rotation_num += 1
            
            # Move file
            shutil.move(str(file_path), str(rotated_path))
            
            logger.debug(f"🔄 Rotated: {file_path.name} -> {rotated_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rotate {file_path}: {e}")
            return False
    
    def _compress_file(self, file_path: Path) -> bool:
        """Compress a log file"""
        try:
            compressed_path = file_path.with_suffix(f"{file_path.suffix}.gz")
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            file_path.unlink()
            
            logger.debug(f"🗜️ Compressed: {file_path.name} -> {compressed_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to compress {file_path}: {e}")
            return False
    
    def _archive_file(self, file_path: Path) -> bool:
        """Archive a log file"""
        try:
            # Create archive subdirectory by date
            archive_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m")
            archive_subdir = self.archive_directory / archive_date
            archive_subdir.mkdir(exist_ok=True)
            
            # Move file to archive
            archive_path = archive_subdir / file_path.name
            shutil.move(str(file_path), str(archive_path))
            
            logger.debug(f"📦 Archived: {file_path.name} -> {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to archive {file_path}: {e}")
            return False
    
    def _delete_file(self, file_path: Path) -> bool:
        """Delete a log file"""
        try:
            file_path.unlink()
            logger.debug(f"🗑️ Deleted: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete {file_path}: {e}")
            return False
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics"""
        try:
            # Log directory usage
            log_usage = self._get_directory_size(self.log_directory)
            
            # Archive directory usage
            archive_usage = self._get_directory_size(self.archive_directory)
            
            # Available space
            statvfs = os.statvfs(self.log_directory)
            available_space = statvfs.f_bavail * statvfs.f_frsize / (1024 * 1024)  # MB
            
            return {
                "log_directory_mb": log_usage,
                "archive_directory_mb": archive_usage,
                "total_usage_mb": log_usage + archive_usage,
                "available_space_mb": available_space,
                "log_directory_path": str(self.log_directory),
                "archive_directory_path": str(self.archive_directory)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get disk usage: {e}")
            return {"error": str(e)}
    
    def _get_directory_size(self, directory: Path) -> float:
        """Get total size of directory in MB"""
        total_size = 0
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"❌ Error calculating directory size for {directory}: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_retention_summary(self) -> Dict[str, Any]:
        """Get retention manager summary"""
        with self._lock:
            disk_usage = self.get_disk_usage()
            
            return {
                "manager_id": self.manager_id,
                "enabled": self.enabled,
                "auto_cleanup_enabled": self.auto_cleanup_enabled,
                "cleanup_running": self.cleanup_running,
                "cleanup_interval_seconds": self.cleanup_interval,
                "policies_count": len(self.policies),
                "statistics": asdict(self.stats),
                "disk_usage": disk_usage,
                "log_directory": str(self.log_directory),
                "archive_directory": str(self.archive_directory)
            }


# Factory function for easy creation
def create_log_retention_manager(manager_id: str = "log_retention_manager") -> LogRetentionManager:
    """Factory function to create log retention manager"""
    return LogRetentionManager(manager_id)
