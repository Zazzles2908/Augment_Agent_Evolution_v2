#!/usr/bin/env python3
"""
Enhanced Resource Monitor for Three-Brain Architecture
Tracks GPU memory, system memory, and performance metrics with optimization support
"""

import os
import time
import psutil
import torch
import logging
from datetime import datetime
from threading import Thread, Event

# Get logger for this module
logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.stop_event = Event()
        self.monitor_thread = None
        self.stats_history = []

        # Check for GPU monitoring packages
        self.nvidia_ml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvidia_ml_available = True
            logger.info("📊 NVIDIA ML monitoring available")
        except ImportError:
            logger.warning("⚠️ NVIDIA ML monitoring not available")

        logger.info("📊 ResourceMonitor initialized")

    def start_monitoring(self, interval=30):
        """Start continuous resource monitoring."""
        if self.monitoring:
            logger.warning("⚠️ Monitoring already active")
            return

        self.monitoring = True
        self.stop_event.clear()

        self.monitor_thread = Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

        logger.info(f"📊 Resource monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("📊 Resource monitoring stopped")

    def _monitor_loop(self, interval):
        """Main monitoring loop."""
        while not self.stop_event.wait(interval):
            try:
                stats = self.get_current_stats()
                self.stats_history.append(stats)

                # Keep only last 100 entries
                if len(self.stats_history) > 100:
                    self.stats_history.pop(0)

                # Log critical memory usage
                if torch.cuda.is_available():
                    gpu_usage_percent = (stats['gpu_memory_allocated'] / stats['gpu_memory_total']) * 100
                    if gpu_usage_percent > 90:
                        logger.warning(f"⚠️ HIGH GPU MEMORY USAGE: {gpu_usage_percent:.1f}%")

                # Log high system memory usage
                if stats['system_memory_percent'] > 85:
                    logger.warning(f"⚠️ HIGH SYSTEM MEMORY USAGE: {stats['system_memory_percent']:.1f}%")

            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")

    def get_current_stats(self):
        """Get current system resource statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_allocated': 0,
            'gpu_memory_cached': 0,
            'gpu_memory_total': 0,
            'gpu_utilization': 0,
            'system_memory_used': 0,
            'system_memory_total': 0,
            'system_memory_percent': 0,
            'cpu_percent': 0,
            'optimization_status': self._get_optimization_status()
        }

        # GPU statistics
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved()
            stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory

            # Enhanced GPU stats with NVIDIA ML if available
            if self.nvidia_ml_available:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats['gpu_utilization'] = util.gpu
                except:
                    pass

        # System statistics
        memory = psutil.virtual_memory()
        stats['system_memory_used'] = memory.used
        stats['system_memory_total'] = memory.total
        stats['system_memory_percent'] = memory.percent
        stats['cpu_percent'] = psutil.cpu_percent()

        return stats

    def _get_optimization_status(self):
        """Get status of optimization packages."""
        status = {
            'unsloth': False,
            'flash_attn': False,
            'bitsandbytes': False
        }

        try:
            import unsloth
            status['unsloth'] = True
        except ImportError:
            pass

        try:
            import flash_attn
            status['flash_attn'] = True
        except ImportError:
            pass

        try:
            import bitsandbytes
            status['bitsandbytes'] = True
        except ImportError:
            pass

        return status

    def log_memory_usage(self, context=""):
        """Log current memory usage with context."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            utilization = (allocated / total) * 100

            logger.info(f"💾 GPU Memory {context}: {allocated:.1f}GB/{total:.1f}GB ({utilization:.1f}%)")

        # System memory
        memory = psutil.virtual_memory()
        system_gb = memory.used / 1e9
        system_total_gb = memory.total / 1e9

        logger.info(f"🖥️ System Memory {context}: {system_gb:.1f}GB/{system_total_gb:.1f}GB ({memory.percent:.1f}%)")

    def get_memory_summary(self):
        """Get a summary of current memory usage."""
        summary = {}

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            summary['gpu'] = {
                'allocated_gb': allocated,
                'total_gb': total,
                'utilization_percent': (allocated / total) * 100,
                'available_gb': total - allocated
            }

        memory = psutil.virtual_memory()
        summary['system'] = {
            'used_gb': memory.used / 1e9,
            'total_gb': memory.total / 1e9,
            'percent': memory.percent,
            'available_gb': memory.available / 1e9
        }

        return summary

    def check_memory_health(self):
        """Check if memory usage is within healthy limits."""
        health = {
            'gpu_healthy': True,
            'system_healthy': True,
            'warnings': []
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_percent = (allocated / total) * 100

            if gpu_percent > 95:
                health['gpu_healthy'] = False
                health['warnings'].append(f"Critical GPU memory usage: {gpu_percent:.1f}%")
            elif gpu_percent > 85:
                health['warnings'].append(f"High GPU memory usage: {gpu_percent:.1f}%")

        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health['system_healthy'] = False
            health['warnings'].append(f"Critical system memory usage: {memory.percent:.1f}%")
        elif memory.percent > 80:
            health['warnings'].append(f"High system memory usage: {memory.percent:.1f}%")

        return health

    def get_performance_metrics(self):
        """Get performance metrics for the three-brain system."""
        if not self.stats_history:
            return None

        recent_stats = self.stats_history[-10:]  # Last 10 measurements

        metrics = {
            'avg_gpu_utilization': 0,
            'avg_system_memory': 0,
            'avg_cpu_usage': 0,
            'gpu_memory_trend': 'stable',
            'optimization_active': self._get_optimization_status()
        }

        if recent_stats:
            # Calculate averages
            gpu_utils = [s.get('gpu_utilization', 0) for s in recent_stats]
            sys_mems = [s.get('system_memory_percent', 0) for s in recent_stats]
            cpu_uses = [s.get('cpu_percent', 0) for s in recent_stats]

            metrics['avg_gpu_utilization'] = sum(gpu_utils) / len(gpu_utils)
            metrics['avg_system_memory'] = sum(sys_mems) / len(sys_mems)
            metrics['avg_cpu_usage'] = sum(cpu_uses) / len(cpu_uses)

            # Determine GPU memory trend
            if len(recent_stats) >= 3:
                gpu_mems = [s.get('gpu_memory_allocated', 0) for s in recent_stats[-3:]]
                if gpu_mems[-1] > gpu_mems[0] * 1.1:
                    metrics['gpu_memory_trend'] = 'increasing'
                elif gpu_mems[-1] < gpu_mems[0] * 0.9:
                    metrics['gpu_memory_trend'] = 'decreasing'

        return metrics

    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()

        if self.nvidia_ml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass
