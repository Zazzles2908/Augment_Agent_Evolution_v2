#!/usr/bin/env python3
"""
Four-Brain System Performance Monitor
Optimized for RTX 5070 Ti Production Environment
Version: Production v1.0
"""

import os
import sys
import time
import json
import psutil
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.append('/app/src')

try:
    import torch
    import nvidia_ml_py3 as nvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    torch = None
    nvml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/performance_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int

@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    timestamp: float
    gpu_id: int
    name: str
    utilization_percent: float
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    clock_graphics_mhz: int
    clock_memory_mhz: int

@dataclass
class ProcessMetrics:
    """Process-specific metrics"""
    timestamp: float
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    num_threads: int
    num_fds: int
    status: str

class PerformanceMonitor:
    """Production performance monitoring system"""
    
    def __init__(self):
        self.brain_role = os.getenv('BRAIN_ROLE', 'unknown')
        self.monitoring_interval = int(os.getenv('MONITORING_INTERVAL', '30'))
        self.metrics_file = f'/app/logs/metrics_{self.brain_role}.json'
        self.running = False
        
        # Initialize NVIDIA ML if available
        if NVIDIA_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                logger.info(f"Initialized NVIDIA ML with {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA ML: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
            logger.warning("NVIDIA ML not available")
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'gpu_warning': 90.0,
            'gpu_critical': 98.0,
            'temperature_warning': 80,
            'temperature_critical': 90
        }
        
        logger.info(f"Performance monitor initialized for {self.brain_role}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Load average
            load_avg = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                load_average=load_avg,
                process_count=process_count
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect GPU performance metrics"""
        if not NVIDIA_AVAILABLE or self.gpu_count == 0:
            return []
        
        gpu_metrics = []
        try:
            for i in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                # Power
                try:
                    power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_draw = 0.0
                    power_limit = 0.0
                
                # Clock speeds
                try:
                    clock_graphics = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                    clock_memory = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                except:
                    clock_graphics = 0
                    clock_memory = 0
                
                gpu_metrics.append(GPUMetrics(
                    timestamp=time.time(),
                    gpu_id=i,
                    name=name,
                    utilization_percent=util.gpu,
                    memory_used_mb=mem_info.used // (1024**2),
                    memory_total_mb=mem_info.total // (1024**2),
                    memory_percent=(mem_info.used / mem_info.total) * 100,
                    temperature_c=temp,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    clock_graphics_mhz=clock_graphics,
                    clock_memory_mhz=clock_memory
                ))
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return gpu_metrics
    
    def get_process_metrics(self) -> List[ProcessMetrics]:
        """Collect process-specific metrics"""
        process_metrics = []
        try:
            current_pid = os.getpid()
            
            # Get current process and children
            current_process = psutil.Process(current_pid)
            processes = [current_process] + current_process.children(recursive=True)
            
            for proc in processes:
                try:
                    # Get process info
                    proc_info = proc.as_dict([
                        'pid', 'name', 'cpu_percent', 'memory_percent',
                        'memory_info', 'num_threads', 'status'
                    ])
                    
                    # Get file descriptors count (Unix only)
                    try:
                        num_fds = proc.num_fds()
                    except:
                        num_fds = 0
                    
                    process_metrics.append(ProcessMetrics(
                        timestamp=time.time(),
                        pid=proc_info['pid'],
                        name=proc_info['name'],
                        cpu_percent=proc_info['cpu_percent'] or 0.0,
                        memory_percent=proc_info['memory_percent'] or 0.0,
                        memory_rss_mb=(proc_info['memory_info'].rss / (1024**2)) if proc_info['memory_info'] else 0.0,
                        memory_vms_mb=(proc_info['memory_info'].vms / (1024**2)) if proc_info['memory_info'] else 0.0,
                        num_threads=proc_info['num_threads'] or 0,
                        num_fds=num_fds,
                        status=proc_info['status'] or 'unknown'
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
        
        return process_metrics
    
    def check_thresholds(self, system_metrics: SystemMetrics, gpu_metrics: List[GPUMetrics]):
        """Check performance thresholds and log warnings"""
        # System thresholds
        if system_metrics:
            if system_metrics.cpu_percent > self.thresholds['cpu_critical']:
                logger.critical(f"CPU usage critical: {system_metrics.cpu_percent:.1f}%")
            elif system_metrics.cpu_percent > self.thresholds['cpu_warning']:
                logger.warning(f"CPU usage high: {system_metrics.cpu_percent:.1f}%")
            
            if system_metrics.memory_percent > self.thresholds['memory_critical']:
                logger.critical(f"Memory usage critical: {system_metrics.memory_percent:.1f}%")
            elif system_metrics.memory_percent > self.thresholds['memory_warning']:
                logger.warning(f"Memory usage high: {system_metrics.memory_percent:.1f}%")
        
        # GPU thresholds
        for gpu in gpu_metrics:
            if gpu.utilization_percent > self.thresholds['gpu_critical']:
                logger.critical(f"GPU {gpu.gpu_id} utilization critical: {gpu.utilization_percent}%")
            elif gpu.utilization_percent > self.thresholds['gpu_warning']:
                logger.warning(f"GPU {gpu.gpu_id} utilization high: {gpu.utilization_percent}%")
            
            if gpu.temperature_c > self.thresholds['temperature_critical']:
                logger.critical(f"GPU {gpu.gpu_id} temperature critical: {gpu.temperature_c}°C")
            elif gpu.temperature_c > self.thresholds['temperature_warning']:
                logger.warning(f"GPU {gpu.gpu_id} temperature high: {gpu.temperature_c}°C")
    
    def save_metrics(self, system_metrics: SystemMetrics, gpu_metrics: List[GPUMetrics], 
                    process_metrics: List[ProcessMetrics]):
        """Save metrics to file"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'brain_role': self.brain_role,
                'system': asdict(system_metrics) if system_metrics else None,
                'gpu': [asdict(gpu) for gpu in gpu_metrics],
                'processes': [asdict(proc) for proc in process_metrics]
            }
            
            # Append to metrics file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_data) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info(f"Starting performance monitoring loop (interval: {self.monitoring_interval}s)")
        
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                gpu_metrics = self.get_gpu_metrics()
                process_metrics = self.get_process_metrics()
                
                # Check thresholds
                self.check_thresholds(system_metrics, gpu_metrics)
                
                # Save metrics
                self.save_metrics(system_metrics, gpu_metrics, process_metrics)
                
                # Log summary
                if system_metrics:
                    logger.info(f"System: CPU {system_metrics.cpu_percent:.1f}%, "
                              f"Memory {system_metrics.memory_percent:.1f}%, "
                              f"GPU {len(gpu_metrics)} devices")
                
                # Wait for next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error
    
    def start(self):
        """Start performance monitoring"""
        self.running = True
        monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Performance monitoring started")
        return monitor_thread
    
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        logger.info("Performance monitoring stopped")

def main():
    """Main function"""
    monitor = PerformanceMonitor()
    
    try:
        # Start monitoring
        monitor_thread = monitor.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
