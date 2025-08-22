#!/usr/bin/env python3
"""
Four-Brain System Real-Time Dashboard
Comprehensive monitoring dashboard for Four-Brain architecture

Created: 2025-07-27 AEST
Author: AugmentAI - Monitoring Implementation
"""

import asyncio
import aiohttp
import redis
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BrainMetrics:
    name: str
    status: str
    response_time_ms: float
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    requests_per_minute: int
    error_rate: float
    last_activity: str

@dataclass
class SystemMetrics:
    timestamp: float
    overall_health: str
    total_containers: int
    healthy_containers: int
    redis_memory_mb: float
    redis_messages: int
    ai_communication_active: bool
    brains: Dict[str, BrainMetrics]

class FourBrainMonitor:
    """Real-time monitoring for Four-Brain system"""
    
    def __init__(self):
        self.brain_endpoints = {
            "brain1_embedding": "http://localhost:8001",
            "brain2_reranker": "http://localhost:8002", 
            "brain3_augment": "http://localhost:8003",
            "brain4_docling": "http://localhost:8004",
            "k2_vector_hub": "http://localhost:9098"
        }
        
        self.redis_client = redis.Redis(
            host="localhost", 
            port=6379, 
            decode_responses=True
        )
        
        self.streams = [
            "embedding_requests", "embedding_results",
            "rerank_requests", "rerank_results", 
            "docling_requests", "docling_results",
            "agentic_tasks", "agentic_results",
            "memory_updates"
        ]
        
        self.metrics_history = []
        self.connected_clients = set()

    async def get_brain_metrics(self, name: str, url: str) -> BrainMetrics:
        """Get metrics for a specific brain"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Health check
                async with session.get(f"{url}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = "healthy"
                        try:
                            health_data = await response.json()
                        except:
                            health_data = {}
                    else:
                        status = "degraded"
                        health_data = {}
                
                # Try to get metrics
                try:
                    async with session.get(f"{url}/metrics") as metrics_response:
                        if metrics_response.status == 200:
                            metrics_text = await metrics_response.text()
                            # Parse basic metrics from Prometheus format
                            cpu_usage = 0.0
                            memory_usage = 0.0
                            gpu_usage = 0.0
                            
                            for line in metrics_text.split('\n'):
                                if 'cpu_usage' in line and not line.startswith('#'):
                                    try:
                                        cpu_usage = float(line.split()[-1])
                                    except:
                                        pass
                                elif 'memory_usage' in line and not line.startswith('#'):
                                    try:
                                        memory_usage = float(line.split()[-1]) / (1024*1024)  # Convert to MB
                                    except:
                                        pass
                                elif 'gpu_usage' in line and not line.startswith('#'):
                                    try:
                                        gpu_usage = float(line.split()[-1])
                                    except:
                                        pass
                        else:
                            cpu_usage = memory_usage = gpu_usage = 0.0
                except:
                    cpu_usage = memory_usage = gpu_usage = 0.0
                
                return BrainMetrics(
                    name=name,
                    status=status,
                    response_time_ms=response_time,
                    cpu_usage=cpu_usage,
                    memory_usage_mb=memory_usage,
                    gpu_usage=gpu_usage,
                    requests_per_minute=0,  # Would need historical data
                    error_rate=0.0,  # Would need error tracking
                    last_activity=datetime.now().strftime("%H:%M:%S")
                )
                
        except Exception as e:
            logger.error(f"Error getting metrics for {name}: {e}")
            return BrainMetrics(
                name=name,
                status="unhealthy",
                response_time_ms=0.0,
                cpu_usage=0.0,
                memory_usage_mb=0.0,
                gpu_usage=0.0,
                requests_per_minute=0,
                error_rate=100.0,
                last_activity="N/A"
            )

    def get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis metrics"""
        try:
            info = self.redis_client.info()
            
            # Get stream message counts
            total_messages = 0
            stream_status = {}
            
            for stream in self.streams:
                try:
                    length = self.redis_client.xlen(stream)
                    stream_status[stream] = length
                    total_messages += length
                except:
                    stream_status[stream] = 0
            
            return {
                "memory_mb": info.get('used_memory', 0) / (1024 * 1024),
                "total_messages": total_messages,
                "streams": stream_status,
                "connected_clients": info.get('connected_clients', 0),
                "hit_rate": info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis metrics: {e}")
            return {
                "memory_mb": 0,
                "total_messages": 0,
                "streams": {},
                "connected_clients": 0,
                "hit_rate": 0
            }

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        logger.info("📊 Collecting system metrics...")
        
        # Get brain metrics
        brain_tasks = [
            self.get_brain_metrics(name, url) 
            for name, url in self.brain_endpoints.items()
        ]
        brain_results = await asyncio.gather(*brain_tasks)
        
        brains = {brain.name: brain for brain in brain_results}
        
        # Get Redis metrics
        redis_metrics = self.get_redis_metrics()
        
        # Calculate overall health
        healthy_brains = sum(1 for brain in brain_results if brain.status == "healthy")
        total_brains = len(brain_results)
        
        if healthy_brains == total_brains:
            overall_health = "healthy"
        elif healthy_brains >= total_brains * 0.7:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            overall_health=overall_health,
            total_containers=total_brains,
            healthy_containers=healthy_brains,
            redis_memory_mb=redis_metrics["memory_mb"],
            redis_messages=redis_metrics["total_messages"],
            ai_communication_active=redis_metrics["total_messages"] > 0,
            brains=brains
        )
        
        # Store in history (keep last 100 entries)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return metrics

    async def broadcast_metrics(self, metrics: SystemMetrics):
        """Broadcast metrics to connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps(asdict(metrics), default=str)
        
        # Send to all connected clients
        disconnected_clients = set()
        for websocket in self.connected_clients:
            try:
                await websocket.send_text(message)
            except:
                disconnected_clients.add(websocket)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients

# Create FastAPI app
app = FastAPI(title="Four-Brain System Dashboard")
monitor = FourBrainMonitor()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Four-Brain System Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
            .header { text-align: center; margin-bottom: 30px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric-card { background: #2d2d2d; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .metric-card.degraded { border-left-color: #FF9800; }
            .metric-card.unhealthy { border-left-color: #F44336; }
            .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
            .metric-value { font-size: 24px; color: #4CAF50; }
            .metric-value.degraded { color: #FF9800; }
            .metric-value.unhealthy { color: #F44336; }
            .brain-status { display: flex; align-items: center; margin: 5px 0; }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
            .status-healthy { background: #4CAF50; }
            .status-degraded { background: #FF9800; }
            .status-unhealthy { background: #F44336; }
            .streams-list { max-height: 200px; overflow-y: auto; }
            .stream-item { display: flex; justify-content: space-between; margin: 3px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 Four-Brain System Dashboard</h1>
            <p>Real-time monitoring and metrics</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card" id="system-health">
                <div class="metric-title">System Health</div>
                <div class="metric-value" id="overall-status">Loading...</div>
                <div id="container-status"></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Brain Services</div>
                <div id="brain-list"></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Redis Communication</div>
                <div class="metric-value" id="redis-messages">0</div>
                <div>Memory: <span id="redis-memory">0</span> MB</div>
                <div>AI Active: <span id="ai-active">No</span></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Stream Status</div>
                <div class="streams-list" id="streams-list"></div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8080/ws');
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            function updateDashboard(metrics) {
                // Update system health
                const healthCard = document.getElementById('system-health');
                const statusElement = document.getElementById('overall-status');
                statusElement.textContent = metrics.overall_health.toUpperCase();
                statusElement.className = 'metric-value ' + metrics.overall_health;
                healthCard.className = 'metric-card ' + metrics.overall_health;
                
                document.getElementById('container-status').textContent = 
                    `${metrics.healthy_containers}/${metrics.total_containers} containers healthy`;
                
                // Update brain services
                const brainList = document.getElementById('brain-list');
                brainList.innerHTML = '';
                for (const [name, brain] of Object.entries(metrics.brains)) {
                    const brainDiv = document.createElement('div');
                    brainDiv.className = 'brain-status';
                    brainDiv.innerHTML = `
                        <div class="status-indicator status-${brain.status}"></div>
                        <div>${brain.name}: ${brain.response_time_ms.toFixed(1)}ms</div>
                    `;
                    brainList.appendChild(brainDiv);
                }
                
                // Update Redis metrics
                document.getElementById('redis-messages').textContent = metrics.redis_messages.toLocaleString();
                document.getElementById('redis-memory').textContent = metrics.redis_memory_mb.toFixed(1);
                document.getElementById('ai-active').textContent = metrics.ai_communication_active ? 'Yes' : 'No';
                
                // Update timestamp
                document.title = `Four-Brain Dashboard - ${new Date().toLocaleTimeString()}`;
            }
            
            // Reconnect on disconnect
            ws.onclose = function() {
                setTimeout(() => location.reload(), 5000);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    monitor.connected_clients.add(websocket)
    
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        monitor.connected_clients.discard(websocket)

@app.get("/api/metrics")
async def get_metrics():
    """REST API endpoint for current metrics"""
    metrics = await monitor.collect_system_metrics()
    return asdict(metrics)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

async def metrics_collector():
    """Background task to collect metrics periodically"""
    while True:
        try:
            metrics = await monitor.collect_system_metrics()
            await monitor.broadcast_metrics(metrics)
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in metrics collector: {e}")
            await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    """Start background metrics collection"""
    asyncio.create_task(metrics_collector())
    logger.info("🚀 Four-Brain Dashboard started on http://localhost:8080")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
