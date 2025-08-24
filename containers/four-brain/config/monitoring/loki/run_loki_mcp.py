#!/usr/bin/env python3
"""
Loki MCP Server Runner for Zazzles's Agent AI
Connects to local Loki instance for Four-Brain log monitoring
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import requests
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Loki MCP Server Implementation
class LokiMCPServer:
    def __init__(self, loki_url: str = "http://localhost:3100"):
        self.loki_url = loki_url.rstrip('/')
        self.server = Server("loki-four-brain")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup MCP tools for Loki operations"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="query_loki_logs",
                    description="Query Loki logs using LogQL",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "LogQL query"},
                            "limit": {"type": "integer", "description": "Number of log lines to return", "default": 100},
                            "start": {"type": "string", "description": "Start time (RFC3339 or relative like '1h')", "default": "1h"},
                            "end": {"type": "string", "description": "End time (RFC3339 or 'now')", "default": "now"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="list_loki_labels",
                    description="List available labels in Loki",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "start": {"type": "string", "description": "Start time", "default": "1h"},
                            "end": {"type": "string", "description": "End time", "default": "now"}
                        }
                    }
                ),
                Tool(
                    name="get_loki_label_values",
                    description="Get values for a specific label",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Label name"},
                            "start": {"type": "string", "description": "Start time", "default": "1h"},
                            "end": {"type": "string", "description": "End time", "default": "now"}
                        },
                        "required": ["label"]
                    }
                ),
                Tool(
                    name="get_loki_stats",
                    description="Get Loki statistics and health",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "query_loki_logs":
                    return await self._query_logs(arguments)
                elif name == "list_loki_labels":
                    return await self._list_labels(arguments)
                elif name == "get_loki_label_values":
                    return await self._get_label_values(arguments)
                elif name == "get_loki_stats":
                    return await self._get_stats(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _parse_time(self, time_str: str) -> str:
        """Parse time string to RFC3339 format"""
        if time_str == "now":
            return datetime.utcnow().isoformat() + "Z"
        elif time_str.endswith('h'):
            hours = int(time_str[:-1])
            return (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
        elif time_str.endswith('m'):
            minutes = int(time_str[:-1])
            return (datetime.utcnow() - timedelta(minutes=minutes)).isoformat() + "Z"
        else:
            return time_str
    
    async def _query_logs(self, args: Dict[str, Any]) -> List[TextContent]:
        """Query Loki logs"""
        query = args["query"]
        limit = args.get("limit", 100)
        start = self._parse_time(args.get("start", "1h"))
        end = self._parse_time(args.get("end", "now"))
        
        params = {
            "query": query,
            "limit": limit,
            "start": start,
            "end": end,
            "direction": "backward"
        }
        
        response = requests.get(f"{self.loki_url}/loki/api/v1/query_range", params=params)
        response.raise_for_status()
        
        data = response.json()
        result_text = f"Loki Query: {query}\n"
        result_text += f"Time Range: {start} to {end}\n"
        result_text += f"Results: {len(data.get('data', {}).get('result', []))} streams\n\n"
        
        for stream in data.get('data', {}).get('result', []):
            labels = stream.get('stream', {})
            result_text += f"Stream: {labels}\n"
            for entry in stream.get('values', [])[:10]:  # Show first 10 entries per stream
                timestamp, log_line = entry
                result_text += f"  {timestamp}: {log_line}\n"
            result_text += "\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def _list_labels(self, args: Dict[str, Any]) -> List[TextContent]:
        """List available labels"""
        start = self._parse_time(args.get("start", "1h"))
        end = self._parse_time(args.get("end", "now"))
        
        params = {"start": start, "end": end}
        response = requests.get(f"{self.loki_url}/loki/api/v1/labels", params=params)
        response.raise_for_status()
        
        data = response.json()
        labels = data.get('data', [])
        
        result_text = f"Available Loki Labels ({len(labels)} total):\n"
        for label in sorted(labels):
            result_text += f"  - {label}\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def _get_label_values(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get values for a specific label"""
        label = args["label"]
        start = self._parse_time(args.get("start", "1h"))
        end = self._parse_time(args.get("end", "now"))
        
        params = {"start": start, "end": end}
        response = requests.get(f"{self.loki_url}/loki/api/v1/label/{label}/values", params=params)
        response.raise_for_status()
        
        data = response.json()
        values = data.get('data', [])
        
        result_text = f"Values for label '{label}' ({len(values)} total):\n"
        for value in sorted(values):
            result_text += f"  - {value}\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def _get_stats(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get Loki statistics"""
        try:
            # Check Loki health
            health_response = requests.get(f"{self.loki_url}/ready")
            health_status = "Healthy" if health_response.status_code == 200 else "Unhealthy"
            
            # Get metrics
            metrics_response = requests.get(f"{self.loki_url}/metrics")
            metrics_text = metrics_response.text if metrics_response.status_code == 200 else "Metrics unavailable"
            
            result_text = f"Loki Statistics:\n"
            result_text += f"Health Status: {health_status}\n"
            result_text += f"Loki URL: {self.loki_url}\n"
            result_text += f"Metrics Size: {len(metrics_text)} bytes\n"
            
            # Extract some key metrics
            if "loki_" in metrics_text:
                lines = metrics_text.split('\n')
                loki_metrics = [line for line in lines if line.startswith('loki_') and not line.startswith('#')]
                result_text += f"Loki Metrics Available: {len(loki_metrics)}\n"
            
            return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting Loki stats: {str(e)}")]

def main():
    """Main entry point for Loki MCP server"""
    loki_url = os.environ.get('LOKI_URL', 'http://localhost:3100')
    
    server = LokiMCPServer(loki_url)
    
    async def run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.server.run(read_stream, write_stream, server.server.create_initialization_options())
    
    asyncio.run(run())

if __name__ == "__main__":
    main()
