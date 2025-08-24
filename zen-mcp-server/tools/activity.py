"""
ActivityTool - Surface recent MCP activity/logs for visibility in clients

Returns recent lines from logs/mcp_server.log, optionally filtered.
Useful when client UI does not show per-step dropdowns.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from tools.simple.base import SimpleTool
from tools.shared.base_models import ToolRequest


class ActivityRequest(ToolRequest):
    lines: Optional[int] = 200
    filter: Optional[str] = None  # regex


class ActivityTool(SimpleTool):
    name = "activity"
    description = (
        "MCP ACTIVITY VIEW - Returns recent server activity from logs/mcp_server.log. "
        "Supports optional regex filtering and line count control."
    )

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_model_category(self):
        from tools.models import ToolModelCategory
        return ToolModelCategory.FAST_RESPONSE

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "lines": {
                "type": "integer",
                "minimum": 10,
                "maximum": 5000,
                "default": 200,
                "description": "Number of log lines from the end of the file to return",
            },
            "filter": {
                "type": "string",
                "description": "Optional regex to filter lines (e.g., 'TOOL_CALL|CallToolRequest')",
            },
        }

    def get_required_fields(self) -> list[str]:
        return []

    def get_system_prompt(self) -> str:
        return ""

    async def prepare_prompt(self, request) -> str:
        return ""

    async def execute(self, arguments: Dict[str, Any]) -> List:
        from mcp.types import TextContent
        from tools.models import ToolOutput

        try:
            req = ActivityRequest(**arguments)
        except Exception as e:
            return [TextContent(type="text", text=ToolOutput(status="error", content=str(e)).model_dump_json())]

        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "mcp_server.log")
        if not os.path.isfile(log_path):
            return [TextContent(type="text", text=ToolOutput(status="error", content=f"Log file not found: {log_path}").model_dump_json())]

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            return [TextContent(type="text", text=ToolOutput(status="error", content=f"Failed to read log: {e}").model_dump_json())]

        n = max(10, int(req.lines or 200))
        tail = lines[-n:]

        if req.filter:
            try:
                pattern = re.compile(req.filter)
                tail = [ln for ln in tail if pattern.search(ln)]
            except Exception as e:
                return [TextContent(type="text", text=ToolOutput(status="error", content=f"Invalid filter regex: {e}").model_dump_json())]

        # Return as plain text block with minimal formatting
        content = "".join(tail[-n:])
        payload = ToolOutput(status="success", content=content, content_type="text")
        return [TextContent(type="text", text=payload.model_dump_json())]

