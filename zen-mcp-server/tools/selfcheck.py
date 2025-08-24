"""
Self-Check Tool - Quick diagnostics for Zen MCP Server

Returns:
- Providers configured (with API keys present)
- Number of visible tools in the active registry
- Presence of key environment variables (names only, no values)
- Tail of logs/mcp_server.log for quick troubleshooting
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from mcp.types import TextContent

from tools.shared.base_tool import BaseTool
from tools.shared.base_models import ToolRequest

logger = logging.getLogger(__name__)


class SelfCheckTool(BaseTool):
    def get_name(self) -> str:
        return "self-check"

    def get_description(self) -> str:
        return (
            "SERVER SELF-CHECK - Return providers configured, tool count, key env presence, and recent log tail. "
            "Helpful for quick diagnostics via Auggie CLI."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "log_lines": {
                    "type": "integer",
                    "description": "How many log lines to return from the end of mcp_server.log (default 40)",
                    "minimum": 0,
                    "maximum": 500,
                }
            },
            "required": [],
        }

    def get_system_prompt(self) -> str:
        return ""

    def requires_model(self) -> bool:
        return False

    def get_annotations(self) -> Optional[dict[str, Any]]:
        return {"readOnlyHint": True}

    async def prepare_prompt(self, request: ToolRequest) -> str:
        return ""

    def format_response(self, response: str, request: ToolRequest, model_info: dict | None = None) -> str:
        return response

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        from providers.registry import ModelProviderRegistry

        log_lines = int(arguments.get("log_lines") or 40)

        # Providers with configured keys
        try:
            provider_types = ModelProviderRegistry.get_available_providers_with_keys()
            providers = [pt.name for pt in provider_types]
        except Exception as e:
            providers = []
            logger.warning(f"Provider discovery failed: {e}")

        # Visible tool count from active registry
        tool_count = None
        try:
            # Prefer the active registry in server module to avoid re-instantiation
            from server import TOOLS as ACTIVE_TOOLS  # type: ignore

            tool_count = len(ACTIVE_TOOLS)
            visible_tool_names = sorted(list(ACTIVE_TOOLS.keys()))
        except Exception as e:
            visible_tool_names = []
            logger.warning(f"Could not access active tool registry: {e}")
            try:
                # Fallback: build a temporary registry and count
                from tools.registry import ToolRegistry

                _tmp = ToolRegistry()
                _tmp.build_tools()
                tool_count = len(_tmp.list_tools())
                visible_tool_names = sorted(list(_tmp.list_tools().keys()))
            except Exception as e2:
                logger.warning(f"Fallback registry build failed: {e2}")

        # Key env presence (names only)
        key_envs = [
            "KIMI_API_KEY",
            "GLM_API_KEY",
            "OPENROUTER_API_KEY",
            "CUSTOM_API_URL",
        ]
        env_presence = {name: (name in os.environ and bool(os.environ.get(name))) for name in key_envs}

        # Log tail
        log_tail_lines: list[str] = []
        try:
            log_path = Path(__file__).resolve().parent.parent / "logs" / "mcp_server.log"
            if log_path.exists():
                # Read last N lines without loading entire file if possible
                # Simple safe approach for moderate file sizes
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.readlines()
                    if log_lines > 0:
                        log_tail_lines = [line.rstrip("\n") for line in content[-log_lines:]]
            else:
                log_tail_lines = [f"Log file not found: {log_path}"]
        except Exception as e:
            log_tail_lines = [f"Error reading log: {e}"]

        # Build output
        lines: list[str] = []
        lines.append("# Zen MCP Server Self-Check")
        lines.append("")
        lines.append("## Providers Configured (keys present)")
        lines.append(", ".join(providers) if providers else "None")
        lines.append("")
        lines.append("## Visible Tools")
        lines.append(f"Count: {tool_count if tool_count is not None else 'Unknown'}")
        if visible_tool_names:
            lines.append("Names: " + ", ".join(visible_tool_names))
        lines.append("")
        lines.append("## Environment Variables (presence only)")
        for k, present in env_presence.items():
            lines.append(f"- {k}: {'[PRESENT]' if present else '[MISSING]'}")
        lines.append("")
        lines.append("## Log Tail (mcp_server.log)")
        if log_tail_lines:
            lines.extend(["    " + l for l in log_tail_lines])
        else:
            lines.append("(no logs available)")

        return [TextContent(type="text", text="\n".join(lines))]

