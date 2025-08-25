"""Lean Tool Registry for Zen MCP.

Build the tool set once at server startup, honoring env flags:
- LEAN_MODE=true|false (default false)
- LEAN_TOOLS=comma,list (when LEAN_MODE=true, overrides default lean set)
- DISABLED_TOOLS=comma,list (always excluded)

Always expose light utility tools (listmodels, version) for diagnostics.
Provide helpful error if a disabled tool is invoked.
"""
from __future__ import annotations

import os
from typing import Any, Dict

# Map tool names to import paths (module, class)
TOOL_MAP: Dict[str, tuple[str, str]] = {
    # Core
    "chat": ("tools.chat", "ChatTool"),
    "analyze": ("tools.analyze", "AnalyzeTool"),
    "debug": ("tools.debug", "DebugIssueTool"),
    "codereview": ("tools.codereview", "CodeReviewTool"),
    "refactor": ("tools.refactor", "RefactorTool"),
    "secaudit": ("tools.secaudit", "SecauditTool"),
    "planner": ("tools.planner", "PlannerTool"),
    "tracer": ("tools.tracer", "TracerTool"),
    "testgen": ("tools.testgen", "TestGenTool"),
    "consensus": ("tools.consensus", "ConsensusTool"),
    "thinkdeep": ("tools.thinkdeep", "ThinkDeepTool"),
    "docgen": ("tools.docgen", "DocgenTool"),
    # Utilities (always on)
    "version": ("tools.version", "VersionTool"),
    "listmodels": ("tools.listmodels", "ListModelsTool"),
    "self-check": ("tools.selfcheck", "SelfCheckTool"),

    # Precommit and Challenge utilities
    "precommit": ("tools.precommit", "PrecommitTool"),
    "challenge": ("tools.challenge", "ChallengeTool"),
    # Orchestrator (experimental)
    "orchestrate_auto": ("tools.orchestrate_auto", "OrchestrateAutoTool"),
}

DEFAULT_LEAN_TOOLS = {
    "chat",
    "analyze",
    "planner",
    "thinkdeep",
    "version",
    "listmodels",
}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}

    def _load_tool(self, name: str) -> None:
        module_path, class_name = TOOL_MAP[name]
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            self._tools[name] = cls()
        except Exception as e:
            self._errors[name] = str(e)

    def build_tools(self) -> None:
        disabled = {t.strip().lower() for t in os.getenv("DISABLED_TOOLS", "").split(",") if t.strip()}
        lean_mode = os.getenv("LEAN_MODE", "false").strip().lower() == "true"
        if lean_mode:
            lean_overrides = {t.strip().lower() for t in os.getenv("LEAN_TOOLS", "").split(",") if t.strip()}
            active = lean_overrides or set(DEFAULT_LEAN_TOOLS)
        else:
            active = set(TOOL_MAP.keys())

        # Ensure utilities are always on
        active.update({"version", "listmodels"})

        # Remove disabled
        active = {t for t in active if t not in disabled}

        # Hide diagnostics-only tools unless explicitly enabled
        if os.getenv("DIAGNOSTICS", "false").strip().lower() != "true":
            active.discard("self-check")

        for name in sorted(active):
            self._load_tool(name)

    def get_tool(self, name: str) -> Any:
        if name in self._tools:
            return self._tools[name]
        if name in self._errors:
            raise RuntimeError(f"Tool '{name}' failed to load: {self._errors[name]}")
        raise KeyError(
            f"Tool '{name}' is not registered. It may be disabled (LEAN_MODE/DISABLED_TOOLS) or unavailable."
        )

    def list_tools(self) -> Dict[str, Any]:
        return dict(self._tools)

    def list_descriptors(self) -> Dict[str, Any]:
        """Return machine-readable descriptors for all loaded tools (MVP)."""
        descs: Dict[str, Any] = {}
        for name, tool in self._tools.items():
            try:
                # Each tool provides a default get_descriptor()
                descs[name] = tool.get_descriptor()
            except Exception as e:
                descs[name] = {"error": f"Failed to get descriptor: {e}"}
        return descs

