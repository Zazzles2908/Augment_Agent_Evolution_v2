"""
RecommendTool - Intelligent tool and model recommendation for Zen MCP

Goals:
- Analyze user prompt (and optional files) to recommend the best Zen tool
- Suggest model category and initial model with cost-aware, free-first bias
- Respect provider restrictions and MAX_COST_PER_REQUEST
- Be deterministic by default (rule-based), with optional future LLM-assist mode

Output:
- JSON with: recommended_tool(s), category, suggested_model, reasons, hints, confidence,
  alternative_models, notes on cost/cap compatibility
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tools.simple.base import SimpleTool
from tools.shared.base_models import ToolRequest


class RecommendRequest(ToolRequest):
    prompt: str
    files: Optional[List[str]] = None
    mode: Optional[str] = None  # "auto" | "rule" (future: "llm")


class RecommendTool(SimpleTool):
    name = "recommend"
    description = (
        "INTELLIGENT TOOL RECOMMENDATIONS - Suggests the best Zen tool(s) and model based on your prompt. "
        "Balances cost and effectiveness; prefers free/cheap paths; escalates only when warranted."
    )

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_model_category(self):
        # Recommendations are lightweight → FAST_RESPONSE
        from tools.models import ToolModelCategory
        return ToolModelCategory.FAST_RESPONSE

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        return {
            "prompt": {
                "type": "string",
                "description": "User intent or task description to analyze",
            },
            "files": self.FILES_FIELD,
            "mode": {
                "type": "string",
                "enum": ["auto", "rule"],
                "description": "Strategy: 'rule' = deterministic heuristics; 'auto' same as rule for now",
                "default": "auto",
            },
        }

    def get_required_fields(self) -> list[str]:
        return ["prompt"]

    def get_system_prompt(self) -> str:
        # Not used (we're not calling an LLM by default)
        return ""

    async def prepare_prompt(self, request) -> str:
        # Not used
        return ""

    async def execute(self, arguments: Dict[str, Any]) -> List:
        from mcp.types import TextContent
        from tools.models import ToolOutput

        try:
            req = RecommendRequest(**arguments)
        except Exception as e:
            out = ToolOutput(status="error", content=f"Invalid request: {e}")
            return [TextContent(type="text", text=out.model_dump_json())]

        prompt = (req.prompt or "").strip()
        text = prompt.lower()

        # Extract simple hints
        hints: List[str] = []
        if any(k in text for k in ("image", "diagram", "vision")):
            hints.append("vision")
        if any(k in text for k in ("deep reasoning", "think", "chain of thought", "cot", "reason")):
            hints.append("deep_reasoning")
        if any(k in text for k in ("security", "owasp", "xss", "sql injection", "csrf")):
            hints.append("security")
        if any(k in text for k in ("test", "unit test", "pytest")):
            hints.append("testing")
        if any(k in text for k in ("refactor", "cleanup", "modernize")):
            hints.append("refactor")
        if any(k in text for k in ("debug", "traceback", "error", "exception")):
            hints.append("debug")
        if any(k in text for k in ("plan", "roadmap", "design")):
            hints.append("planning")
        if any(k in text for k in ("review", "pull request", "diff")):
            hints.append("codereview")
        if any(k in text for k in ("docs", "documentation", "readme")):
            hints.append("docgen")
        if any(k in text for k in ("precommit", "lint", "format")):
            hints.append("precommit")
        if any(k in text for k in ("call flow", "trace", "stack")):
            hints.append("tracer")

        # Choose tool(s)
        recommended_tools: List[str] = []
        if "vision" in hints:
            recommended_tools.append("chat")  # vision chat or analyze
        if "security" in hints:
            recommended_tools.append("secaudit")
        if "testing" in hints:
            recommended_tools.append("testgen")
        if "refactor" in hints:
            recommended_tools.append("refactor")
        if "debug" in hints:
            recommended_tools.append("debug")
        if "planning" in hints:
            recommended_tools.append("planner")
        if "codereview" in hints:
            recommended_tools.append("codereview")
        if "docgen" in hints:
            recommended_tools.append("docgen")
        if "precommit" in hints:
            recommended_tools.append("precommit")
        if "tracer" in hints:
            recommended_tools.append("tracer")

        # Default if no strong signal
        if not recommended_tools:
            # If looks complex, recommend thinkdeep + analyze chain
            if "deep_reasoning" in hints:
                recommended_tools = ["thinkdeep"]
            else:
                recommended_tools = ["chat"]

        # Map to category
        from tools.models import ToolModelCategory
        if "thinkdeep" in recommended_tools or "secaudit" in recommended_tools or "debug" in hints:
            category = ToolModelCategory.EXTENDED_REASONING
        elif "chat" in recommended_tools or "planner" in recommended_tools:
            category = ToolModelCategory.FAST_RESPONSE
        else:
            category = ToolModelCategory.BALANCED

        # Suggest model using registry preference and hints-aware bias
        from providers.registry import ModelProviderRegistry
        # Use get_preferred_fallback_model for a baseline suggestion
        baseline_model = ModelProviderRegistry.get_preferred_fallback_model(category)

        # Build candidate list according to our earlier policy
        candidates: List[str] = []
        if category.name == "FAST_RESPONSE":
            candidates = ["glm-4.5-flash", "glm-4.5-air", "glm-4.5"]
        elif category.name == "EXTENDED_REASONING":
            candidates = ["kimi-k2-thinking", "glm-4.5-airx", "kimi-k2-0711-preview", "glm-4.5"]
        else:  # BALANCED
            candidates = ["glm-4.5-air", "glm-4.5", "kimi-k2-turbo-preview"]

        # Vision override if hinted
        if "vision" in hints:
            candidates = ["glm-4.5v"] + candidates

        # Filter to allowed models and respect MAX_COST_PER_REQUEST
        allowed = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        allowed_set = set(allowed.keys())
        candidates = [m for m in candidates if m in allowed_set]

        # Apply cost cap ordering using env costs
        def load_costs() -> dict[str, float]:
            try:
                raw = os.getenv("MODEL_COSTS_JSON", "{}")
                data = json.loads(raw)
                return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}
            except Exception:
                return {}
        costs = load_costs()
        max_cost = None
        try:
            val = os.getenv("MAX_COST_PER_REQUEST")
            max_cost = float(val) if val else None
        except Exception:
            max_cost = None
        def within_cap(m: str) -> bool:
            return (m not in costs) or (max_cost is None) or (costs[m] <= max_cost)
        filtered = [m for m in candidates if within_cap(m)]
        # Sort by configured costs (unknowns last)
        filtered.sort(key=lambda m: (0, costs[m]) if m in costs else (1, float('inf')))
        suggested_model = filtered[0] if filtered else baseline_model

        # Provide alternatives (next 2)
        alternatives = [m for m in filtered[1:3]]

        # Relative cost label
        def cost_label(m: str) -> str:
            c = costs.get(m)
            if c is None:
                return "unknown"
            if c == 0.0:
                return "free"
            if c <= 1.5:
                return "low"
            if c <= 4.5:
                return "moderate"
            return "high"

        # Construct response
        reasons = []
        if "vision" in hints:
            reasons.append("Vision hint detected; favoring glm-4.5v when allowed and within cap")
        if category.name == "EXTENDED_REASONING":
            reasons.append("Complex reasoning; preferring Kimi thinking or GLM AirX within cap")
        if category.name == "FAST_RESPONSE":
            reasons.append("Fast path prioritizes free/cheap GLM Flash/Air")

        result = {
            "recommended_tools": recommended_tools,
            "category": category.name,
            "suggested_model": suggested_model,
            "suggested_model_cost": costs.get(suggested_model),
            "suggested_model_cost_label": cost_label(suggested_model),
            "alternatives": alternatives,
            "hints": hints,
            "reasons": reasons or ["Heuristic mapping based on prompt content and cost policy"],
            "confidence": "high" if hints else "medium",
            "notes": {
                "baseline_model": baseline_model,
                "max_cost_per_request": max_cost,
                "free_first": os.getenv("FREE_TIER_PREFERENCE_ENABLED", "false").lower() == "true",
            },
        }

        out = ToolOutput(status="success", content=json.dumps(result, ensure_ascii=False), content_type="json")
        return [TextContent(type="text", text=out.model_dump_json())]

