"""
Auggie-optimized wrapper tools for Zen MCP Server.

These wrappers provide CLI-friendly output, progress indicators, error handling,
and plug into the registry's category-aware fallback and telemetry.

Wrappers are safe no-ops when Auggie config is not present. They attempt to
integrate with optional auggie.session and auggie.templates modules when available.

Functions (async):
- aug_chat(args: dict) -> str
- aug_thinkdeep(args: dict) -> str
- aug_consensus(args: dict) -> str
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from providers.base import ModelResponse
from providers.registry import ModelProviderRegistry as Registry
from auggie.selector import select_model
from auggie.templates import render_template

try:
    from auggie.config import get_auggie_settings
except Exception:  # pragma: no cover
    def get_auggie_settings():
        return {}

logger = logging.getLogger(__name__)


# --- Optional session integration (file/redis-backed implemented in Phase 1 Step 4) ---
# Session integration
try:
    from auggie.session import get_session as _sess_get, update_session as _sess_update, append_history as _sess_append
except Exception:  # pragma: no cover
    def _sess_get(sid: Optional[str]): return {}
    def _sess_update(sid: Optional[str], **kv): return {}
    def _sess_append(sid: Optional[str], entry: dict): return {}



# --- Optional templates integration ---

def _render_template_if_available(name: Optional[str], variables: dict[str, Any], default: str) -> str:
    if not name:
        return default
    try:
        from auggie.templates import render_template  # type: ignore

        return render_template(name, variables)
    except Exception:
        return default


# --- CLI formatting helpers ---

def _format_cli(data: dict[str, Any], compact: bool = True) -> str:
    """Format a small, structured CLI output."""
    if compact:
        # Compact: single JSON line for machine-friendly consumption
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            pass
    # Pretty text fallback
    parts = []
    if m := data.get("model"):
        parts.append(f"[Model] {m}")
    if p := data.get("provider"):
        parts.append(f"[Provider] {p}")
    if lat := data.get("latency_ms"):
        parts.append(f"[Latency] {lat} ms")
    if usage := data.get("usage"):
        it = usage.get("input_tokens", 0)
        ot = usage.get("output_tokens", 0)
        tt = usage.get("total_tokens", it + ot)
        parts.append(f"[Tokens] in={it} out={ot} total={tt}")
    if pr := data.get("progress"):
        parts.append("[Progress]\n- " + "\n- ".join(pr))
    if err := data.get("error"):
        parts.append(f"[Error] {err}")
    if content := data.get("content"):
        parts.append("[Output]\n" + content)
    return "\n".join(parts)


def _get_wrapper_settings() -> dict[str, Any]:
    s = get_auggie_settings() or {}
    w = s.get("wrappers") or {}
    return {
        "show_progress": bool(w.get("show_progress", True)),
        "compact_output": bool(w.get("compact_output", True)),
        "error_detail": str(w.get("error_detail", "detailed")),
    }


# --- Core execution helper using fallback ---
async def _exec_with_fallback(
    category: Any,
    prompt: str,
    model: Optional[str],
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    extra: Optional[dict] = None,
) -> tuple[Optional[ModelResponse], str, list[str]]:
    """Execute a model call via registry fallback.
    Returns: (response, selected_model_name, progress)
    """
    settings = _get_wrapper_settings()
    progress: list[str] = []
    if settings["show_progress"]:
        progress.append("Resolving model and calling provider...")

    # Prepare call fn for registry
    def call_fn(selected_model: str) -> ModelResponse:
        provider = Registry.get_provider_for_model(selected_model)
        if provider is None:
            raise RuntimeError(f"No provider for model {selected_model}")
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if extra:
            kwargs.update(extra)
        return provider.generate_content(prompt=prompt, model_name=selected_model, **kwargs)

    # If a specific model is requested, run directly but still collect telemetry via call_with_fallback on single-item chain
    if model and model.lower() != "auto":
        # Wrap single model execution in fallback to reuse telemetry
        def single_chain(cat):
            return [model]

        # Monkeypatch chain temporarily by wrapping call
        orig_chain = Registry._auggie_fallback_chain
        try:
            Registry._auggie_fallback_chain = classmethod(lambda cls, c: [model])  # type: ignore
            resp = Registry.call_with_fallback(category, call_fn)
            if settings["show_progress"]:
                progress.append(f"Response received from {model}")
            return resp, model, progress
        finally:
            Registry._auggie_fallback_chain = orig_chain  # type: ignore

    # Auto or unspecified: use full fallback chain
    resp = Registry.call_with_fallback(category, call_fn)
    chosen = getattr(resp, "model_name", "") or "auto"
    if settings["show_progress"]:
        progress.append(f"Response received from {chosen}")
    return resp, chosen, progress


# --- Public wrappers ---
async def aug_chat(args: dict) -> str:
    """Auggie-optimized chat wrapper (FAST_RESPONSE category)."""
    from tools.models import ToolModelCategory as Cat

    ws = _get_wrapper_settings()
    session_id = args.get("session_id") or args.get("continuation_id")
    prompt = args.get("prompt", "").strip()
    template = args.get("template")
    model = args.get("model", "auto")
    temperature = args.get("temperature")

    # Choose default template if configured to auto_use
    ws_all = get_auggie_settings() or {}
    tconf = (ws_all.get("templates") or {})
    auto_use = bool(tconf.get("auto_use", True))
    if auto_use and not template:
        template = "chat_basic"
    final_prompt = _render_template_if_available(template, {"prompt": prompt, "session_id": session_id}, prompt)

    # selector: explainable choice; still executed via fallback
    if (model or "auto").lower() == "auto":
        sel_model, sel_reason = select_model("FAST_RESPONSE", {"code": False})
        if sel_model:
            model = sel_model
            if (ws_all.get("selector") or {}).get("explanations", True) and ws["show_progress"]:
                # Include explanation in progress
                progress_note = f"Selector chose: {model} ({sel_reason})"
                # Delay list creation until after progress initialized below
                pass

    try:
        resp, chosen, progress = await _exec_with_fallback(Cat.FAST_RESPONSE, final_prompt, model, temperature)
        if ws["show_progress"] and model and model != "auto":
            note = f"Selector chose: {model}"
            if 'sel_reason' in locals() and (ws_all.get("selector") or {}).get("explanations", True):
                note = f"{note} ({sel_reason})"
            progress.insert(0, note)
        usage = getattr(resp, "usage", {}) if resp else {}
        data = {
            "tool": "aug_chat",
            "model": chosen,
            "content": resp.content if resp else "",
            "usage": usage,
            "progress": progress if ws["show_progress"] else [],
        }
        _sess_update(session_id, last_model=chosen)
        _sess_append(session_id, {"tool": "aug_chat", "model": chosen, "prompt": final_prompt, "content": data["content"]})
        return _format_cli(data, compact=ws["compact_output"])
    except Exception as e:
        err = str(e)
        data = {"tool": "aug_chat", "error": err, "progress": ["retry suggested: change model or simplify prompt"] if ws["show_progress"] else []}
        return _format_cli(data, compact=ws["compact_output"])


async def aug_thinkdeep(args: dict) -> str:
    """Auggie-optimized thinkdeep wrapper (EXTENDED_REASONING category)."""
    from tools.models import ToolModelCategory as Cat
    from config import DEFAULT_THINKING_MODE_THINKDEEP as DEFAULT_THINKING

    ws = _get_wrapper_settings()
    ws_all = get_auggie_settings() or {}
    session_id = args.get("session_id") or args.get("continuation_id")
    prompt = args.get("step") or args.get("prompt") or ""
    template = args.get("template")
    model = args.get("model", "auto")
    # Map thinking_mode to provider extra kwargs when supported (placeholder)
    thinking_mode = args.get("thinking_mode", DEFAULT_THINKING)

    final_prompt = _render_template_if_available(template, {"prompt": prompt, "session_id": session_id, "thinking_mode": thinking_mode}, str(prompt))

    if (model or "auto").lower() == "auto":
        sel_model, sel_reason = select_model("EXTENDED_REASONING", {"code": False})
        if sel_model:
            model = sel_model

    try:
        resp, chosen, progress = await _exec_with_fallback(Cat.EXTENDED_REASONING, final_prompt, model, extra={"thinking_mode": thinking_mode})
        if ws["show_progress"] and model and model != "auto":
            note = f"Selector chose: {model}"
            if 'sel_reason' in locals() and (ws_all.get("selector") or {}).get("explanations", True):
                note = f"{note} ({sel_reason})"
            progress.insert(0, note)
        usage = getattr(resp, "usage", {}) if resp else {}
        data = {
            "tool": "aug_thinkdeep",
            "model": chosen,
            "content": resp.content if resp else "",
            "usage": usage,
            "progress": (["Thinking mode: "+str(thinking_mode)] + progress) if ws["show_progress"] else [],
        }
        _sess_update(session_id, last_model=chosen)
        _sess_append(session_id, {"tool": "aug_thinkdeep", "model": chosen, "prompt": final_prompt, "thinking_mode": thinking_mode, "content": data["content"]})
        return _format_cli(data, compact=ws["compact_output"])
    except Exception as e:
        data = {"tool": "aug_thinkdeep", "error": str(e), "progress": ["retry suggested: lower thinking_mode or switch model"] if ws["show_progress"] else []}
        return _format_cli(data, compact=ws["compact_output"])


async def aug_consensus(args: dict) -> str:
    """Auggie-optimized consensus wrapper: compare two models and synthesize.
    If 'models' not provided, pick first two from fallback chain for EXTENDED_REASONING.
    """
    from tools.models import ToolModelCategory as Cat

    ws = _get_wrapper_settings()
    session_id = args.get("session_id") or args.get("continuation_id")
    prompt = args.get("step") or args.get("prompt") or ""
    template = args.get("template")
    models: list[str] = args.get("models") or []

    final_prompt = _render_template_if_available(template, {"prompt": prompt, "session_id": session_id}, str(prompt))

    # Build candidate models if not specified
    if not models:
        try:
            chain = Registry._auggie_fallback_chain(Cat.EXTENDED_REASONING)
            # Choose first two distinct
            seen = set()
            models = []
            for m in chain:
                if m not in seen:
                    seen.add(m)
                    models.append(m)
                if len(models) >= 2:
                    break
        except Exception:
            models = []

    chosen_models = models[:2] if models else []
    if not chosen_models:
        # Fallback to best single model
        chosen_models = [Registry.get_preferred_fallback_model(Cat.EXTENDED_REASONING)]

    results: list[dict[str, Any]] = []
    progress: list[str] = ["Comparing models..."] if ws["show_progress"] else []

    for m in chosen_models:
        try:
            # Run each model through fallback but pin to model m so telemetry still records
            resp, chosen, _ = await _exec_with_fallback(Cat.EXTENDED_REASONING, final_prompt, m)
            usage = getattr(resp, "usage", {}) if resp else {}
            results.append({"model": chosen, "content": resp.content if resp else "", "usage": usage})
            if ws["show_progress"]:
                progress.append(f"Received response from {chosen}")
        except Exception as e:
            results.append({"model": m, "error": str(e)})
            if ws["show_progress"]:
                progress.append(f"Error from {m}: {e}")

    # Simple synthesis placeholder: prefer first content, append second delta note
    synthesis = ""
    if results:
        synthesis = (results[0].get("content") or "").strip()
        if len(results) > 1:
            synthesis += "\n\n---\nConsensus note: Compared against " + results[1]["model"]

    data = {
        "tool": "aug_consensus",
        "models": [r.get("model") for r in results],
        "results": results,
        "synthesis": synthesis,
        "progress": progress if ws["show_progress"] else [],
    }
    return _format_cli(data, compact=ws["compact_output"])

