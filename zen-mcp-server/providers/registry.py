"""Model provider registry for managing available providers."""

import logging
import os
import threading
from typing import TYPE_CHECKING, Optional, Any

from .base import ModelProvider, ProviderType

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.health import HealthManager, CircuitState  # self-healing integration
import asyncio
import time

# Feature flags (env-driven)
_DEF = lambda k, d: os.getenv(k, d).lower()

def _health_enabled() -> bool:
    return _DEF("HEALTH_CHECKS_ENABLED", "false") == "true"

def _cb_enabled() -> bool:
    return _DEF("CIRCUIT_BREAKER_ENABLED", "false") == "true"

def _health_log_only() -> bool:
    return _DEF("HEALTH_LOG_ONLY", "true") == "true"

def _retry_attempts() -> int:
    try:
        return int(os.getenv("RETRY_ATTEMPTS", "2"))
    except Exception:
        return 2

def _backoff_base() -> float:
    try:
        return float(os.getenv("RETRY_BACKOFF_BASE", "0.5"))
    except Exception:
        return 0.5

def _backoff_max() -> float:
    try:
        return float(os.getenv("RETRY_BACKOFF_MAX", "4.0"))
    except Exception:
        return 4.0
def _free_tier_enabled() -> bool:
    return os.getenv("FREE_TIER_PREFERENCE_ENABLED", "false").lower() == "true"

def _free_model_list() -> set[str]:
    raw = os.getenv("FREE_MODEL_LIST", "")
    return set([m.strip().lower() for m in raw.split(",") if m.strip()])

def _apply_free_first(models: list[str]) -> list[str]:
    if not _free_tier_enabled():
        return models
    free = _free_model_list()
    if not free:
        return models
    free_models = [m for m in models if m.lower() in free]
    paid_models = [m for m in models if m.lower() not in free]
    return free_models + paid_models

def _cost_aware_enabled() -> bool:
    return os.getenv("COST_AWARE_ROUTING_ENABLED", "false").lower() == "true"

def _load_model_costs() -> dict[str, float]:
    import json
    raw = os.getenv("MODEL_COSTS_JSON", "{}")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def _max_cost_per_request() -> float | None:
    try:
        val = os.getenv("MAX_COST_PER_REQUEST")
        return float(val) if val else None
    except Exception:
        return None

def _apply_cost_aware(models: list[str], tool_category) -> list[str]:
    if not _cost_aware_enabled():
        return models
    costs = _load_model_costs()
    if not costs:
        return models
    max_cost = _max_cost_per_request()
    # Filter and sort by configured cost if present; unknown costs retain original order at end
    def model_key(m: str):
        return (0, costs[m]) if m in costs else (1, float('inf'))
    filtered = [m for m in models if (max_cost is None or (m in costs and costs[m] <= max_cost)) or (m not in costs)]
    filtered.sort(key=model_key)
    return filtered


_HEALTH_MANAGER: HealthManager | None = None

def _get_health_manager() -> HealthManager:
    global _HEALTH_MANAGER
    if _HEALTH_MANAGER is None:
        _HEALTH_MANAGER = HealthManager()
    return _HEALTH_MANAGER

class HealthWrappedProvider(ModelProvider):
    """Wrapper that records provider health and applies simple retry/backoff.
    Only active when HEALTH_CHECKS_ENABLED=true. Selection gating happens in registry.
    """
    def __init__(self, inner: ModelProvider):
        self._inner = inner
        self._ptype = inner.get_provider_type()
        self._health = _get_health_manager().get(self._ptype.value)

    # Pass-throughs to satisfy abstract interface
    def get_provider_type(self) -> ProviderType:
        return self._ptype

    def supports_thinking_mode(self, model_name: str) -> bool:
        return self._inner.supports_thinking_mode(model_name)

    # Ensure restriction validation sees real model lists/capabilities
    def get_model_configurations(self) -> dict[str, Any]:
        return self._inner.get_model_configurations()

    def get_all_model_aliases(self) -> dict[str, list[str]]:
        return self._inner.get_all_model_aliases()

    def list_models(self, respect_restrictions: bool = True) -> list[str]:
        return self._inner.list_models(respect_restrictions)

    def list_all_known_models(self) -> list[str]:
        return self._inner.list_all_known_models()

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        # Delegate if inner provider implements preference; else no preference
        try:
            return self._inner.get_preferred_model(category, allowed_models)  # type: ignore[attr-defined]
        except Exception:
            return None

    # Legacy compatibility: alias to inner when callers expect raw provider type
    def get_provider_type_raw(self):
        return self._inner.get_provider_type()

    @staticmethod
    def _schedule(coro) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except Exception:
            # If no loop or running in a threadpool, skip recording instead of crashing
            pass

    # Forwarding methods
    def get_capabilities(self, model_name: str):
        return self._inner.get_capabilities(model_name)

    def validate_model_name(self, model_name: str) -> bool:
        return self._inner.validate_model_name(model_name)

    def list_models(self, respect_restrictions: bool = True):
        fn = getattr(self._inner, "list_models", None)
        if callable(fn):
            return fn(respect_restrictions=respect_restrictions)
        raise NotImplementedError

    def supports_thinking_mode(self, model_name: str) -> bool:
        fn = getattr(self._inner, "supports_thinking_mode", None)
        if callable(fn):
            try:
                return bool(fn(model_name))
            except Exception:
                return False
        return False

    def count_tokens(self, text: str, model_name: str) -> int:
        # Token count is fast; treat errors as failure but no retries
        try:
            val = self._inner.count_tokens(text, model_name)
            if _health_enabled() and not _health_log_only():
                HealthWrappedProvider._schedule(self._health.record_result(True))
            return val
        except Exception:
            if _health_enabled():
                # log-only still records but does not gate selection here
                HealthWrappedProvider._schedule(self._health.record_result(False))
            raise

    def generate_content(self, prompt: str, model_name: str, system_prompt: str | None = None, temperature: float = 0.3, max_output_tokens: int | None = None, **kwargs):
        attempts = max(1, _retry_attempts())
        delay = _backoff_base()
        last_exc = None
        for i in range(attempts):
            try:
                t0 = time.perf_counter()
                result = self._inner.generate_content(
                    prompt=prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    **kwargs,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if _health_enabled() and not _health_log_only():
                    HealthWrappedProvider._schedule(self._health.record_result(True))
                try:
                    from utils.metrics import record_provider_call
                    record_provider_call(self._ptype.value, model_name, True, latency_ms)
                except Exception:
                    pass
                return result
            except Exception as e:
                last_exc = e
                if _health_enabled():
                    HealthWrappedProvider._schedule(self._health.record_result(False))
                try:
                    from utils.metrics import record_provider_call
                    record_provider_call(self._ptype.value, model_name, False, None)
                except Exception:
                    pass
                # Simple backoff before retrying
                if i < attempts - 1:
                    time.sleep(min(delay, _backoff_max()))
                    delay *= 2
        # Exceeded retries
        raise last_exc


class ModelProviderRegistry:
    """Registry for managing model providers."""

    _instance = None
    # In-memory telemetry (lightweight). Structure:
    # telemetry = {
    #   "model_name": {"success": int, "failure": int, "latency_ms": [..], "input_tokens": int, "output_tokens": int}
    # }
    _telemetry: dict[str, dict[str, Any]] = {}
    _telemetry_lock = threading.RLock()

    # Provider priority order for model selection
    # Native APIs first (prioritize Kimi/GLM per project usage), then custom endpoints, then catch-all providers
    PROVIDER_PRIORITY_ORDER = [
        ProviderType.KIMI,   # Direct Kimi/Moonshot access (preferred)
        ProviderType.GLM,    # Direct GLM/ZhipuAI access (preferred)
        ProviderType.CUSTOM, # Local/self-hosted models
        ProviderType.OPENROUTER,  # Catch-all for cloud models (optional)
    ]

    def __new__(cls):
        """Singleton pattern for registry."""
        if cls._instance is None:
            logging.debug("REGISTRY: Creating new registry instance")
            cls._instance = super().__new__(cls)
            # Initialize instance dictionaries on first creation
            cls._instance._providers = {}
            cls._instance._initialized_providers = {}
            logging.debug(f"REGISTRY: Created instance {cls._instance}")
        return cls._instance

    @classmethod
    def register_provider(cls, provider_type: ProviderType, provider_class: type[ModelProvider]) -> None:
        """Register a new provider class.

        Args:
            provider_type: Type of the provider (e.g., ProviderType.GOOGLE)
            provider_class: Class that implements ModelProvider interface
        """
        instance = cls()
        instance._providers[provider_type] = provider_class

    @classmethod
    def get_provider(cls, provider_type: ProviderType, force_new: bool = False) -> Optional[ModelProvider]:
        """Get an initialized provider instance.

        Args:
            provider_type: Type of provider to get
            force_new: Force creation of new instance instead of using cached

        Returns:
            Initialized ModelProvider instance or None if not available
        """
        instance = cls()

        # Return cached instance if available and not forcing new
        if not force_new and provider_type in instance._initialized_providers:
            return instance._initialized_providers[provider_type]

        # Check if provider class is registered
        if provider_type not in instance._providers:
            return None

        # Get API key from environment
        api_key = cls._get_api_key_for_provider(provider_type)

        # Get provider class or factory function
        provider_class = instance._providers[provider_type]

        # Defensive: transient init protection
        try:
            # For custom providers, handle special initialization requirements
            if provider_type == ProviderType.CUSTOM:
                # Check if it's a factory function (callable but not a class)
                if callable(provider_class) and not isinstance(provider_class, type):
                    # Factory function - call it with api_key parameter
                    provider = provider_class(api_key=api_key)
                else:
                    # Regular class - need to handle URL requirement
                    custom_url = os.getenv("CUSTOM_API_URL", "")
                    if not custom_url:
                        if api_key:  # Key is set but URL is missing
                            logging.warning("CUSTOM_API_KEY set but CUSTOM_API_URL missing – skipping Custom provider")
                        return None
                    # Use empty string as API key for custom providers that don't need auth (e.g., Ollama)
                    # This allows the provider to be created even without CUSTOM_API_KEY being set
                    api_key = api_key or ""
                    # Initialize custom provider with both API key and base URL
                    provider = provider_class(api_key=api_key, base_url=custom_url)
            else:
                if not api_key:
                    return None
                # Initialize non-custom provider with API key and optional base_url from env for specific providers
                if provider_type in (ProviderType.KIMI, ProviderType.GLM):
                    base_url_env = {
                        ProviderType.KIMI: "KIMI_API_URL",
                        ProviderType.GLM: "GLM_API_URL",
                    }[provider_type]
                    base_url = os.getenv(base_url_env)
                    if base_url:
                        provider = provider_class(api_key=api_key, base_url=base_url)
                    else:
                        provider = provider_class(api_key=api_key)
                else:
                    provider = provider_class(api_key=api_key)
        except Exception as e:
            # Failed to create provider (transient or misconfig). Do not raise here.
            logging.warning("Provider %s initialization failed: %s", provider_type, e)
            return None

        # Wrap with health if enabled
        if _health_enabled():
            provider = HealthWrappedProvider(provider)

        # Cache the instance
        instance._initialized_providers[provider_type] = provider

        return provider

    @classmethod
    def get_provider_for_model(cls, model_name: str) -> Optional[ModelProvider]:
        """Get provider instance for a specific model name.

        Provider priority order:
        1. Native APIs (GOOGLE, OPENAI) - Most direct and efficient
        2. CUSTOM - For local/private models with specific endpoints
        3. OPENROUTER - Catch-all for cloud models via unified API

        Args:
            model_name: Name of the model (e.g., "gemini-2.5-flash", "gpt5")

        Returns:
            ModelProvider instance that supports this model
        """
        logging.debug(f"get_provider_for_model called with model_name='{model_name}'")

        # Check providers in priority order
        instance = cls()
        logging.debug(f"Registry instance: {instance}")
        logging.debug(f"Available providers in registry: {list(instance._providers.keys())}")

        for provider_type in cls.PROVIDER_PRIORITY_ORDER:
            if provider_type in instance._providers:
                logging.debug(f"Found {provider_type} in registry")

                # Health gating: skip if circuit is OPEN (only when enabled and not log-only)
                if _health_enabled() and _cb_enabled():
                    health = _get_health_manager().get(provider_type.value)
                    if health.breaker.state == CircuitState.OPEN:
                        logging.warning("Skipping provider %s due to OPEN circuit", provider_type)
                        continue

                # Get or create provider instance
                provider = cls.get_provider(provider_type)
                if provider and provider.validate_model_name(model_name):
                    logging.debug(f"{provider_type} validates model {model_name}")
                    return provider
                else:
                    logging.debug(f"{provider_type} does not validate model {model_name}")
            else:
                logging.debug(f"{provider_type} not found in registry")

        logging.debug(f"No provider found for model {model_name}")
        return None

    @classmethod
    def get_available_providers(cls) -> list[ProviderType]:
        """Get list of registered provider types."""
        instance = cls()
        return list(instance._providers.keys())

    @classmethod
    def get_available_models(cls, respect_restrictions: bool = True) -> dict[str, ProviderType]:
        """Get mapping of all available models to their providers.

        Args:
            respect_restrictions: If True, filter out models not allowed by restrictions

        Returns:
            Dict mapping model names to provider types
        """
        # Import here to avoid circular imports
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service() if respect_restrictions else None
        models: dict[str, ProviderType] = {}
        instance = cls()

        for provider_type in instance._providers:
            provider = cls.get_provider(provider_type)
            if not provider:
                continue

            try:
                available = provider.list_models(respect_restrictions=respect_restrictions)
            except NotImplementedError:
                logging.warning("Provider %s does not implement list_models", provider_type)
                continue

            for model_name in available:
                # =====================================================================================
                # CRITICAL: Prevent double restriction filtering (Fixed Issue #98)
                # =====================================================================================
                # Previously, both the provider AND registry applied restrictions, causing
                # double-filtering that resulted in "no models available" errors.
                #
                # Logic: If respect_restrictions=True, provider already filtered models,
                # so registry should NOT filter them again.
                # TEST COVERAGE: tests/test_provider_routing_bugs.py::TestOpenRouterAliasRestrictions
                # =====================================================================================
                if (
                    restriction_service
                    and not respect_restrictions  # Only filter if provider didn't already filter
                    and not restriction_service.is_allowed(provider_type, model_name)
                ):
                    logging.debug("Model %s filtered by restrictions", model_name)
                    continue
                models[model_name] = provider_type

        return models

    @classmethod
    def get_available_model_names(cls, provider_type: Optional[ProviderType] = None) -> list[str]:
        """Get list of available model names, optionally filtered by provider.

        This respects model restrictions automatically.

        Args:
            provider_type: Optional provider to filter by

        Returns:
            List of available model names
        """
        available_models = cls.get_available_models(respect_restrictions=True)

        if provider_type:
            # Filter by specific provider
            return [name for name, ptype in available_models.items() if ptype == provider_type]
        else:
            # Return all available models
            return list(available_models.keys())

    @classmethod
    def _get_api_key_for_provider(cls, provider_type: ProviderType) -> Optional[str]:
        """Get API key for a provider from environment variables.

        Args:
            provider_type: Provider type to get API key for

        Returns:
            API key string or None if not found
        """
        key_mapping = {
            ProviderType.GOOGLE: "GEMINI_API_KEY",
            ProviderType.OPENAI: "OPENAI_API_KEY",
            ProviderType.XAI: "XAI_API_KEY",
            ProviderType.KIMI: "KIMI_API_KEY",
            ProviderType.GLM: "GLM_API_KEY",
            ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
            ProviderType.CUSTOM: "CUSTOM_API_KEY",  # Can be empty for providers that don't need auth
            ProviderType.DIAL: "DIAL_API_KEY",
        }

        env_var = key_mapping.get(provider_type)
        if not env_var:
            return None

        return os.getenv(env_var)

    @classmethod
    def _get_allowed_models_for_provider(cls, provider: ModelProvider, provider_type: ProviderType) -> list[str]:
        """Get a list of allowed canonical model names for a given provider.

        Args:
            provider: The provider instance to get models for
            provider_type: The provider type for restriction checking

        Returns:
            List of model names that are both supported and allowed
        """
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()

        allowed_models = []

        # Get the provider's supported models
        try:
            # Use list_models to get all supported models (handles both regular and custom providers)
            supported_models = provider.list_models(respect_restrictions=False)
        except (NotImplementedError, AttributeError):
            # Fallback to SUPPORTED_MODELS if list_models not implemented
            try:
                supported_models = list(provider.SUPPORTED_MODELS.keys())
            except AttributeError:
                supported_models = []

        # Filter by restrictions
        for model_name in supported_models:
            try:
                if restriction_service.is_allowed(provider_type, model_name, model_name):
                    allowed_models.append(model_name)
            except Exception:
                # Be permissive on errors; better to include than exclude due to diagnostics
                allowed_models.append(model_name)

        return allowed_models

    @classmethod
    def _auggie_fallback_chain(cls, category: Optional["ToolModelCategory"], hints: Optional[list[str]] = None) -> list[str]:
        """Return a list of models in desired fallback order for the given category.
        Sources:
          - Auggie config 'fallback' mapping if present
          - Otherwise, reuse get_preferred_fallback_model heuristics to seed order
        """
        try:
            from auggie.config import get_auggie_settings
            settings = get_auggie_settings() or {}
            fb = settings.get("fallback") or {}
            key = None
            if category:
                key = {
                    # Map existing categories to custom keys users may set
                    # FAST_RESPONSE -> chat, EXTENDED_REASONING -> reasoning
                    "FAST_RESPONSE": "chat",
                    "EXTENDED_REASONING": "reasoning",
                }.get(category.name, category.name.lower())
            chain = fb.get(key or "", [])
            if chain:
                return chain
        except Exception:
            pass
        # Default: attempt to derive at least one model then expand by provider order
        seed = cls.get_preferred_fallback_model(category)
        order = [seed] if seed else []
        # Add other candidates from providers in priority order
        instance = cls()
        for ptype in cls.PROVIDER_PRIORITY_ORDER:
            provider = cls.get_provider(ptype)
            if not provider:
                continue
            try:
                allowed = cls._get_allowed_models_for_provider(provider, ptype)
                for name in allowed:
                    if name not in order:
                        order.append(name)
            except Exception:
                continue
        # Simple context-aware biasing: if hints provided, move matching models earlier
        if hints:
            low = {m.lower(): m for m in chain}
            priorities = []
            # Keywords → model families
            for h in [s.lower() for s in hints if isinstance(s, str)]:
                if any(k in h for k in ("vision", "image", "diagram")):
                    for cand in ("glm-4.5v",):
                        m = low.get(cand)
                        if m and m not in priorities:
                            priorities.append(m)
                if any(k in h for k in ("think", "reason", "chain of thought", "cot", "deep")):
                    for cand in ("kimi-k2-thinking", "kimi-k2-0711-preview", "glm-4.5-airx"):
                        m = low.get(cand)
                        if m and m not in priorities:
                            priorities.append(m)
            # Reorder: prioritized models first, keep relative order for the rest
            rest = [m for m in chain if m not in priorities]
            chain = priorities + rest

        return order

    @classmethod
    def call_with_fallback(
        cls,
        category: Optional["ToolModelCategory"],
        call_fn,
        hints: Optional[list[str]] = None,
    ):
        """Execute a provider call with category-aware fallback and telemetry.
        call_fn receives the selected model name and must return a ModelResponse.
        """
        import time as _t
        chain = cls._auggie_fallback_chain(category, hints)
        last_exc = None
        for model in chain:
            t0 = _t.perf_counter()
            try:
                resp = call_fn(model)
                dt_ms = int((_t.perf_counter() - t0) * 1000)
                usage = getattr(resp, "usage", None) if resp is not None else None
                cls._telemetry_update(model, True, dt_ms, usage)
                return resp
            except Exception as e:
                dt_ms = int((_t.perf_counter() - t0) * 1000)
                cls._telemetry_update(model, False, dt_ms, None)
                last_exc = e
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError("No models available for fallback execution")


    @classmethod
    def get_preferred_fallback_model(cls, tool_category: Optional["ToolModelCategory"] = None) -> str:
        """Get the preferred fallback model based on provider priority and tool category.

        This method orchestrates model selection by:
        1. Getting allowed models for each provider (respecting restrictions)
        2. Asking providers for their preference from the allowed list
        3. Falling back to first available model if no preference given

        Args:
            tool_category: Optional category to influence model selection

        Returns:
            Model name string for fallback use
        """
        from tools.models import ToolModelCategory

        effective_category = tool_category or ToolModelCategory.BALANCED
        first_available_model = None

        # If free-tier preference is enabled, attempt cross-provider free-first selection
        try:
            if _free_tier_enabled():
                free_set = _free_model_list()
                free_all: list[str] = []
                # Gather all allowed models per provider and collect free ones
                for ptype in cls.PROVIDER_PRIORITY_ORDER:
                    prov = cls.get_provider(ptype)
                    if not prov:
                        continue
                    allowed = cls._get_allowed_models_for_provider(prov, ptype)
                    if not allowed:
                        continue
                    # Track first available while scanning
                    if not first_available_model:
                        first_available_model = sorted(allowed)[0]
                    for m in allowed:
                        if m.lower() in free_set and m not in free_all:
                            free_all.append(m)
                if free_all:
                    # Apply category heuristics to free-only candidates
                    from tools.models import ToolModelCategory as _Cat
                    preferred_order: list[str] = []
                    if effective_category == _Cat.FAST_RESPONSE:
                        preferred_order = [
                            "GLM-4-Flash-250414", "glm-4.5-flash", "GLM-4.5-Flash",
                            "GLM-4-Air-250414", "glm-4.5-air", "GLM-4.5-Air",
                        ]
                    elif effective_category == _Cat.EXTENDED_REASONING:
                        preferred_order = [
                            "GLM-4-5", "glm-4.5", "GLM-4.5-Flash", "glm-4.5-flash",
                        ]
                    else:  # BALANCED
                        preferred_order = [
                            "GLM-4-Air-250414", "glm-4.5-air", "GLM-4.5-Air",
                            "GLM-4-Flash-250414", "glm-4.5-flash", "GLM-4.5-Flash",
                        ]
                    free_lower = {m.lower(): m for m in free_all}
                    for cand in preferred_order:
                        if cand.lower() in free_lower:
                            return free_lower[cand.lower()]
                    # Fallback to any free candidate
                    return free_all[0]
        except Exception:
            pass

        # Ask each provider for their preference in priority order
        for provider_type in cls.PROVIDER_PRIORITY_ORDER:
            provider = cls.get_provider(provider_type)
            if provider:
                # 1. Registry filters the models first
                allowed_models = cls._get_allowed_models_for_provider(provider, provider_type)
                # 1b. Apply cost-aware + free-tier preference ordering if enabled (intra-provider)
                allowed_models = _apply_cost_aware(allowed_models, tool_category)
                allowed_models = _apply_free_first(allowed_models)

                if not allowed_models:
                    continue

                # 2. Keep track of the first available model as fallback
                if not first_available_model:
                    first_available_model = sorted(allowed_models)[0]

                # 3. Ask provider to pick from allowed list
                preferred_model = provider.get_preferred_model(effective_category, allowed_models)

                if preferred_model:
                    logging.debug(
                        f"Provider {provider_type.value} selected '{preferred_model}' for category '{effective_category.value}'"
                    )
                    return preferred_model

                # 3b. Category-based heuristics when no provider preference is defined
                try:
                    from tools.models import ToolModelCategory as _Cat
                    heuristics: dict[ProviderType, list[str]] = {}
                    if effective_category == _Cat.FAST_RESPONSE:
                        heuristics = {
                            ProviderType.GLM: ["glm-4.5-air", "glm-4.5-flash", "glm-4.5"],
                            ProviderType.KIMI: ["kimi-k2-turbo-preview", "kimi-k2-0711-preview"],
                        }
                    elif effective_category == _Cat.EXTENDED_REASONING:
                        heuristics = {
                            ProviderType.KIMI: ["kimi-k2-0711-preview", "kimi-k2-turbo-preview"],
                            ProviderType.GLM: ["glm-4.5", "glm-4.5-flash", "glm-4.5-air"],
                        }
                    else:  # BALANCED
                        heuristics = {
                            ProviderType.GLM: ["glm-4.5", "glm-4.5-flash", "glm-4.5-air"],
                            ProviderType.KIMI: ["kimi-k2-0711-preview", "kimi-k2-turbo-preview"],
                        }

                    desired = heuristics.get(provider_type, [])
                    if desired:
                        allowed_lower = {m.lower(): m for m in allowed_models}
                        for candidate in desired:
                            if candidate.lower() in allowed_lower:
                                return allowed_lower[candidate.lower()]
                except Exception:
                    pass

        # If no provider returned a preference, use first available model
        if first_available_model:
            logging.debug(f"No provider preference, using first available: {first_available_model}")
            return first_available_model

        # Ultimate fallback if no providers have models
        logging.warning("No models available from any provider, using default fallback")
        # Prefer GLM/Kimi defaults if keys exist; avoid unrelated providers
        try:
            if cls.get_provider(ProviderType.GLM):
                # Prefer free/fast default when possible
                return "glm-4.5-flash"
            if cls.get_provider(ProviderType.KIMI):
                return "kimi-k2-0711-preview"
        except Exception:
            pass
        # Final safe fallback within our supported stack
        return "glm-4.5-flash"

    @classmethod
    def get_available_providers_with_keys(cls) -> list[ProviderType]:
        """Get list of provider types that have valid API keys.

        Returns:
            List of ProviderType values for providers with valid API keys
        """
        available = []
        instance = cls()
        for provider_type in instance._providers:
            if cls.get_provider(provider_type) is not None:
                available.append(provider_type)
        return available

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached provider instances."""
        instance = cls()
        instance._initialized_providers.clear()

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the registry to a clean state for testing.

        This provides a safe, public API for tests to clean up registry state
        without directly manipulating private attributes.
        """
        cls._instance = None
        if hasattr(cls, "_providers"):
            cls._providers = {}

    @classmethod
    def _telemetry_update(cls, model: str, success: bool, latency_ms: int, usage: dict | None = None) -> None:
        """Update in-memory telemetry counters (thread-safe enough for single-process MCP)."""
        with cls._telemetry_lock:
            bucket = cls._telemetry.setdefault(model, {"success": 0, "failure": 0, "latency_ms": [], "input_tokens": 0, "output_tokens": 0})
            if success:
                bucket["success"] += 1
            else:
                bucket["failure"] += 1
            bucket["latency_ms"].append(int(latency_ms))
            if usage:
                bucket["input_tokens"] += int(usage.get("input_tokens", 0))
                bucket["output_tokens"] += int(usage.get("output_tokens", 0))

    @classmethod
    def get_telemetry(cls) -> dict:
        with cls._telemetry_lock:
            from utils.costs import estimate_cost
            # Add estimated_cost per model based on accumulated usage
            result = {}
            for model, data in cls._telemetry.items():
                usage_cost = estimate_cost(
                    model,
                    input_tokens=int(data.get("input_tokens", 0)),
                    output_tokens=int(data.get("output_tokens", 0)),
                )
                bucket = dict(data)
                bucket["estimated_cost_usd"] = usage_cost
                result[model] = bucket
            return result


    @classmethod
    def unregister_provider(cls, provider_type: ProviderType) -> None:
        """Unregister a provider (mainly for testing)."""
        instance = cls()
        instance._providers.pop(provider_type, None)
        instance._initialized_providers.pop(provider_type, None)
