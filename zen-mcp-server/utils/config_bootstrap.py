"""
Config bootstrap and validation for zen MCP server (opt-in via ENABLE_CONFIG_VALIDATOR).

This module centralizes light-weight validation of environment-derived settings and
emits a one-line startup summary to stderr logs. It never writes to stdout.
"""
from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

PROVIDER_KEYS = [
    "KIMI_API_KEY",
    "GLM_API_KEY",
    "OPENROUTER_API_KEY",
]

@dataclass
class ServerConfig:
    present_keys: List[str] = field(default_factory=list)
    default_model: str = "auto"
    locale: str = ""
    custom_api_url: str = ""

    @classmethod
    def load_and_validate(cls) -> "ServerConfig":
        cfg = cls()
        cfg.default_model = os.getenv("DEFAULT_MODEL", "auto")
        cfg.locale = os.getenv("LOCALE", "")
        cfg.custom_api_url = os.getenv("CUSTOM_API_URL", "")
        cfg.present_keys = [k for k in PROVIDER_KEYS if os.getenv(k)]

        # Validate: at least one provider or CUSTOM_API_URL
        if not cfg.present_keys and not cfg.custom_api_url:
            logger.warning(
                "CONFIG VALIDATION: No API keys or CUSTOM_API_URL found. Server will start but providers may be unavailable."
            )
        # Summarize once to stderr
        keys_display = ",".join(cfg.present_keys) if cfg.present_keys else "none"
        logger.info(
            f"CONFIG SUMMARY: DEFAULT_MODEL={cfg.default_model} LOCALE={cfg.locale or 'unset'} PROVIDERS={keys_display} CUSTOM_API_URL={'set' if cfg.custom_api_url else 'unset'}"
        )
        return cfg

