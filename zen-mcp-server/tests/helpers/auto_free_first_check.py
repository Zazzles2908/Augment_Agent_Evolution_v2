#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
from pathlib import Path
import time

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import mcp.types as types

PROJECT_DIR = Path(__file__).resolve().parents[2]
PYTHON = str(PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe') if (PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe').exists() else 'python'
WRAPPER = str(PROJECT_DIR / 'mcp_server_wrapper.py')
LOG_FILE = PROJECT_DIR / 'logs' / 'mcp_server.log'

async def run_auto_chat(env_overrides: dict[str,str], prompt: str) -> dict:
    params = StdioServerParameters(
        command=PYTHON,
        args=[WRAPPER],
        cwd=str(PROJECT_DIR),
        env={
            'PYTHONPATH': str(PROJECT_DIR),
            'LOG_LEVEL': 'INFO',
            **env_overrides
        }
    )

    # Capture starting size to tail only new lines
    start_size = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Easy prompt should bias towards free-tier when enabled
            res = await session.call_tool('chat', arguments={
                'model': 'auto',
                'prompt': prompt,
                'thinking_mode': 'minimal',
                'use_websearch': False,
            })
            texts = [c.text for c in res.content if isinstance(c, types.TextContent)]

    # Read appended log content
    resolved_model = None
    if LOG_FILE.exists():
        with LOG_FILE.open('r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_size)
            for line in f.readlines():
                if 'Auto mode resolved to ' in line and 'for chat' in line:
                    # Example: "Auto mode resolved to glm-4.5-flash for chat (category: FAST_RESPONSE)"
                    try:
                        resolved_model = line.split('Auto mode resolved to ')[1].split(' for ')[0].strip()
                    except Exception:
                        pass
    return {
        'resolved_model': resolved_model,
        'preview': (texts[0][:160] if texts else ''),
    }

async def main():
    env = {
        'DEFAULT_MODEL': 'auto',
        'PROMETHEUS_ENABLED': 'true',
        'METRICS_PORT': os.getenv('METRICS_PORT', '9108'),
        # Self-healing on, observe logs; OPEN still gates with breaker enabled
        'HEALTH_CHECKS_ENABLED': 'true',
        'HEALTH_LOG_ONLY': 'true',
        'CIRCUIT_BREAKER_ENABLED': 'true',
        'HALF_OPEN_SEC': os.getenv('HALF_OPEN_SEC', '30'),
        # Free-first for GLM
        'FREE_TIER_PREFERENCE_ENABLED': 'true',
        'FREE_MODEL_LIST': 'glm-4.5-flash,glm-4.5-air',
    }
    out = await run_auto_chat(env, "Say 'MCP AUTO OK'.")
    print('AUTO_FREE_FIRST_SUMMARY_START')
    print(out)
    print('AUTO_FREE_FIRST_SUMMARY_END')

if __name__ == '__main__':
    asyncio.run(main())

