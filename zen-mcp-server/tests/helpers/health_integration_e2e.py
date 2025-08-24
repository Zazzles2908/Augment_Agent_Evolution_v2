# Moved from scripts/health_integration_e2e.py
# This helper is intended for manual/integration use. Pytest can import and run it selectively.
from __future__ import annotations
import asyncio
import os
from pathlib import Path

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import mcp.types as types

PROJECT_DIR = Path(__file__).resolve().parents[2]
PYTHON = str(PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe') if (PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe').exists() else 'python'
WRAPPER = str(PROJECT_DIR / 'mcp_server_wrapper.py')

async def run_chat_attempts(env_overrides: dict[str,str], model: str, attempts: int) -> list[str]:
    params = StdioServerParameters(
        command=PYTHON,
        args=[WRAPPER],
        cwd=str(PROJECT_DIR),
        env={
            'PYTHONPATH': str(PROJECT_DIR),
            'LOG_LEVEL': 'DEBUG',
            **env_overrides
        }
    )

    outputs: list[str] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for i in range(attempts):
                try:
                    res = await session.call_tool('chat', arguments={
                        'model': model,
                        'prompt': f'Attempt {i+1}: return exactly OK'
                    })
                    texts = [c.text for c in res.content if isinstance(c, types.TextContent)]
                    outputs.append(texts[0] if texts else '')
                except Exception as e:
                    outputs.append(f'ERROR: {e}')
    return outputs

async def main():
    envA = {
        'DEFAULT_MODEL': 'auto',
        'KIMI_API_KEY': 'INVALID_KEY',
        'GLM_API_KEY': os.getenv('GLM_API_KEY', ''),
        'HEALTH_CHECKS_ENABLED': 'true',
        'HEALTH_LOG_ONLY': 'true',
        'CIRCUIT_BREAKER_ENABLED': 'false',
        'RETRY_ATTEMPTS': '1'
    }
    outA = await run_chat_attempts(envA, model='kimi-k2-0711-preview', attempts=2)

    envB = {
        'DEFAULT_MODEL': 'auto',
        'KIMI_API_KEY': 'INVALID_KEY',
        'GLM_API_KEY': os.getenv('GLM_API_KEY', ''),
        'HEALTH_CHECKS_ENABLED': 'true',
        'HEALTH_LOG_ONLY': 'true',
        'CIRCUIT_BREAKER_ENABLED': 'true',
        'RETRY_ATTEMPTS': '1',
        'FAILURE_THRESHOLD': '3',
    }
    outB = await run_chat_attempts(envB, model='kimi-k2-0711-preview', attempts=4)

    print('HEALTH_E2E_SUMMARY_START')
    print({'phaseA': outA, 'phaseB': outB})
    print('HEALTH_E2E_SUMMARY_END')

if __name__ == '__main__':
    asyncio.run(main())

