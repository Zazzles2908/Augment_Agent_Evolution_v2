import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import mcp.types as types

PROJECT_DIR = Path(__file__).parent
PYTHON = str(PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe') if (PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe').exists() else 'python'
WRAPPER = str(PROJECT_DIR / 'mcp_server_wrapper.py')

KIMI_MODELS = [
    'moonshot-v1-8k',
    'moonshot-v1-32k',
    'moonshot-v1-128k',
    'moonshot-v1-auto',
    'kimi-k2-0711-preview',
    'kimi-k2-turbo-preview',
    'kimi-k2-thinking',
]

async def run_once(model: str, prompt: str):
    params = StdioServerParameters(
        command=PYTHON,
        args=[WRAPPER],
        cwd=str(PROJECT_DIR),
        env={
            'PYTHONPATH': str(PROJECT_DIR),
            'LOG_LEVEL': os.getenv('LOG_LEVEL','INFO'),
            'KIMI_API_KEY': os.getenv('KIMI_API_KEY',''),
            'KIMI_API_URL': os.getenv('KIMI_API_URL','https://api.moonshot.ai/v1'),
            'GLM_API_KEY': os.getenv('GLM_API_KEY',''),
            'GLM_API_URL': os.getenv('GLM_API_URL','https://api.z.ai/api/paas/v4'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL','auto'),
            # Allow all Kimi variants for this evaluation
            'KIMI_ALLOWED_MODELS': ','.join(KIMI_MODELS),
        }
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            args = {
                'model': model,
                'prompt': prompt,
                'thinking_mode': 'minimal',
                'use_websearch': False,
            }
            t0 = time.perf_counter()
            result = await session.call_tool('chat', arguments=args)
            dt = (time.perf_counter() - t0) * 1000.0
            texts = []
            for c in result.content:
                if isinstance(c, types.TextContent):
                    texts.append(c.text)
            text_joined = '\n'.join(texts)
            print(f'MODEL={model} | latency_ms={dt:.0f} | ok={"MCP OK" in text_joined} | preview={text_joined[:160].replace("\n"," ")}')

async def main():
    for m in KIMI_MODELS:
        try:
            await run_once(m, f"Say 'MCP OK {m}'.")
        except Exception as e:
            print(f'MODEL={m} | ERROR: {e}')

if __name__ == '__main__':
    asyncio.run(main())
