import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import mcp.types as types

PROJECT_DIR = Path(__file__).parent
PYTHON = str(PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe') if (PROJECT_DIR / '.zen_venv' / 'Scripts' / 'python.exe').exists() else 'python'
WRAPPER = str(PROJECT_DIR / 'mcp_server_wrapper.py')

async def run_chat(session: ClientSession, model: str, prompt: str):
    args = {
        'model': model,
        'prompt': prompt,
        'thinking_mode': 'minimal',
        'use_websearch': False,
    }
    res = await session.call_tool('chat', arguments=args)
    texts = [c.text for c in res.content if isinstance(c, types.TextContent)]
    print(f"CHAT[{model}]:", ("\n".join(texts))[:400])

async def run_thinkdeep(session: ClientSession, model: str):
    args = {
        'model': model,
        'step': 'Analyze pros/cons of using Kimi K2 Turbo for fast iterative coding tasks.',
        'step_number': 1,
        'total_steps': 1,
        'next_step_required': False,
        'findings': 'Initial exploration focusing on latency, coherence, and instruction following.'
    }
    res = await session.call_tool('thinkdeep', arguments=args)
    texts = [c.text for c in res.content if isinstance(c, types.TextContent)]
    print(f"THINKDEEP[{model}]:", ("\n".join(texts))[:500])

async def run_consensus(session: ClientSession):
    args = {
        'step': 'Evaluate whether Kimi K2 Turbo is preferable to K2 0711 for quick architectural brainstorming.',
        'step_number': 1,
        'total_steps': 2,
        'next_step_required': True,
        'findings': 'Independent analysis: we care about speed, relevance, and clarity over long-form chain-of-thought.',
        'models': [
            { 'model': 'kimi-k2-turbo-preview', 'stance': 'for' },
            { 'model': 'kimi-k2-0711-preview', 'stance': 'against' }
        ]
    }
    args['model']='kimi-k2-0711-preview'
    res = await session.call_tool('consensus', arguments=args)
    texts = [c.text for c in res.content if isinstance(c, types.TextContent)]
    print("CONSENSUS(step1):", ("\n".join(texts))[:600])

async def main():
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
        }
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print('Initialized server:', init.serverInfo.name, init.serverInfo.version)
            tools = await session.list_tools()
            print('TOOLS AVAILABLE:', sorted([t.name for t in tools.tools]))

            await run_chat(session, 'kimi-k2-0711-preview', 'Summarize how you differ from K2 Turbo in 2 lines.')
            await run_chat(session, 'kimi-k2-turbo-preview', 'Summarize how you differ from K2 0711 in 2 lines.')
            await run_thinkdeep(session, 'kimi-k2-turbo-preview')
            await run_consensus(session)

if __name__ == '__main__':
    asyncio.run(main())

