#!/usr/bin/env python3
"""
Perform conservative string replacements to migrate legacy 'K2' naming to 'Orchestrator' in selected files.
Run from repo root:
  python scripts/cleanup/rename_k2_to_orchestrator.py

Only updates docstrings/comments/strings; avoids altering functional identifiers except in clearly safe contexts.
"""
from pathlib import Path

FILES = [
    'containers/four-brain/src/orchestrator_hub/task_router.py',
    'containers/four-brain/src/orchestrator_hub/strategy_planner.py',
    'containers/four-brain/nginx/nginx.conf',
    'containers/four-brain/src/orchestrator_hub/communication/redis_coordinator.py',
]

REPLACEMENTS = [
    ('K2-Vector-Hub', 'Orchestrator Hub'),
    ('K2 Vector Bridge', 'Orchestrator Bridge'),
    ('K2-Hub', 'Orchestrator Hub'),
    ('K2 Hub', 'Orchestrator Hub'),
    ('k2_vector_hub', 'orchestrator_hub'),
    ('k2_hub', 'orchestrator_hub'),
]

def apply(path: Path):
    text = path.read_text(encoding='utf-8')
    orig = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding='utf-8')
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")

def main():
    for rel in FILES:
        p = Path(rel)
        if p.exists():
            apply(p)
        else:
            print(f"Skip (not found): {rel}")

if __name__ == '__main__':
    main()

