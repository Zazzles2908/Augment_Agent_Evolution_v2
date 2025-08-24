#!/usr/bin/env python3
"""
MCP Server Wrapper for auggie compatibility
This wrapper ensures environment variables are properly inherited and the server starts correctly
"""

import os
import sys
from pathlib import Path

# Robust autodiscovery: locate project root (zen-mcp-server), .zen_venv, and server.py from any cwd
cur = Path(__file__).resolve()
# Try typical structure: scripts/ is under project root
project_dir = cur.parent.parent if cur.parent.name == 'scripts' else cur.parent
# Fallback: search upwards for server.py
if not (project_dir / 'server.py').exists():
    p = project_dir
    while p != p.parent:
        if (p / 'server.py').exists():
            project_dir = p
            break
        p = p.parent

# Ensure logs dir exists
(project_dir / 'logs').mkdir(exist_ok=True)

# Prefer venv python if available
venv_py = project_dir / '.zen_venv' / 'Scripts' / 'python.exe'
if venv_py.exists():
    # Re-exec with venv Python to ensure correct packages
    if sys.executable != str(venv_py):
        os.execv(str(venv_py), [str(venv_py), __file__])

# Set working dir and sys.path
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))
# Set Python path for child imports
os.environ.setdefault('PYTHONPATH', str(project_dir))
# Minimal logging unless user overrides
os.environ.setdefault('LOG_LEVEL', 'ERROR')

# Import and run the server
try:
    from server import run
    if __name__ == "__main__":
        run()
except Exception as e:
    with open(project_dir / "logs" / "wrapper_error.log", "a", encoding="utf-8") as f:
        f.write(f"Wrapper error: {e}\n")
    sys.exit(1)
