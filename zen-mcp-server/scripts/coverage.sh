#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$ROOT_DIR/.zen_venv/Scripts/python.exe"
if [ ! -f "$VENV_PY" ]; then
  VENV_PY="python"
fi
cd "$ROOT_DIR"
"$VENV_PY" -m pytest -q --maxfail=1 --disable-warnings --cov=zen-mcp-server --cov-report=term-missing "$@"

