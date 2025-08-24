#!/usr/bin/env python3
"""
Zen MCP Server Diagnostic Script

Purpose: Systematically detect root causes for MCP startup failures and silent exits.
Safe to run: Does NOT start the server's stdio loop; only optional handshake test with timeout.

What it checks:
- File structure and permissions
- Python executable and sys.path
- Virtual environment status and key packages
- Environment variables and .env parsing
- Config files (auggie-config.json, mcp-config.json)
- Module imports (mcp.types, providers, server) and known pitfalls
- Optional MCP client handshake (ListTools) with short timeout

Outputs a clear PASS/FAIL summary per check with actionable hints.
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_PY = PROJECT_DIR / ".zen_venv" / "Scripts" / "python.exe"
LOGS_DIR = PROJECT_DIR / "logs"
WRAPPER = PROJECT_DIR / "mcp_server_wrapper.py"
ENV_FILE = PROJECT_DIR / ".env"
AUG_CONF = PROJECT_DIR / "auggie-config.json"
MCP_CONF = PROJECT_DIR / "mcp-config.json"

@dataclass
class CheckResult:
    name: str
    ok: bool
    info: str = ""
    hint: str = ""

results: list[CheckResult] = []

def add_result(name: str, ok: bool, info: str = "", hint: str = ""):
    results.append(CheckResult(name, ok, info.strip(), hint.strip()))


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_files():
    section("1) File structure and permissions")
    expected = [
        PROJECT_DIR / "server.py",
        PROJECT_DIR / "mcp_server_wrapper.py",
        PROJECT_DIR / "minimal_server.py",
        PROJECT_DIR / "providers",
        PROJECT_DIR / "tools",
        PROJECT_DIR / "auggie",
        PROJECT_DIR / "templates" / "auggie",
        PROJECT_DIR / "config.py",
        AUG_CONF,
        MCP_CONF,
        LOGS_DIR,
    ]
    all_ok = True
    for p in expected:
        exists = p.exists()
        readable = os.access(p, os.R_OK)
        writable = os.access(p, os.W_OK)
        typ = "dir" if p.is_dir() else "file"
        print(f"- {p.relative_to(PROJECT_DIR)} [{typ}] exists={exists} r={readable} w={writable}")
        if not exists:
            all_ok = False
    add_result("structure", all_ok, info="Checked core files and directories",
               hint="Create missing paths or fix permissions")


def check_python_and_sys_path():
    section("2) Python executable and sys.path")
    exe = sys.executable
    print(f"- Current Python: {exe}")
    print(f"- Version: {sys.version.replace('\n',' ')}")
    # Ensure project dir in sys.path
    in_path = str(PROJECT_DIR) in sys.path
    print(f"- PROJECT_DIR in sys.path: {in_path}")
    if not in_path:
        sys.path.insert(0, str(PROJECT_DIR))
        print("  -> Added PROJECT_DIR to sys.path for diagnostics")
    add_result("python_path", True, info=f"exe={exe}")


def check_virtualenv():
    section("3) Virtual environment status (.zen_venv)")
    exists = VENV_PY.exists()
    print(f"- VENV Python: {VENV_PY} exists={exists}")
    if exists:
        try:
            import subprocess
            out = subprocess.check_output([str(VENV_PY), "-c", "import sys;print(sys.version)"], cwd=PROJECT_DIR, timeout=15)
            ver = out.decode().strip()
            print(f"- VENV Python version: {ver}")
            # Package versions
            for mod in ["mcp", "google.genai", "openai", "pydantic", "dotenv"]:
                cmd = (
                    "import importlib,sys;\n"
                    "mod='" + mod + "'\n"
                    "try:\n m=importlib.import_module(mod)\n print(f'{mod} OK', getattr(m,'__version__',getattr(m,'VERSION',None)))\n"
                    "except Exception as e:\n print(f'{mod} FAIL {e.__class__.__name__}: {e}')\n"
                )
                out = subprocess.check_output([str(VENV_PY), "-c", cmd], cwd=PROJECT_DIR, timeout=20)
                print("  ", out.decode().strip())
            add_result("venv", True, info="VENV present and usable")
        except Exception as e:
            print(f"- Error invoking VENV Python: {e}")
            add_result("venv", False, hint="Recreate .zen_venv or fix permissions")
    else:
        add_result("venv", False, hint="Create venv and install deps (run run-server.ps1)")


def load_env():
    section("4) Environment variables (.env + process)")
    # Try to load .env with python-dotenv if available
    loaded = False
    if ENV_FILE.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(ENV_FILE))
            loaded = True
            print(f"- Loaded .env from {ENV_FILE}")
        except Exception as e:
            print(f"- Could not load .env via dotenv: {e} (will parse manually)")
            # Fallback manual parse
            try:
                for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
                loaded = True
                print("- Loaded .env manually")
            except Exception as e2:
                print(f"- Manual .env parse failed: {e2}")
    keys = [
        "DEFAULT_MODEL", "LOCALE", "LOG_LEVEL", "ENV_FILE",
        "KIMI_API_KEY", "GLM_API_KEY", "KIMI_API_URL", "GLM_API_URL",
        "OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY",
        "CUSTOM_API_URL", "CUSTOM_API_KEY", "CUSTOM_MODEL_NAME",
        "AUGGIE_CLI", "AUGGIE_CONFIG",
    ]
    for k in keys:
        val = os.getenv(k)
        present = val is not None and val != ""
        redacted = ("_KEY" in k) and present
        shown = "[SET]" if redacted else (val if present else "[MISSING]")
        print(f"- {k}: {shown}")
    add_result("env", loaded or True, info=".env processed and env inspected")


def check_configs():
    section("5) Configuration files parsing")
    ok_all = True
    # auggie-config.json
    if AUG_CONF.exists():
        try:
            aug = json.loads(AUG_CONF.read_text(encoding="utf-8"))
            has_root = isinstance(aug.get("auggie"), dict)
            print(f"- auggie-config.json parsed ok, has 'auggie' root: {has_root}")
            if not has_root:
                ok_all = False
        except Exception as e:
            ok_all = False
            print(f"- auggie-config.json parse error: {e}")
    else:
        print("- auggie-config.json missing")
        ok_all = False
    # mcp-config.json
    if MCP_CONF.exists():
        try:
            mcpj = json.loads(MCP_CONF.read_text(encoding="utf-8"))
            zen = (mcpj.get("mcpServers") or {}).get("zen")
            ok = isinstance(zen, dict)
            print(f"- mcp-config.json parsed ok, has mcpServers.zen: {ok}")
            if ok:
                cmd = zen.get("command"); args = zen.get("args") or []
                cwd = zen.get("cwd")
                print(f"  command: {cmd}\n  args: {args}\n  cwd: {cwd}")
                # Validate paths
                if cmd:
                    print(f"  command exists: {Path(cmd).exists()}")
                for a in args:
                    if isinstance(a, str) and a.endswith(".py"):
                        print(f"  arg path exists: {a} -> {Path(a).exists()}")
            else:
                ok_all = False
        except Exception as e:
            ok_all = False
            print(f"- mcp-config.json parse error: {e}")
    else:
        print("- mcp-config.json missing")
        ok_all = False
    add_result("configs", ok_all, hint="Fix JSON structure and paths if any failures above")


def check_imports_and_providers():
    section("6) Module imports (mcp, server, providers)")
    ok = True
    # mcp.types and ToolAnnotations presence
    try:
        import mcp.types as mtypes
        has_ann = hasattr(mtypes, "ToolAnnotations")
        print(f"- mcp.types imported, ToolAnnotations present: {has_ann}")
        if not has_ann:
            print("  NOTE: Server handles absence gracefully; ensure server uses fallback path.")
    except Exception as e:
        ok = False
        print(f"- mcp import failed: {e}")
    # google.genai vs google shadowing
    try:
        import google  # type: ignore
        g_loc = getattr(google, "__file__", None) or getattr(google, "__path__", None)
        print(f"- google module present at: {g_loc}")
        try:
            from google import genai  # type: ignore
            print("  google.genai imported OK")
        except Exception as e:
            print(f"  google.genai import FAILED: {e}")
            # Common cause: local 'google' package shadowing the namespace
            ok = False
    except Exception as e:
        print(f"- 'google' namespace import failed (ok if not using Gemini): {e}")
    # Import server module (no stdio start on import)
    try:
        import server as zen_server  # loads logging and registry
        print("- server.py imported OK")
        # Try provider configuration lightly
        try:
            zen_server.configure_providers()
            print("  providers configured OK")
        except Exception as e:
            ok = False
            print(f"  provider config FAILED: {e}")
    except Exception as e:
        ok = False
        print(f"- server import FAILED: {e}")
    add_result("imports", ok, hint="Resolve missing deps or shadowed packages (see above)")


def check_logs_tail():
    section("7) Existing logs (tail)")
    if LOGS_DIR.exists():
        for logname in ["wrapper_error.log", "mcp_server.log", "mcp_activity.log"]:
            p = LOGS_DIR / logname
            if p.exists():
                print(f"--- tail {logname} ---")
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for line in lines[-20:]:
                        print(line)
                except Exception as e:
                    print(f"(could not read {logname}: {e})")
            else:
                print(f"(no {logname})")
    else:
        print("(no logs dir)")
    add_result("logs", True)


async def optional_handshake_test():
    # Lightweight MCP handshake using python-mcp client, 15s timeouts to be Windows-friendly
    # Safe: spawns wrapper in a subprocess and requests list_tools
    try:
        import asyncio
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.session import ClientSession
        print("- Running MCP handshake test (5s timeout)...")
        try:
            async with stdio_client(StdioServerParameters(
                command=str(VENV_PY if VENV_PY.exists() else sys.executable),
                args=[str(WRAPPER)],
                cwd=str(PROJECT_DIR),
                env={**os.environ, "PYTHONPATH": str(PROJECT_DIR), "LOG_LEVEL": os.getenv("LOG_LEVEL", "ERROR")},
            )) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=15)
                    tools = await asyncio.wait_for(session.list_tools(), timeout=15)
                    print(f"  Handshake OK, tools: {len(tools.tools)}")
                    add_result("handshake", True, info=f"tools={len(tools.tools)}")
        except Exception as e:
            print(f"  Handshake FAILED: {e}")
            add_result("handshake", False, hint="Check wrapper_error.log and imports above")
    except Exception as e:
        print(f"- MCP client not available or failed: {e}")
        add_result("handshake", False, hint="Install 'mcp' in venv for handshake test")


def summarize():
    section("SUMMARY")
    ok = all(r.ok for r in results)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"- {r.name:12} {status} :: {r.info or ''} {(' | ' + r.hint) if r.hint and not r.ok else ''}")
    print("-" * 80)
    print(f"OVERALL: {'PASS' if ok else 'ATTENTION NEEDED'}")


def main():
    print(f"Zen MCP Diagnostics - {time.strftime('%Y-%m-%d %H:%M:%S')}\nProject: {PROJECT_DIR}")
    check_files()
    check_python_and_sys_path()
    check_virtualenv()
    load_env()
    check_configs()
    check_imports_and_providers()
    check_logs_tail()
    # Optional async handshake
    try:
        import asyncio
        # Give the server a bit more time on Windows to start (10s total)
        asyncio.run(optional_handshake_test())
    except Exception as e:
        print(f"- Optional handshake step skipped: {e}")
    summarize()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL: Diagnostic script crashed:\n" + traceback.format_exc())
        sys.exit(2)

