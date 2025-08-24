# Auggie CLI Terminal Defaults and Orchestrator Selection

This short guide shows two reliable ways to set GPT‑5 as the default Auggie orchestrator from your terminal without editing package files.

## Option A (PowerShell profile)
Add this line to your PowerShell profile and restart the terminal:

```
function auggie { & "$env:APPDATA\npm\auggie.ps1" --model gpt5 @args }
```

Then:
- auggie --print "test"  (uses GPT‑5)
- auggie --model sonnet4 ... (override to Claude Sonnet 4)

## Option B (Cross-shell wrapper cmd)
Create %USERPROFILE%\bin\auggie.cmd with:

```
@echo off
setlocal
set "AU=%APPDATA%\npm\auggie.ps1"
set "ALL=%*"
echo %ALL% | findstr /i /c:"--model" >nul
if %errorlevel%==0 (
  powershell -NoLogo -NoProfile -File "%AU%" %*
) else (
  powershell -NoLogo -NoProfile -File "%AU%" --model gpt5 %*
)
```

Add %USERPROFILE%\bin to PATH (System Properties → Environment Variables), reopen terminal. Now auggie uses GPT‑5 by default across shells.

## Notes
- Use --model gpt5 or --model sonnet4 explicitly to force a specific orchestrator
- The auggie-config.json in zen-mcp-server is not authoritative for the Auggie orchestrator; CLI flags/wrappers are
- The MCP server continues to use Kimi/GLM for tool execution per its own configuration

