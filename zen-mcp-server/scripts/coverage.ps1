$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Join-Path $ScriptDir '..'
$VenvPy = Join-Path $RootDir '.zen_venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { $VenvPy = 'python' }
Set-Location $RootDir
& $VenvPy -m pytest -q --maxfail=1 --disable-warnings --cov=zen-mcp-server --cov-report=term-missing @args

