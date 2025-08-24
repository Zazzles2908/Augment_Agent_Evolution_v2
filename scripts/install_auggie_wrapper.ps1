$ErrorActionPreference = 'Stop'
$bin = Join-Path $env:USERPROFILE 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$cmd = Join-Path $bin 'auggie.cmd'
$lines = @(
  '@echo off',
  'setlocal',
  'set "AU=%APPDATA%\npm\auggie.ps1"',
  'set "ALL=%*"',
  'echo %ALL% ^| findstr /i /c:"--model" >nul',
  'if %errorlevel%==0 (',
  '  powershell -NoLogo -NoProfile -File "%AU%" %*',
  ') else (',
  '  powershell -NoLogo -NoProfile -File "%AU%" --model gpt5 %*',
  ')'
)
Set-Content -Path $cmd -Value $lines -Encoding ASCII
Write-Host "Wrapper created at $cmd"
# Prepend to PATH for this session only
$env:PATH = "$bin;$env:PATH"
Write-Host "Temporarily added $bin to PATH for this session. To persist, add it to System Environment Variables (PATH)."

