param(
  [string]$User = $env:USERNAME,
  [string]$TarName = 'TensorRT-10.13.2.6.Linux.x86_64-gnu.cuda-13.0.tar.gz'
)

$ErrorActionPreference = 'Stop'

Write-Host '(AugmentAI) TensorRT .tar installer for WSL Ubuntu 24.04 (CUDA 13)' -ForegroundColor Cyan
Write-Host "User: $User" -ForegroundColor Cyan
Write-Host "TarName: $TarName" -ForegroundColor Cyan

# 1) Locate tarball on Windows
$TarWinPath = "C:\\Users\\$User\\Downloads\\$TarName"
if (!(Test-Path $TarWinPath)) {
  Write-Host ("❌ Not found: " + $TarWinPath) -ForegroundColor Red
  Write-Host "Listing candidates in Downloads:" -ForegroundColor Yellow
  Get-ChildItem -Path ("C:\\Users\\" + $User + "\\Downloads") -Filter 'TensorRT-*.tar*' -File | Sort-Object LastWriteTime -Desc | Select-Object -First 10 Name,Length,LastWriteTime
  throw "Tarball not found; adjust -User or -TarName and retry."
}
Write-Host ("Found: " + $TarWinPath) -ForegroundColor Green

# 2) Convert to WSL path
$WslSrc = & wsl.exe wslpath -a "$TarWinPath"
if ([string]::IsNullOrWhiteSpace($WslSrc)) { throw 'wslpath failed to convert Windows path to WSL path' }
Write-Host ("WSL path: " + $WslSrc) -ForegroundColor Green

# 3) Stage into /opt/tensorrt and extract (run as root to avoid prompts)
& wsl.exe -u root -e mkdir -p /opt/tensorrt
& wsl.exe -u root -e cp "$WslSrc" /opt/tensorrt/
& wsl.exe -u root -e tar -xzf "/opt/tensorrt/$TarName" -C /opt/tensorrt

# 4) Detect extracted directory (directory only, not the tar file)
$TrtDir = (& wsl.exe -e bash -lc "find /opt/tensorrt -maxdepth 1 -type d -name 'TensorRT-*' | sort | tail -n 1").Trim()
if ([string]::IsNullOrWhiteSpace($TrtDir)) { throw 'Failed to detect extracted TensorRT directory under /opt/tensorrt' }
Write-Host ("Detected TRT_DIR: " + $TrtDir) -ForegroundColor Green

# 5) Install headers and libraries
& wsl.exe -u root -e mkdir -p "/usr/local/include/tensorrt"
& wsl.exe -u root -e cp -r "$TrtDir/include/." "/usr/local/include/tensorrt/"
& wsl.exe -u root -e cp -r "$TrtDir/lib/." "/usr/local/lib/"
& wsl.exe -u root -e bash -lc "echo /usr/local/lib > /etc/ld.so.conf.d/tensorrt.conf"
# Ensure ldconfig exists and run it
try {
  & wsl.exe -u root -e bash -lc "ldconfig"
} catch {
  Write-Host 'Installing libc-bin to provide ldconfig...' -ForegroundColor Yellow
  & wsl.exe -u root -e apt-get update
  & wsl.exe -u root -e apt-get install -y libc-bin
  & wsl.exe -u root -e bash -lc "ldconfig"
}

# 6) Install Python wheel (prefer wheel matching current Python version)
$PyTag = (& wsl.exe python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')").Trim()
$Wheel = (& wsl.exe bash -lc "ls -1 '$TrtDir'/python/tensorrt-*${PyTag}*.whl 2>/dev/null | head -n 1").Trim()
if ([string]::IsNullOrWhiteSpace($Wheel)) {
  Write-Host ("No matching wheel for ${PyTag} found. Available wheels:") -ForegroundColor Yellow
  & wsl.exe bash -lc "ls -1 '$TrtDir'/python/tensorrt-*.whl 2>/dev/null" | Out-Host
}

# Ensure python3 and venv/pip exist
try {
  & wsl.exe -e python3 --version | Out-Host
} catch {
  Write-Host 'Installing Python3 inside WSL...' -ForegroundColor Yellow
  & wsl.exe -u root -e apt-get update
  & wsl.exe -u root -e apt-get install -y python3
}
& wsl.exe -u root -e apt-get update
& wsl.exe -u root -e apt-get install -y python3-venv python3-pip

# Create venv and install
$VenvDir = "/opt/tensorrt/venv"
& wsl.exe -u root -e python3 -m venv "$VenvDir"
& wsl.exe -u root -e "$VenvDir/bin/pip" install --upgrade pip
if (-not [string]::IsNullOrWhiteSpace($Wheel)) {
  & wsl.exe -u root -e "$VenvDir/bin/pip" install "$Wheel"
} else {
  Write-Host 'Falling back to PyPI tensorrt==10.13.2 for current Python version' -ForegroundColor Yellow
  & wsl.exe -u root -e "$VenvDir/bin/pip" install "tensorrt==10.13.2"
}


# 7) Install trtexec and environment setup
& wsl.exe -u root -e install -m 0755 "$TrtDir/bin/trtexec" "/usr/local/bin/trtexec"
& wsl.exe -u root -e bash -lc "printf '%s\n' 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' > /etc/profile.d/tensorrt.sh"

# 8) Verification
Write-Host '(AugmentAI) Verifying trtexec...' -ForegroundColor Cyan
& wsl.exe -e trtexec --version | Out-Host

Write-Host '(AugmentAI) Verifying Python tensorrt import (venv)...' -ForegroundColor Cyan
& wsl.exe -e "/opt/tensorrt/venv/bin/python" -c "import tensorrt as trt; print('TensorRT:', trt.__version__); print('FP4 flag available:', hasattr(trt.BuilderFlag,'FP4'))" | Out-Host

Write-Host '(AugmentAI) Done.' -ForegroundColor Green

