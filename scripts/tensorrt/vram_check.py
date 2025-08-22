#!/usr/bin/env python3
import subprocess, re, sys

def get_used_total_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).decode()
        used, total = out.strip().split("\n")[0].split(",")
        return int(used), int(total)
    except Exception:
        return None, None

if __name__ == "__main__":
    used, total = get_used_total_mb()
    if used is None:
        print("VRAM check skipped (nvidia-smi not available)")
        sys.exit(0)
    print(f"VRAM used: {used} MB / {total} MB")
    if total < 16000:
        print("Warning: GPU total memory < 16GB")
    if used > 15000:
        print("Error: VRAM usage exceeds 15GB threshold during build/run")
        sys.exit(1)
    sys.exit(0)

