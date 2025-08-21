#!/usr/bin/env python3
"""
Remove legacy/junk directories and backups (approved by maintainer).
Run from repo root:
  python scripts/cleanup/remove_legacy.py

This version is more robust on Windows:
 - Clears read-only attributes during deletion
 - Handles long paths using the extended prefix (\\\\?\\)
 - Avoids following directory symlinks/junctions
"""
import os
import shutil
import stat
import sys
from pathlib import Path


def _to_long_path(p: Path) -> str:
    # Enable long path handling on Windows
    s = str(p.resolve())
    if os.name == 'nt':
        if s.startswith('\\\\?\\'):
            return s
        # Prepend extended-length path prefix
        return r"\\\\?\\" + s
    return s


def _on_rm_error(func, path, exc_info):
    # When rmtree fails (e.g., due to read-only), try to make writable and retry
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except Exception:
        # If it's a symlink, unlink it instead of descending
        try:
            if os.path.islink(path):
                os.unlink(path)
                return
        except Exception:
            pass
        raise


def robust_remove_tree(path: Path):
    """Remove directory tree robustly across platforms."""
    if not path.exists():
        return
    # If it's a symlinked dir or file, just unlink
    if path.is_symlink():
        path.unlink()
        return
    # Pre-chmod files to writable to avoid EACCES on Windows
    for root, dirs, files in os.walk(path, topdown=False):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                os.chmod(fpath, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass
        for dname in dirs:
            dpath = os.path.join(root, dname)
            try:
                # Do not follow dir symlinks; unlink them
                if os.path.islink(dpath):
                    os.unlink(dpath)
                else:
                    os.chmod(dpath, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass
    # Use long path string on Windows
    target = _to_long_path(path)
    shutil.rmtree(target, onerror=_on_rm_error)


targets = [
    'docker_config_backup',
    'hrm_config_backup',
    'hrm_phase0',
    'zen-mcp-server',
    'model_backup_20250819_152158.tar.gz',
    'model_backup_20250819_152624.tar.gz',
    'qwen3_embedding_trt_backup.plan',
]

results = []
root = Path.cwd()

for p in targets:
    path = root / p
    if path.exists():
        try:
            if path.is_dir():
                print(f"Removing directory: {p}")
                robust_remove_tree(path)
            else:
                print(f"Removing file: {p}")
                # Ensure file is writable then unlink
                try:
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                except Exception:
                    pass
                path.unlink()
            results.append((p, 'removed'))
        except Exception as e:
            print(f"FAILED to remove {p}: {e}")
            results.append((p, f'failed: {e}'))
    else:
        print(f"Skip (not found): {p}")
        results.append((p, 'not_found'))

print("\nSummary:")
for p, status in results:
    print(f" - {p}: {status}")

# Exit non-zero if any failed
if any('failed' in status for _, status in results):
    raise SystemExit(1)
