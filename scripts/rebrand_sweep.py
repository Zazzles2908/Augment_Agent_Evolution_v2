#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

REPLACEMENTS = [
    # Order matters: handle longer phrases first
    (re.compile(r"\bAugment Code\b", re.IGNORECASE), "Zazzles's Agent"),
    (re.compile(r"\bAugment Agent Evolution\b", re.IGNORECASE), "Zazzles's Agent"),
    (re.compile(r"\bAugmentAI\b", re.IGNORECASE), "Zazzles's Agent"),
    (re.compile(r"\bAugment Agent\b", re.IGNORECASE), "Zazzles's Agent"),
    (re.compile(r"\bAugment\b", re.IGNORECASE), "Zazzles's Agent"),
    (re.compile(r"\bLocionne\b", re.IGNORECASE), "Zazzles's Agent"),
]

SKIP_DIRS = {
    ".git", ".venv", ".zen_venv", "venv", "node_modules", "engines", "models", "cache", ".pytest_cache", "dist", "build"
}

ALLOWED_EXT = {
    ".md", ".mdx", ".py", ".ps1", ".sh", ".txt", ".cfg", ".conf", ".ini", ".toml",
    ".json", ".yml", ".yaml", ".ts", ".tsx", ".js", ".jsx"
}

SKIP_FILES_EXACT = {
    "LICENSE", "LICENSE.txt", "NOTICE", "THIRD_PARTY_NOTICES", "THIRD_PARTY_NOTICES.txt"
}

# Conservative: don't touch file names, only file contents

def should_skip(path: Path) -> bool:
    parts = set(p.name for p in path.parents)
    if parts & SKIP_DIRS:
        return True
    if path.name in SKIP_FILES_EXACT:
        return True
    if path.suffix and path.suffix.lower() in ALLOWED_EXT:
        return False
    # skip unknown types
    return True


def sweep(root: Path, dry_run: bool = True):
    changed_files = 0
    total_changes = 0
    details = []

    for p in root.rglob('*'):
        # Skip early by path to avoid filesystem stat on inaccessible entries
        if should_skip(p):
            continue
        is_file = False
        try:
            is_file = p.is_file()
        except Exception:
            # Inaccessible (e.g., symlink or permission). Skip.
            continue
        if not is_file:
            continue
        try:
            data = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        original = data
        file_changes = 0
        for pat, repl in REPLACEMENTS:
            data, n = pat.subn(repl, data)
            file_changes += n
        if file_changes > 0:
            details.append((str(p.relative_to(root)), file_changes))
            changed_files += 1
            total_changes += file_changes
            if not dry_run:
                p.write_text(data, encoding='utf-8')

    return changed_files, total_changes, details


def main():
    ap = argparse.ArgumentParser(description='Rebranding sweep: Zazzles's Agent/Zazzles's Agent -> Zazzles\'s Agent')
    ap.add_argument('--apply', action='store_true', help='Apply changes (default: dry-run)')
    ap.add_argument('--root', default='.', help='Root directory (default: .)')
    ap.add_argument('--max-print', type=int, default=50, help='Max files to list in summary')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dry = not args.apply
    changed_files, total_changes, details = sweep(root, dry_run=dry)

    print(f"Mode: {'DRY-RUN' if dry else 'APPLY'}")
    print(f"Root: {root}")
    print(f"Files changed: {changed_files}")
    print(f"Total replacements: {total_changes}")
    print("---")
    for i, (rel, n) in enumerate(details):
        if i >= args.max_print:
            print(f"... ({len(details)-args.max_print} more files)")
            break
        print(f"{rel}: {n}")

if __name__ == '__main__':
    main()

