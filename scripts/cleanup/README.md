# Cleanup Utilities

This folder provides robust cleanup utilities tailored for Windows environments.

## Remove legacy items
- Python (robust): `python scripts/cleanup/remove_legacy.py`
- PowerShell: `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup/remove_legacy.ps1`

`remove_legacy.py` has been upgraded to handle Windows long paths and read-only files.

## Force remove hrm_phase0
If `hrm_phase0` resists deletion (e.g., wandb `latest-run` reparse point, long paths, read-only files), use one of:

- PowerShell (recommended):
  - `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup/force_remove_hrm_phase0.ps1`

- Python (portable):
  - `python scripts/cleanup/force_remove_hrm_phase0.py`

Both scripts:
- Clear read-only/hidden/system attributes
- Remove reparse points (junctions/symlinks)
- Handle long paths with `\\?\` prefix
- Fall back to `robocopy` mirror-to-empty trick

If deletion still fails, close apps that might lock files (e.g., editors, W&B) and retry.

