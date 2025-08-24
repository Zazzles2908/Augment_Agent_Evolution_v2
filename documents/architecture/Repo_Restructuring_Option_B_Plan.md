# Repository Restructuring Plan (Option B: Clean main-new + PR)

This plan describes the safer restructuring workflow to establish a clean main branch without rewriting history on the remote immediately.

## Goals
- Preserve current remote history (no force-push)
- Introduce a clean branch (main-new) with the curated content
- Review via PR, then switch default branch to main-new
- After adoption, optionally archive the old main

## Preconditions
- Local repo is in good state: all tests pass, auggie CLI smoke passes
- Backup branch created and pushed: backup/pre-reshape

## Steps
1) Backup
   - git checkout -b backup/pre-reshape
   - git push origin backup/pre-reshape
   - Optional: zip archive of workspace

2) Create clean branch locally
   - git checkout --orphan main-new
   - git rm -r --cached .
   - Clean working tree to the curated structure (retain zen-mcp-server, documents, etc.)
   - git add . && git commit -m "chore(repo): initialize clean main-new"

3) Push and open PR
   - git push -u origin main-new
   - Open PR from main-new to main with description

4) Validation gates (CI and manual)
   - Auggie CLI:
     - auggie --mcp-config <path> --print "listmodels"
     - auggie --mcp-config <path> --print "version"
     - DIAGNOSTICS=true auggie --mcp-config <path> --print "self-check"
     - auggie --model gpt5 --mcp-config <path> --print "chat with glm-4.5-flash: ok"
   - Coverage:
     - ./zen-mcp-server/scripts/coverage.ps1

5) Switch default branch
   - In GitHub settings, set default to main-new
   - Merge PR or fast-forward main to main-new as desired
   - Optionally rename main-new to main and archive old main (tag or branch name archive/pre-clean)

## Notes
- This avoids destructive remote operations
- Collaborators will update their local clones after default switch
- Use DIAGNOSTICS=true when running self-check to keep the public tool list at 16

