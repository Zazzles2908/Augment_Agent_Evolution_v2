# GLM-4.5 Implementation Recommendations – 11_docker_and_git_hygiene.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Docker Cleanup & Reset (Commands)
```bash
# Preferred
scripts/env/cleanup_docker.sh
# Manual
docker system prune -af --volumes
```

## 2. Git Hygiene & Workflow
```bash
git config user.name "Jazeel"
git config user.email "jajireen1@gmail.com"
git checkout -b stack-reset-2025-08-21
git add -A && git commit -m "Stack reset + hygiene updates" && git push -u origin stack-reset-2025-08-21
```

## 3. Repository Hygiene (.gitignore/.dockerignore)
.gitignore additions:
```
*.onnx
*.plan
*.engine
*.trt
*.tmp
*.log
__pycache__/
```
.dockerignore essentials:
```
.git
__pycache__/
*.onnx
*.plan
```

## 4. Model Repo Guardrails (pre-commit)
Place in `.git/hooks/pre-commit`:
```bash
#!/bin/bash
staged_files=$(git diff --cached --name-only --diff-filter=A | grep -E '\\.(onnx|plan|engine)$')
if [ -n "$staged_files" ]; then
  echo "Refusing to commit model artifacts:" && echo "$staged_files" && exit 1
fi
```

## 5. Supabase CLI Validation
```bash
supabase --version
supabase status
supabase db reset && supabase start
```

