# Docker and Git Hygiene (Clean Slate)

Goal
- Keep the workspace clean: remove unused containers/images/volumes, avoid committing large artifacts, and prepare a clean branch for the stack reset.

Docker cleanup (Ubuntu 24.04)
- Use the helper script (interactive):
  - chmod +x scripts/env/cleanup_docker.sh
  - ./scripts/env/cleanup_docker.sh
- Manual commands (equivalent):
  - docker system prune -af --volumes
  - docker ps -a --format '{{.ID}}\t{{.Names}}' | grep -Ei 'four-brain|orchestrator|embedding|reranker|document|triton' | awk '{print $1}' | xargs -r docker rm -f

Git hygiene
- Ensure identity (local repo only):
  - git config user.name "Jazeel"
  - git config user.email "jajireen1@gmail.com"
- Create the reset branch and commit changes:
  - git checkout -b stack-reset-2025-08-21
  - git add -A
  - git commit -m "Stack reset: remove HRM; CUDA13/TRT/Triton updates; Ubuntu24.04 setup+validation; docker-compose cleaned; .gitignore hardened; Dockerfiles updated (2025-08-21)"
- Push and open PR to main (recommended):
  - git push origin stack-reset-2025-08-21
- After review: merge PR to main

Model repository hygiene
- Keep only config.pbtxt in git; do NOT commit *.onnx/*.plan engines.
- Current canonical model names in code/scripts:
  - qwen3_embedding_trt (embedding)
  - qwen3_reranker_trt (reranker)
  - glm45_air (generation)
- Remove old HRM directories under containers/four-brain/triton/model_repository/* if present.

Supabase CLI & Postgres
- Install Supabase CLI v2.x as per 08_ubuntu_24_04_clean_setup.md
- Start and validate local stack as needed

Notes
- If Docker Desktop is used on Windows/macOS, engine version should map to a recent 26/27.x. For Linux, ensure Docker Engine 27.x and NVIDIA Container Toolkit >= 1.17.8 (see setup doc).

