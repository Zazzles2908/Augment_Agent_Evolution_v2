#!/usr/bin/env bash
set -euo pipefail

red() { echo -e "\e[31m$*\e[0m"; }
green() { echo -e "\e[32m$*\e[0m"; }
yellow() { echo -e "\e[33m$*\e[0m"; }

info() { yellow "[INFO] $*"; }
confirm() { read -r -p "$1 [y/N]: " ans; [[ "$ans" =~ ^[Yy]$ ]]; }

info "This will remove stopped containers, unused images, networks, build cache, and dangling volumes."
info "It is SAFE for a clean slate, but will reclaim space and remove caches."

if confirm "Proceed with full Docker prune?"; then
  docker system prune -af --volumes
  green "Docker system prune complete."
else
  red "Aborted."
  exit 1
fi

# Optional: remove specific project containers by name pattern
if confirm "Remove containers/images related to four-brain project (by name filters)?"; then
  docker ps -a --format '{{.ID}}\t{{.Image}}\t{{.Names}}' | grep -Ei 'four-brain|orchestrator|embedding|reranker|document|triton' || true
  if confirm "Stop and remove matched containers?"; then
    docker ps -a --format '{{.ID}}\t{{.Names}}' | grep -Ei 'four-brain|orchestrator|embedding|reranker|document|triton' | awk '{print $1}' | xargs -r docker rm -f
    green "Project containers removed."
  fi
fi

green "Docker cleanup complete."

