# Contributing (Private Repository)

This is a private, canonical repository for a production-grade RAG stack.

## Workflow
- All changes go through Pull Requests into `main`.
- Require at least 1 approving review.
- Keep PRs small and focused; link to an issue when possible.

## Commit Style
- Prefer clear, imperative commit messages (e.g., `docs: update runbook`).
- Squash or merge commits per PR as preferred.

## Tests & Linting
- Add tests when practical. The CI runs `ruff` and `pytest` (if tests exist).

## Security
- Do not commit secrets. Use GitHub Secrets or local `.env`.
- Report vulnerabilities privately to the maintainer.

## License
- Internal/private use only unless the LICENSE is changed.

