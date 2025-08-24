# GLM-4.5 Implementation Recommendations – Augment_Code_Master_Guidelines_v2.md

Generated via Zen MCP using GLM-4.5 based on the stack document.

## 1. Repo-wide Policies (Consent Gates, Risk Levels)
- High-impact: require explicit consent (migrations, prod changes, infra, secrets, spend, long jobs)
- Medium-impact: approval + rationale (API contracts, deps, security)
- Low-impact: auto (local fixes, tests, docs)
- Modes: Auto / Review-First / Budget-Guarded

## 2. Task Management & Planning
- Use Zen MCP modes: planner, analyze, refactor, precommit, codereview, debug, tracer, testgen, secaudit, docgen
- Maintain continuation threads; plan → actions → results traceability
- Tasklists must define scope, success criteria, and end conditions

## 3. Execution & Validation (Safe-by-default)
- Pipeline: lint/type-check → unit tests → build → smoke
- Explicit approval for data writes/deploys; report commands+outputs
- Failure format: `FAIL: <what> | exit=<code> | reason=<stderr>` then `NEXT: <recovery>`

## 4. Documentation Standards
- CHANGES.md: one-liners with rationale
- LIMITATIONS.md: known constraints/perf ceilings
- API/runbooks: update on behaviour/contract changes; keep documents/stack in sync

## 5. Multi-model Use
- Second opinions when confidence < high or stakes high (security/perf/arch)
- Structured validation with concrete examples; record perspectives
- Confidence scoring to gate extra validation

## 6. CI Checks & PR Templates
- Commit format: `type(scope): summary`; branches: feature/fix/docs/chore
- PR template: summary, risks, validations, before/after, docs updated
- Pre-commit hooks: lint/tests/build/smoke; CI enforces docs and change logs

