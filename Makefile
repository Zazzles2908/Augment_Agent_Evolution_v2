# Phase 1 Makefile (Ubuntu 24.04 / Docker host)
# Goals: Build TRT engines, start Triton, run smoke tests, validate with Zen at each step
# Safety: Dry-run supported via DRY_RUN=1. GPU/disk safety checks before heavy steps.

# -------- Config --------
MODEL_REPO ?= containers/four-brain/triton/model_repository
TRITON_HTTP ?= 8000
TRITON_METRICS ?= 8002
PRECISION ?= nvfp4
REDIS_URL ?= redis://localhost:6379
TRTEXEC ?= trtexec
PY ?= python3
ZEN ?= zen
CONVERT ?= $(PY) scripts/tensorrt/convert_model.py --repo $(MODEL_REPO) --precision $(PRECISION)
DRY_RUN ?= 1

# When DRY_RUN=1, all RUN commands become 'echo' (no execution)
ifeq ($(DRY_RUN),1)
  RUN := echo
  NOTE := "[DRY-RUN]"
else
  RUN :=
  NOTE :=
endif

# -------- Helpers --------
check_gpu := nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits || true
check_disk := df -h . | tail -1 || true

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make zen-audit-scripts         # Run Zen analysis on conversion/export scripts"
	@echo "  make build-embed               # Build qwen3_4b_embedding TRT engine"
	@echo "  make build-rerank              # Build qwen3_0_6b_reranking TRT engine"
	@echo "  make build-gen                 # Build glm45_air TRT engine"
	@echo "  make start-triton              # Start Triton (explicit mode)"
	@echo "  make zen-triton-load           # Trace server model loading with Zen"
	@echo "  make smoke-all                 # Run repo ops + embed/rerank/gen smoke tests"
	@echo "  make ingest                    # Run Docling ingestion with optional Redis cache"
	@echo "  make query Q='question'        # Run E2E query call"
	@echo "  make zen-report                # Generate Phase 1 Zen validation report"
	@echo "  make compose-up                # Start Triton+Redis+Prometheus+Grafana"
	@echo "  make compose-down              # Stop and remove compose stack"
	@echo "  make compose-logs              # Tail logs for a service (S=service)"
	@echo "  (Use DRY_RUN=0 to execute, DRY_RUN=1 to simulate)"

# -------- Zen audits --------
.PHONY: zen-audit-scripts
zen-audit-scripts:
	@echo $(NOTE) "Zen audit: scripts/tensorrt for precision and IO shapes"
	$(RUN) $(ZEN) analyze_zen --target scripts/tensorrt/ --context documents/stack --checks precision,shapes,hardcoded

# -------- Build engines --------
.PHONY: build-embed build-rerank build-gen
build-embed:
	@echo $(NOTE) "GPU memory before build:" && $(check_gpu)
	@echo $(NOTE) "Disk space:" && $(check_disk)
	@echo $(NOTE) "Build qwen3_4b_embedding"
	$(RUN) $(CONVERT) --model qwen3_4b_embedding --config containers/four-brain/triton/config/embed.yaml

build-rerank:
	@echo $(NOTE) "GPU memory before build:" && $(check_gpu)
	@echo $(NOTE) "Disk space:" && $(check_disk)
	@echo $(NOTE) "Build qwen3_0_6b_reranking"
	$(RUN) $(CONVERT) --model qwen3_0_6b_reranking --config containers/four-brain/triton/config/rerank.yaml

build-gen:
	@echo $(NOTE) "GPU memory before build:" && $(check_gpu)
	@echo $(NOTE) "Disk space:" && $(check_disk)
	@echo $(NOTE) "Build glm45_air"
	$(RUN) $(CONVERT) --model glm45_air --config containers/four-brain/triton/config/generate.yaml

# -------- Triton --------
.PHONY: start-triton zen-triton-load
start-triton:
	@echo $(NOTE) "Start Triton with explicit model control"
	$(RUN) tritonserver --model-repository=$(MODEL_REPO) --model-control-mode=explicit --http-port=$(TRITON_HTTP) --metrics-port=$(TRITON_METRICS)

zen-triton-load:
	@echo $(NOTE) "Trace Triton server load"
	$(RUN) $(ZEN) tracer_zen --command "tritonserver --model-repository=$(MODEL_REPO) --model-control-mode=explicit" --output traces/model_loading

# -------- Smoke tests --------
.PHONY: smoke-repo smoke-embed smoke-rerank smoke-gen smoke-all
smoke-repo:
	@echo $(NOTE) "Trace repo ops (load/unload)"
	$(RUN) $(ZEN) tracer_zen --command "bash scripts/smoke/triton_repository.sh" --output traces/model_ops

smoke-embed:
	@echo $(NOTE) "Trace embed infer"
	$(RUN) $(ZEN) tracer_zen --command "python3 scripts/smoke/embed_infer.py" --output traces/embed

smoke-rerank:
	@echo $(NOTE) "Trace rerank infer"
	$(RUN) $(ZEN) tracer_zen --command "python3 scripts/smoke/rerank_infer.py" --output traces/rerank

smoke-gen:
	@echo $(NOTE) "Trace generate infer"
	$(RUN) $(ZEN) tracer_zen --command "python3 scripts/smoke/generate_infer.py" --output traces/generate

smoke-all: smoke-repo smoke-embed smoke-rerank smoke-gen
	@echo $(NOTE) "Smoke suite complete"

# -------- Ingestion / Query --------
.PHONY: ingest query
ingest:
	@echo $(NOTE) "Run Docling ingestion with Redis caching if REDIS_URL is set"
	$(RUN) REDIS_URL=$(REDIS_URL) $(PY) services/ingestion/docling_ingest.py --input ./documents --batch-size 16
	@echo $(NOTE) "Zen analyze cache function"
	$(RUN) $(ZEN) analyze_zen --function cache_embedding --target services/ingestion/docling_ingest.py --context documents/stack

query:
	@echo $(NOTE) "Run E2E query"
	$(RUN) REDIS_URL=$(REDIS_URL) $(PY) services/query/e2e_query_service.py "$(Q)"
	@echo $(NOTE) "Zen trace query service"
	$(RUN) $(ZEN) tracer_zen --command "python3 services/query/e2e_query_service.py '$(Q)'" --output traces/query

# -------- Zen report --------
.PHONY: zen-report
zen-report:
	@echo $(NOTE) "Generate Phase 1 validation report from traces"
	$(RUN) $(ZEN) docgen_zen --source traces/ --output documents/reports/phase1_validation_report.md --context documents/stack

# -------- Docker Compose --------
.PHONY: compose-up compose-down compose-logs
compose-up:
	@echo $(NOTE) "Starting compose stack"
	$(RUN) docker compose up -d
	@echo $(NOTE) "Prometheus at http://localhost:$${PROMETHEUS_PORT:-9090}"
	@echo $(NOTE) "Grafana at http://localhost:$${GRAFANA_PORT:-3000} (admin/admin)"
	@echo $(NOTE) "Triton metrics at http://localhost:$${TRITON_METRICS_PORT:-8002}/metrics"

compose-down:
	@echo $(NOTE) "Stopping compose stack"
	$(RUN) docker compose down -v

compose-logs:
	@echo $(NOTE) "Logs for $(S)"
	$(RUN) docker compose logs -f $(S)

# -------- Defaults --------
.PHONY: all
all: zen-audit-scripts build-embed build-rerank build-gen start-triton zen-triton-load smoke-all ingest query zen-report
	@echo $(NOTE) "Phase 1 flow complete"

