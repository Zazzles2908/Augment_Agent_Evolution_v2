#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-containers/four-brain/triton/model_repository}

from=$ROOT/qwen3_embedding_trt
to=$ROOT/qwen3_4b_embedding
if [ -d "$from" ] && [ ! -d "$to" ]; then
  mv "$from" "$to"
  echo "Renamed $from -> $to"
fi

from=$ROOT/qwen3_reranker_trt
to=$ROOT/qwen3_0_6b_reranking
if [ -d "$from" ] && [ ! -d "$to" ]; then
  mv "$from" "$to"
  echo "Renamed $from -> $to"
fi

echo "Done. Ensure config.pbtxt name fields match these directory names."

