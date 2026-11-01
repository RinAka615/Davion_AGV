#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_ARGS=()
if [ -f "$PROJECT_DIR/nhamang_20261025_030235.mp4" ]; then
  VIDEO_ARGS=(--video "$PROJECT_DIR/nhamang_20261025_030235.mp4")
fi

exec bash "$PROJECT_DIR/start.sh" \
  --mode ai \
  "${VIDEO_ARGS[@]}" \
  --model "$PROJECT_DIR/models/batden_fast_256.pt" \
  --profile "$PROJECT_DIR/profiles/batden_fast_profile.json" \
  --device auto \
  --imgsz 256 \
  --conf 0.25 \
  "$@"
