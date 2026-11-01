#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_ARGS=()
if [ -f "$PROJECT_DIR/Quỳnh Robotics [13-10-2026 20_33]/thamden.mp4" ]; then
  VIDEO_ARGS=(--video "$PROJECT_DIR/Quỳnh Robotics [13-10-2026 20_33]/thamden.mp4")
fi

exec bash "$PROJECT_DIR/start.sh" \
  --mode ai \
  "${VIDEO_ARGS[@]}" \
  --model "$PROJECT_DIR/models/thamden_seg.pt" \
  --profile "$PROJECT_DIR/profiles/thamden_profile.json" \
  --device auto \
  --imgsz 416 \
  --conf 0.25 \
  "$@"
