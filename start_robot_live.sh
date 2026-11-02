#!/usr/bin/env bash
# ==============================================================================
# Script khởi động trực tiếp Camera Orbbec Astra Pro + UART STM32 trên Robot
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================================="
echo "  Robot Nông Nghiệp: Khởi chạy Trực tiếp Camera Orbbec & UART STM32       "
echo "=========================================================================="

exec bash "$PROJECT_DIR/start.sh" \
  --cam \
  --uart \
  --model "$PROJECT_DIR/models/nhamang_batden_seg.pt" \
  --profile "$PROJECT_DIR/profiles/nhamang_batden_profile.json" \
  --device cuda \
  "$@"

