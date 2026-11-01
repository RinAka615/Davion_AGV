#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Dừng ứng dụng RobotNongNghiep                          "
echo "=========================================================="

# Tìm PID của tiến trình chạy main.py
PIDS=$(pgrep -f "python.*main.py" || true)

if [ -z "$PIDS" ]; then
    echo "ℹ️ Không có tiến trình RobotNongNghiep nào đang chạy."
    exit 0
fi

echo "▶ Đang dừng tiến trình PID: $PIDS..."
kill $PIDS 2>/dev/null || true
sleep 0.5

# Kiểm tra nếu tiến trình vẫn còn thì dùng kill -9
REMAINING=$(pgrep -f "python.*main.py" || true)
if [ -n "$REMAINING" ]; then
    echo "⚠️ Thực hiện buộc dừng tiến trình (kill -9): $REMAINING..."
    kill -9 $REMAINING 2>/dev/null || true
fi

echo "✅ Đã dừng toàn bộ ứng dụng RobotNongNghiep thành công!"

