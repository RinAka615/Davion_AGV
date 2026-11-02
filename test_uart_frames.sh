#!/usr/bin/env bash
# ==============================================================================
# Script Kiểm Tra Truyền Thông UART với STM32: Gửi giá trị 1-100 và 100-1
# (Đồng bộ chuẩn Frame 5-Byte của start_combined.sh / combined_app.py)
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Chọn môi trường Python (.venv-yolo nếu có sẵn)
if [ -f "$PROJECT_DIR/.venv-yolo/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv-yolo/bin/python"
elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
else
    echo "❌ Lỗi: Không tìm thấy Python trên hệ thống!"
    exit 1
fi

# 2. Đảm bảo quyền truy cập cổng serial (nếu tồn tại)
for port in /dev/ttySTM32 /dev/stm32 /dev/ttyUSB0 /dev/ttyACM0; do
    if [ -e "$port" ] && [ ! -w "$port" ]; then
        echo "ℹ️ Đang cấp quyền ghi cho cổng $port..."
        sudo chmod 666 "$port" 2>/dev/null || true
    fi
done

# 3. Khởi chạy script test UART với tất cả tham số truyền vào
exec "$PYTHON_EXE" "$PROJECT_DIR/test_uart_ramp.py" "$@"

