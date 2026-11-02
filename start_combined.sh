#!/usr/bin/env bash
# ==============================================================================
# Script khởi động Hệ Thống Tích Hợp: Dò Đường Nhà Màng & Quét Mã QR (PyQt5)
# (Autonomous Navigation Path Tracking & Real-time QR Code Dewarp/Decoder)
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================================="
echo "    Robot Nông Nghiệp: Hệ Thống Tích Hợp Dò Đường & Quét Mã QR (PyQt5)    "
echo "  (YOLOv8 Segmentation + Path Geometric Tracking + QR Size & Telemetry)   "
echo "=========================================================================="

# 1. Chọn môi trường Python tối ưu (Ưu tiên .venv-yolo vì có PyTorch CUDA & Ultralytics)
if [ -f "$PROJECT_DIR/.venv-yolo/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv-yolo/bin/python"
    PYQT5_PLUGINS="$PROJECT_DIR/.venv-yolo/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
    echo "▶ Sử dụng Python VirtualEnv: .venv-yolo (CUDA PyTorch & Ultralytics)"
elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv/bin/python"
    PYQT5_PLUGINS="$PROJECT_DIR/.venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
    echo "▶ Sử dụng Python VirtualEnv: .venv"
else
    PYTHON_EXE="python3"
    PYQT5_PLUGINS=""
    echo "▶ Sử dụng Python hệ thống: $(which python3)"
fi

# Tự động tìm thư mục plugin PyQt5 (hỗ trợ đa phiên bản Python 3.10, 3.14...)
PYQT5_PLUGINS=""
if [ -d "$PROJECT_DIR/.venv-yolo" ]; then
    PYQT5_PLUGINS=$(find "$PROJECT_DIR/.venv-yolo" -path "*/PyQt5/Qt5/plugins" 2>/dev/null | head -n 1 || true)
fi
if [ -z "$PYQT5_PLUGINS" ] || [ ! -d "$PYQT5_PLUGINS" ]; then
    for _p in /usr/lib/*-linux-gnu/qt5/plugins /usr/lib/qt5/plugins; do
        if [ -d "$_p" ]; then
            PYQT5_PLUGINS="$_p"
            break
        fi
    done
fi

if [ -n "$PYQT5_PLUGINS" ] && [ -d "$PYQT5_PLUGINS" ]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT5_PLUGINS"
fi
# Tự động tối ưu nền tảng đồ họa hiển thị trên Linux (Wayland / X11)
if [ "$XDG_SESSION_TYPE" = "wayland" ] && [ -z "$QT_QPA_PLATFORM" ]; then
    export QT_QPA_PLATFORM="wayland;xcb"
fi
export QT_LOGGING_RULES="*.warning=false"

# Cấu hình số luồng CPU an toàn nguồn Pi 5
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-2}

# 2. Xử lý tham số truyền vào hoặc chọn video mẫu nhà màng
ARGS=("$@")
HAS_VIDEO=false
HAS_UART_FLAG=false
for arg in "$@"; do
    if [ "$arg" = "--uart" ]; then
        HAS_UART_FLAG=true
    fi
    if [ -f "$arg" ] || [[ "$arg" =~ \.(mp4|avi|mkv|mov)$ ]]; then
        HAS_VIDEO=true
    fi
done

if [ "$HAS_UART_FLAG" = false ] && { [ -e "/dev/ttySTM32" ] || [ -e "/dev/ttyUSB0" ]; }; then
    echo "▶ Tự động phát hiện cổng STM32 -> Tự động bật truyền UART (--uart)"
    ARGS=(--uart "${ARGS[@]}")
fi

if [ "$HAS_VIDEO" = false ]; then
    SAMPLE_VIDEO="$PROJECT_DIR/nhamang_20261025_030235.mp4"
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        SAMPLE_VIDEO="$PROJECT_DIR/2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4"
    fi
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        SAMPLE_VIDEO="$PROJECT_DIR/test_qr_dataset/test_qr_multiscene.mp4"
    fi

    if [ -f "$SAMPLE_VIDEO" ]; then
        echo "▶ Tự động nạp video mẫu nhà màng: $(basename "$SAMPLE_VIDEO")"
        echo "  (Bạn có thể bấm '📂 Chọn Video...' trên giao diện để chọn bất kỳ video nào khác)"
        echo "  (Hoặc bấm '📷 Mở Webcam (Live Stream)' để truyền trực tiếp từ camera)"
        ARGS=("$SAMPLE_VIDEO" "${ARGS[@]}")
    fi
fi

echo "▶ Khởi chạy giao diện với tham số: ${ARGS[*]}"
exec "$PYTHON_EXE" combined_app.py "${ARGS[@]}"
