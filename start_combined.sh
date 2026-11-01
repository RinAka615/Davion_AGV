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
if [ -z "$QT_QPA_PLATFORM" ]; then
    export QT_QPA_PLATFORM="wayland;xcb"
fi
export QT_LOGGING_RULES="*.warning=false"

# 2. Xử lý tham số truyền vào hoặc chọn video mẫu nhà màng
if [ $# -gt 0 ]; then
    echo "▶ Khởi chạy với file video chỉ định: $@"
    exec "$PYTHON_EXE" combined_app.py "$@"
else
    # Ưu tiên video quay lối đi nhà màng thực tế
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
        exec "$PYTHON_EXE" combined_app.py "$SAMPLE_VIDEO"
    else
        echo "▶ Mở giao diện ứng dụng..."
        exec "$PYTHON_EXE" combined_app.py
    fi
fi
