#!/usr/bin/env bash
# ==============================================================================
# Script khởi động Giao diện Đơn giản: Quét & Đọc mã QR (Có thanh trượt Độ tin cậy)
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "    Hệ Thống Quét & Đọc Mã QR - Giao Diện Đồ Họa Đơn Giản "
echo "    (Tùy chỉnh Độ tin cậy / Confidence Threshold & Dewarp) "
echo "=========================================================="

# 1. Chọn môi trường Python tối ưu (.venv hoặc .venv-yolo)
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv/bin/python"
    PYQT5_PLUGINS="$PROJECT_DIR/.venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
elif [ -f "$PROJECT_DIR/.venv-yolo/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv-yolo/bin/python"
    PYQT5_PLUGINS="$PROJECT_DIR/.venv-yolo/lib/python3.10/site-packages/PyQt5/Qt5/plugins"
else
    PYTHON_EXE="python3"
    PYQT5_PLUGINS=""
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

# 2. Xử lý tham số truyền vào
if [ $# -gt 0 ]; then
    echo "▶ Khởi chạy với file video chỉ định: $@"
    exec "$PYTHON_EXE" qr_gui.py "$@"
else
    SAMPLE_VIDEO="$PROJECT_DIR/test_qr_dataset/test_qr_multiscene.mp4"
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        SAMPLE_VIDEO="$PROJECT_DIR/sample_qr_office.mp4"
    fi

    if [ -f "$SAMPLE_VIDEO" ]; then
        echo "▶ Tự động nạp video mẫu thử nghiệm: $(basename "$SAMPLE_VIDEO")"
        echo "  (Bạn có thể bấm nút '📂 Chọn File Video...' trên giao diện để chọn bất kỳ video nào khác)"
        exec "$PYTHON_EXE" qr_gui.py "$SAMPLE_VIDEO"
    else
        echo "▶ Mở giao diện ứng dụng (Hãy bấm '📂 Chọn File Video...' để chọn video)..."
        exec "$PYTHON_EXE" qr_gui.py
    fi
fi
