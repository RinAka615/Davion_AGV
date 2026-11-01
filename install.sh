#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "  RobotNongNghiep - CÀI ĐẶT TỰ ĐỘNG CHO MÁY TÍNH MỚI     "
echo "  (Hỗ trợ Ubuntu 20.04/22.04/24.04, Debian, Jetson/x86)   "
echo "=========================================================="
echo ""

# 1. Kiểm tra quyền sudo
HAS_SUDO=false
if command -v sudo &> /dev/null; then
    HAS_SUDO=true
fi

# 2. Cài đặt các gói hệ thống APT cần thiết (nếu có sudo và dùng debian/ubuntu)
if [ "$HAS_SUDO" = true ] && command -v apt-get &> /dev/null; then
    echo "▶ [Bước 1/5] Cập nhật và cài đặt thư viện hệ thống cần thiết..."
    echo "  (Bao gồm: python3-venv, thư viện đồ họa OpenCV và PySide6/Qt xcb)"
    sudo apt-get update -y
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libxcb-cursor0 \
        libxcb-xinerama0 \
        libxkbcommon-x11-0 \
        v4l-utils \
        usbutils
else
    echo "ℹ️ Bỏ qua cài đặt APT (không dùng Debian/Ubuntu hoặc không có sudo)."
fi

# 3. Tạo môi trường ảo Python .venv-yolo
echo ""
echo "▶ [Bước 2/5] Khởi tạo môi trường ảo Python (.venv-yolo)..."
if [ ! -d "$PROJECT_DIR/.venv-yolo" ]; then
    python3 -m venv "$PROJECT_DIR/.venv-yolo"
    echo "✓ Đã tạo thành công thư mục .venv-yolo"
else
    echo "✓ Thư mục .venv-yolo đã tồn tại, tiếp tục sử dụng."
fi

VENV_PYTHON="$PROJECT_DIR/.venv-yolo/bin/python"
VENV_PIP="$PROJECT_DIR/.venv-yolo/bin/pip"

# Nâng cấp pip, setuptools, wheel
echo "▶ Đang cập nhật pip & wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel

# 4. Kiểm tra GPU NVIDIA & Cài đặt PyTorch phù hợp
echo ""
echo "▶ [Bước 3/5] Phát hiện phần cứng GPU để cài đặt PyTorch..."
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=true
    fi
fi

if [ "$HAS_NVIDIA_GPU" = false ] && command -v lspci &> /dev/null; then
    if lspci | grep -i nvidia &> /dev/null; then
        HAS_NVIDIA_GPU=true
    fi
fi

if [ "$HAS_NVIDIA_GPU" = true ]; then
    echo "🎮 Phát hiện GPU NVIDIA trên máy!"
    echo "▶ Đang cài đặt PyTorch hỗ trợ GPU CUDA (CUDA 12.4)..."
    "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
    echo "💻 Không phát hiện GPU NVIDIA rời (hoặc máy dùng CPU/Intel/AMD)."
    echo "▶ Đang cài đặt PyTorch phiên bản CPU..."
    "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 5. Cài đặt các thư viện trong requirements.txt
echo ""
echo "▶ [Bước 4/5] Cài đặt các thư viện cốt lõi (requirements.txt)..."
"$VENV_PIP" install -r requirements.txt

# 6. Cài đặt Udev Rules cho phần cứng (STM32 & Camera Orbbec)
echo ""
echo "▶ [Bước 5/5] Cấu hình quyền phần cứng (Udev Rules)..."
if [ -f "$PROJECT_DIR/setup/setup_udev.sh" ]; then
    chmod +x "$PROJECT_DIR/setup/setup_udev.sh"
    "$PROJECT_DIR/setup/setup_udev.sh" || echo "⚠️ Cảnh báo: Cài đặt udev gặp lỗi nhỏ hoặc bị bỏ qua."
fi

# 7. Kiểm tra tự chẩn đoán (Self-Diagnostic Check)
echo ""
echo "=========================================================="
echo "  KIỂM TRA CHẨN ĐOÁN MÔI TRƯỜNG SAU CÀI ĐẶT               "
echo "=========================================================="

"$VENV_PYTHON" -c '
import sys
print(f"▶ Python: {sys.version.split()[0]} ({sys.executable})")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else "Không có"
    print(f"▶ PyTorch: {torch.__version__} | CUDA Khả dụng: {cuda_ok} ({dev_name})")
except Exception as e:
    print(f"❌ Lỗi nạp PyTorch: {e}")

try:
    import cv2
    print(f"▶ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"❌ Lỗi nạp OpenCV: {e}")

try:
    import PySide6
    print(f"▶ PySide6 (GUI): {PySide6.__version__}")
except Exception as e:
    print(f"❌ Lỗi nạp PySide6: {e}")

try:
    import ultralytics
    print(f"▶ Ultralytics YOLO: {ultralytics.__version__}")
except Exception as e:
    print(f"❌ Lỗi nạp Ultralytics: {e}")

try:
    import serial
    print(f"▶ PySerial (UART): {serial.__version__}")
except Exception as e:
    print(f"❌ Lỗi nạp PySerial: {e}")
'

echo ""
echo "=========================================================="
echo "🎉 HOÀN TẤT CÀI ĐẶT HỆ THỐNG!"
echo "Để khởi động ứng dụng trên máy mới:"
echo "  1. Chạy lệnh: ./start.sh"
echo "  2. Hoặc kiểm tra camera Orbbec: ./.venv-yolo/bin/python test_orbbec_camera.py"
echo "=========================================================="

