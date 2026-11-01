#!/usr/bin/env bash
set -e
echo "=========================================================="
echo "  Cài đặt Udev Rule: STM32 Controller & Camera Orbbec     "
echo "=========================================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Cài đặt udev rule cho STM32 Controller
if [ -f "$SCRIPT_DIR/99-stm32.rules" ]; then
    echo "▶ Đang cài đặt udev rule STM32 (/dev/ttySTM32)..."
    sudo cp "$SCRIPT_DIR/99-stm32.rules" /etc/udev/rules.d/99-stm32.rules
fi

# 2. Cài đặt udev rule cho Camera Orbbec
if [ -f "$SCRIPT_DIR/99-orbbec-astra.rules" ]; then
    echo "▶ Đang cài đặt udev rule Camera Orbbec..."
    sudo cp "$SCRIPT_DIR/99-orbbec-astra.rules" /etc/udev/rules.d/99-orbbec-astra.rules
fi

# 3. Thêm user vào group video và dialout
echo "▶ Cấp quyền truy cập cổng COM và Camera cho user: $USER..."
sudo usermod -aG dialout,video,plugdev "$USER" 2>/dev/null || true

# 4. Nạp lại udev rules
echo "▶ Nạp lại udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "✅ Cài đặt thành công!"
echo "  - Thiết bị STM32 CP2102 sẽ tự động nhận diện: /dev/ttySTM32"
echo "  - Camera Orbbec Astra Pro sẽ có đầy đủ quyền đọc/ghi không cần sudo"
echo "ℹ️ (Nếu vừa được thêm vào nhóm video/dialout, bạn có thể cần logout/login lại để quyền có hiệu lực)"
