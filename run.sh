#!/usr/bin/env bash
# ==============================================================================
# Script khởi chạy Hệ thống Phát hiện & Giải mã mã QR Thời gian thực (Ubuntu/Linux)
# Hỗ trợ Webcam, Video File, RTSP/HTTP Stream và Headless mode.
# Tự động quản lý virtualenv (.venv) và cài đặt dependencies không cần quyền sudo.
# ==============================================================================

set -e

# 1. Tự động chuyển về thư mục gốc của dự án dựa trên vị trí file script
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
    SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    [[ $SCRIPT_SOURCE != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# Tắt cảnh báo font vô hại từ Qt/OpenCV plugin trên Linux
export QT_LOGGING_RULES="*.warning=false"

# 2. Kiểm tra Python hệ thống
if ! command -v python3 &>/dev/null; then
    echo "====================================================================" >&2
    echo "[LỖI NGHIÊM TRỌNG] Không tìm thấy Python 3 trên hệ thống!" >&2
    echo "Vui lòng cài đặt Python 3 trên Ubuntu/Linux: sudo apt update && sudo apt install python3" >&2
    echo "====================================================================" >&2
    exit 1
fi

# 3. Quản lý Môi trường ảo Python (.venv)
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
REQS_FILE="$SCRIPT_DIR/requirements_qr.txt"
HASH_FILE="$VENV_DIR/.requirements_hash"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[INFO] Đang khởi tạo môi trường ảo Python (.venv)..."
    if command -v virtualenv &>/dev/null; then
        virtualenv "$VENV_DIR" >/dev/null 2>&1
    elif [ -f "$SCRIPT_DIR/.venv-yolo/bin/virtualenv" ]; then
        "$SCRIPT_DIR/.venv-yolo/bin/virtualenv" "$VENV_DIR" >/dev/null 2>&1
    else
        python3 -m venv "$VENV_DIR" || {
            echo "[LỖI] Không thể tạo virtualenv bằng 'python3 -m venv' do thiếu package python3-venv." >&2
            echo "Đang thử tạo bằng virtualenv..." >&2
            python3 -m pip install --user virtualenv && python3 -m virtualenv "$VENV_DIR"
        }
    fi
fi

# 4. Kiểm tra và cài đặt dependencies (chỉ cài lần đầu hoặc khi requirements_qr.txt thay đổi)
CURRENT_HASH=""
if command -v sha256sum &>/dev/null; then
    CURRENT_HASH="$(sha256sum "$REQS_FILE" | awk '{print $1}')"
elif command -v md5sum &>/dev/null; then
    CURRENT_HASH="$(md5sum "$REQS_FILE" | awk '{print $1}')"
fi

LAST_HASH=""
[ -f "$HASH_FILE" ] && LAST_HASH="$(cat "$HASH_FILE" 2>/dev/null || true)"

if [ "$CURRENT_HASH" != "$LAST_HASH" ]; then
    echo "[INFO] Đang đồng bộ thư viện vào .venv từ requirements_qr.txt..."
    "$VENV_PIP" install -q -r "$REQS_FILE"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    echo "[INFO] Đồng bộ thư viện hoàn tất."
fi

# 5. Nếu người dùng gọi --help hoặc -h
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        echo "===================================================================="
        echo "  HƯỚNG DẪN SỬ DỤNG - HỆ THỐNG QUÉT & ĐỌC MÃ QR (UBUNTU/LINUX)     "
        echo "===================================================================="
        echo "Cách chạy cơ bản:"
        echo "  $0                                   # Mở menu tương tác chọn nguồn"
        echo "  $0 --gui                             # Mở Giao diện Đồ họa (GUI có slider Độ tin cậy)"
        echo "  $0 --source 0                        # Chạy trực tiếp với Webcam 0"
        echo "  $0 --source \"/duong/dan/video.mp4\"   # Chạy với file video"
        echo "  $0 --source \"rtsp://...\"             # Chạy với luồng RTSP/HTTP camera"
        echo "  $0 --headless --benchmark --source \"video.mp4\"  # Đo tốc độ quét tối đa"
        echo "  $0 --help                            # Hiển thị hướng dẫn này"
        echo "--------------------------------------------------------------------"
        exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" --help
    fi
    if [ "$arg" == "--gui" ] || [ "$arg" == "-g" ]; then
        echo "▶ Mở Giao diện Đồ họa Đơn giản (GUI)..."
        # Bỏ qua cờ --gui khi truyền tiếp cho python script
        shift || true
        exec "$VENV_PYTHON" "$SCRIPT_DIR/qr_gui.py" "$@"
    fi
done

# 6. Nếu chạy không có tham số: Hiển thị Menu đơn giản chọn Camera/Video
if [ $# -eq 0 ]; then
    # Kiểm tra xem có webcam /dev/video* nào không
    HAS_WEBCAM=0
    if compgen -G "/dev/video*" >/dev/null 2>&1; then
        HAS_WEBCAM=1
    fi

    SAMPLE_GREENHOUSE="Quỳnh Robotics [13-10-2026 20_33]/2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4"
    SAMPLE_TEST_QR="test_qr_dataset/test_qr_multiscene.mp4"

    # Đảm bảo video test đã có sẵn
    if [ ! -f "$SAMPLE_TEST_QR" ]; then
        "$VENV_PYTHON" "$SCRIPT_DIR/generate_test_data.py" >/dev/null 2>&1
    fi

    echo ""
    echo "===================================================================="
    echo "      HỆ THỐNG NHẬN DIỆN & GIẢI MÃ MÃ QR THỜI GIAN THỰC             "
    echo "             (OpenCV + ZXing-C++ Pipeline on Linux)                 "
    echo "===================================================================="
    echo " Vui lòng chọn nguồn dữ liệu đầu vào:"
    echo "   [G] MỞ GIAO DIỆN ĐỒ HỌA (GUI - Có thanh trượt Độ tin cậy & Đọc mã) [KHUYÊN DÙNG]"
    if [ $HAS_WEBCAM -eq 1 ]; then
        echo "   [1] Webcam mặc định (Camera #0: /dev/video0)"
        DEFAULT_CHOICE="G"
    else
        echo "   [1] Webcam #0 (Không tìm thấy /dev/video* cắm ngoài)"
        DEFAULT_CHOICE="G"
    fi
    echo "   [2] Video nhà màng dưa lưới (2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4)"
    echo "   [3] Video kiểm thử mã QR đa cảnh (test_qr_multiscene.mp4: chuẩn, nghiêng, đa mã)"
    echo "   [4] Nhập đường dẫn file video hoặc URL stream tùy chọn"
    echo "   [0] Thoát"
    echo "--------------------------------------------------------------------"
    read -rp " Nhập lựa chọn [Mặc định: $DEFAULT_CHOICE]: " USER_CHOICE
    USER_CHOICE="${USER_CHOICE:-$DEFAULT_CHOICE}"

    case "$USER_CHOICE" in
        [gG])
            exec "$VENV_PYTHON" "$SCRIPT_DIR/qr_gui.py"
            ;;
        1)
            exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" --source "0"
            ;;
        2)
            exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" --source "$SAMPLE_GREENHOUSE"
            ;;
        3)
            exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" --source "$SAMPLE_TEST_QR"
            ;;
        4)
            read -rp " Nhập đường dẫn file video hoặc URL stream: " CUSTOM_SRC
            if [ -z "$CUSTOM_SRC" ]; then
                echo "[LỖI] Đường dẫn không được để trống!" >&2
                exit 1
            fi
            exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" --source "$CUSTOM_SRC"
            ;;
        0)
            echo "Đã thoát."
            exit 0
            ;;
        *)
            echo "[LỖI] Lựa chọn '$USER_CHOICE' không hợp lệ." >&2
            exit 1
            ;;
    esac
fi

# Tắt cảnh báo font vô hại từ OpenCV Qt plugin trên Linux
export QT_LOGGING_RULES="*.warning=false"

# 7. Chạy với các tham số được truyền trực tiếp từ dòng lệnh
exec "$VENV_PYTHON" "$SCRIPT_DIR/main_qr.py" "$@"
