#!/usr/bin/env bash
set -e

# Chuyển về thư mục chứa mã nguồn của dự án
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "  RobotNongNghiep - Hệ thống nhận diện lối đi nhà màng   "
echo "  (Hỗ trợ 2 Chế độ: AI YOLO Segmentation & Xử lý ảnh)    "
echo "=========================================================="

# Tự động tối ưu nền tảng đồ họa hiển thị trên Linux (Wayland / X11)
if [ "$XDG_SESSION_TYPE" = "wayland" ] && [ -z "$QT_QPA_PLATFORM" ]; then
    export QT_QPA_PLATFORM="wayland;xcb"
fi
export QT_LOGGING_RULES="*.warning=false"

# Cấu hình số luồng CPU (Mặc định 2 luồng là điểm ngọt FPS cao nhất trên Pi 5)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-2}

# 1. Chọn môi trường Python tối ưu (.venv-yolo trước)
if [ -f "$PROJECT_DIR/.venv-yolo/bin/python" ]; then
    PYTHON_EXE="$PROJECT_DIR/.venv-yolo/bin/python"
    echo "▶ Sử dụng Python VirtualEnv: .venv-yolo"
else
    echo "⚠️ Chưa tìm thấy môi trường ảo .venv-yolo trên máy này!"
    if [ -t 0 ]; then
        read -p "▶ Bạn có muốn chạy './install.sh' để tự động cài đặt môi trường ngay bây giờ không? [Y/n] " answer
        answer=${answer:-Y}
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            ./install.sh
            PYTHON_EXE="$PROJECT_DIR/.venv-yolo/bin/python"
        else
            PYTHON_EXE="python3"
            echo "▶ Tiếp tục với python3 hệ thống (có thể thiếu thư viện)..."
        fi
    elif command -v python3 &> /dev/null; then
        PYTHON_EXE="python3"
        echo "▶ Chạy ở chế độ không tương tác, sử dụng hệ thống python3"
    else
        echo "❌ Lỗi: Không tìm thấy python trên hệ thống!"
        echo "Vui lòng chạy './install.sh' để thiết lập môi trường cho máy mới."
        exit 1
    fi
fi

if [ -f "$PROJECT_DIR/models/nhamang_batden_seg.pt" ]; then
    MODEL_FILE="$PROJECT_DIR/models/nhamang_batden_seg.pt"
    PROFILE_ARGS=(--profile "$PROJECT_DIR/profiles/nhamang_batden_profile.json")
else
    MODEL_FILE="$PROJECT_DIR/models/farm_path_bootstrap.pt"
    PROFILE_ARGS=()
fi

# 2. Xử lý tham số truyền vào
if [ $# -gt 0 ]; then
    echo "▶ Khởi chạy với tham số: $@"
    "$PYTHON_EXE" main.py "$@"
else
    # Nếu không truyền tham số, ưu tiên video bạt đen mới nhất hoặc video mẫu nhà màng
    if [ -f "$PROJECT_DIR/nhamang_20261025_030235.mp4" ]; then
        SAMPLE_VIDEO="$PROJECT_DIR/nhamang_20261025_030235.mp4"
    else
        SAMPLE_VIDEO=$(find "$PROJECT_DIR/Quỳnh Robotics [13-10-2026 20_33]" -maxdepth 1 -name "*.mp4" 2>/dev/null | head -n 1 || true)
    fi
    
    if [ -n "$SAMPLE_VIDEO" ] && [ -f "$SAMPLE_VIDEO" ]; then
        echo "▶ Tự động nạp video mẫu có sẵn: $(basename "$SAMPLE_VIDEO")"
        echo "▶ Nạp model tối ưu: $(basename "$MODEL_FILE")"
        "$PYTHON_EXE" main.py --video "$SAMPLE_VIDEO" --model "$MODEL_FILE" "${PROFILE_ARGS[@]}"
    else
        echo "▶ Mở giao diện chính (chưa nạp video)..."
        "$PYTHON_EXE" main.py --model "$MODEL_FILE" "${PROFILE_ARGS[@]}"
    fi
fi
