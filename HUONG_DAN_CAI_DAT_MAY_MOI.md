# HƯỚNG DẪN CÀI ĐẶT & TRIỂN KHAI TRÊN MÁY TÍNH MỚI
### Dự án: RobotNongNghiep - Hệ thống nhận diện lối đi & Dẫn hướng Robot

Tài liệu này hướng dẫn chi tiết từng bước để thiết lập dự án trên một máy tính hoàn toàn mới (PC, Laptop, Mini PC công nghiệp, hoặc máy trạm gắn trên robot chạy Linux/Ubuntu).

---

## 1. Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu khuyến nghị | Ghi chú |
| :--- | :--- | :--- |
| **Hệ điều hành** | Ubuntu 20.04 / 22.04 / 24.04 LTS hoặc Debian 11/12 | Khuyến nghị Ubuntu 22.04 LTS x86_64 |
| **Python** | Python 3.9 - 3.12 | Có sẵn trên Ubuntu |
| **RAM** | Tối thiểu 4 GB (Khuyến nghị 8 GB) | Khi chạy AI YOLO Seg và luồng camera HD |
| **GPU (Tùy chọn)** | NVIDIA GTX/RTX (Hỗ trợ CUDA) | Nếu không có GPU NVIDIA, hệ thống **tự động chạy bằng CPU** |
| **Cổng kết nối** | Tối thiểu 2 cổng USB | 1 cổng cho Camera Orbbec, 1 cổng cho cáp nạp/UART STM32 |

---

## 2. Cách 1: Cài Đặt Tự Động 1-Click (Khuyên Dùng Nhất)

Toàn bộ quy trình cài đặt hệ thống, tạo môi trường ảo, nạp thư viện và phân quyền phần cứng đã được tự động hóa thông qua script `install.sh`.

### Bước 1: Giải nén gói mã nguồn
Chuyển file `RobotNongNghiep_code.zip` sang máy tính mới và giải nén:
```bash
unzip RobotNongNghiep_code.zip -d RobotNongNghiep
cd RobotNongNghiep
```

### Bước 2: Chạy script cài đặt
Cấp quyền thực thi và khởi chạy script `install.sh`:
```bash
chmod +x install.sh
./install.sh
```

**Script `install.sh` sẽ tự động thực hiện:**
1. Cài đặt các thư viện hệ thống cần thiết qua APT (`python3-venv`, `libgl1`, `libxcb-cursor0`, `v4l-utils`,...).
2. Khởi tạo môi trường ảo Python cô lập tại `.venv-yolo`.
3. Tự động kiểm tra xem máy mới có GPU NVIDIA hay không:
   * **Nếu có GPU NVIDIA**: Tự động cài đặt **PyTorch với CUDA 12.4** để tăng tốc nhận diện AI (~9–15 ms/frame).
   * **Nếu là máy CPU thuần**: Tự động cài đặt **PyTorch CPU** tối ưu tài nguyên.
4. Cài đặt toàn bộ các thư viện: OpenCV, PySide6 (giao diện GUI), Ultralytics YOLO, PySerial.
5. Cài đặt Udev Rules cho phần cứng:
   * Đăng ký thiết bị UART điều khiển STM32 cố định tại `/dev/ttySTM32`.
   * Cấp quyền đọc/ghi Camera Orbbec Astra Pro không cần gõ `sudo`.
6. Tự động chạy chẩn đoán kiểm tra tính toàn vẹn của môi trường.

---

## 3. Cách 2: Cài Đặt Thủ Công Từng Bước (Manual Setup)

Nếu máy tính của bạn không có kết nối mạng trực tiếp hoặc bạn muốn tự kiểm soát từng bước, hãy thực hiện theo thứ tự sau:

### Bước 1: Cài đặt các gói hệ thống (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential libgl1 libglib2.0-0 libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 v4l-utils usbutils
```

### Bước 2: Tạo môi trường ảo Python
```bash
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
pip install --upgrade pip setuptools wheel
```

### Bước 3: Cài đặt PyTorch
* **Trường hợp máy có GPU NVIDIA (CUDA 12.x)**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
* **Trường hợp máy không có GPU (Chạy CPU)**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### Bước 4: Cài đặt các thư viện dự án
```bash
pip install -r requirements.txt
```

### Bước 5: Cài đặt Udev Rule cho Phần Cứng
```bash
chmod +x setup/setup_udev.sh
./setup/setup_udev.sh
```
> [!NOTE]
> Sau khi chạy script udev, user của bạn sẽ được thêm vào các nhóm `dialout`, `video`, `plugdev`. Để quyền có hiệu lực ngay lập tức, bạn có thể chạy:
> ```bash
> sudo usermod -aG dialout,video,plugdev $USER
> newgrp dialout
> newgrp video
> ```
> Hoặc Đăng xuất (Logout) và Đăng nhập lại vào hệ điều hành.

---

## 4. Thiết Lập & Kiểm Tra Phần Cứng Trên Máy Mới

### 1. Camera Orbbec Astra Pro
* **Cắm cáp USB**: Cắm vào cổng USB (ưu tiên cổng USB 3.0 màu xanh hoặc cổng USB gắn trực tiếp trên mainboard).
* **Kiểm tra nhận diện**:
  ```bash
  lsusb | grep -i 2bc5
  ```
  Kết quả chuẩn:
  * `2bc5:0501`: Cảm biến màu HD (Astra Pro HD Camera).
  * `2bc5:0403`: Cảm biến chiều sâu (Astra Depth Sensor).
* **Chạy script kiểm tra camera**:
  ```bash
  ./.venv-yolo/bin/python test_orbbec_camera.py
  ```
  *(Cửa sổ hình ảnh 1280x720 mượt mà 30 FPS sẽ hiện lên. Nhấn phím `q` để thoát).*

### 2. Mạch Điều Khiển Robot STM32 (UART)
* **Cắm cáp USB-to-UART**: Cắm cổng USB kết nối với bo mạch STM32.
* **Quy tắc udev linh hoạt**: Hệ thống đã được cấu hình nhận diện tự động tất cả các dòng chip USB-Serial phổ biến:
  * Silicon Labs CP2102 / CP210x (`10c4:ea60`)
  * WCH CH340 / CH341 (`1a86:7523`)
  * FTDI FT232R / FT232H (`0403:6001`)
  * Prolific PL2303 (`067b:2303`)
  * STM32 Native USB CDC Virtual COM (`0483:5740`)
* **Kiểm tra cổng đã gán cố định**:
  ```bash
  ls -l /dev/ttySTM32
  ```
  *(Cổng symlink sẽ trỏ trực tiếp về thiết bị USB tương ứng mà không bị đổi tên khi cắm lại).*

---

## 5. Khởi Động & Sử Dụng Ứng Dụng

### 1. Khởi động Giao diện Đồ họa chính (GUI)
Chỉ cần chạy lệnh:
```bash
./start.sh
```
Giao diện chính sẽ mở lên:
* Để phát luồng trực tiếp từ Camera Orbbec: Bấm nút **`📷 Mở Camera Orbbec`** ở panel bên phải.
* Để bật/tắt gửi tín hiệu UART xuống STM32: Tích chọn checkbox **`Bật gửi UART`** (gửi gói 5-byte union mỗi 0.1s).
* Chuyển đổi giữa 2 chế độ:
  * **AI YOLO Seg**: Sử dụng model `models/farm_path_bootstrap.pt`.
  * **Xử lý ảnh màu CIELAB**: Chạy phân đoạn màu không cần GPU.

### 2. Khởi động nhanh các chế độ chuyên biệt
* **Hệ Thống Tích Hợp (Dò Đường + Quét Mã QR + Đo Kích Thước):**
  ```bash
  ./start_combined.sh
  ```
* **Chế độ Dò Đường Nhà Màng Bạt Đen (YOLOv8 Segmentation):**
  ```bash
  ./start_nhamang.sh
  ```
* **Chế độ Dò Đường Thảm Đen:**
  ```bash
  ./start_thamden.sh
  ```
* **Chế độ Quét & Đọc Mã QR độc lập (Giao diện tùy chỉnh Confidence & Dewarp):**
  ```bash
  ./start_qr.sh
  ```

### 3. Khởi động nhanh từ dòng lệnh (CLI Options)
* **Mở camera Orbbec trực tiếp ở chế độ AI:**
  ```bash
  ./.venv-yolo/bin/python main.py --video 0 --mode ai --uart
  ```
* **Chạy trên Robot không có màn hình (Headless Mode):**
  Dành cho máy tính nhúng gắn trên robot chạy nền tiết kiệm tài nguyên CPU/GPU:
  ```bash
  ./.venv-yolo/bin/python main.py --headless --video 0 --mode ai --uart
  ```
* **Chạy Quét QR từ dòng lệnh (Terminal QR Scanner):**
  ```bash
  ./run.sh --source 0
  ```

### 4. Chạy toàn bộ Unit Tests kiểm tra tính toàn vẹn
```bash
./.venv-yolo/bin/python -m unittest discover -s tests -p "test_*.py"
```
*(Tất cả 41 test phải báo `OK`)*.

---

## 6. Xử Lý Các Sự Cố Thường Gặp (Troubleshooting)

### Lỗi 1: `Could not load the Qt platform plugin "xcb"` khi mở GUI
* **Nguyên nhân**: Bản phân phối Linux của máy mới thiếu một số thư viện X11/XCB của Qt.
* **Cách khắc phục**:
  ```bash
  sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0
  ```

### Lỗi 2: `ImportError: libGL.so.1: cannot open shared object file`
* **Nguyên nhân**: OpenCV thiếu thư viện OpenGL đồ họa.
* **Cách khắc phục**:
  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0
  ```

### Lỗi 3: `Permission denied: '/dev/ttySTM32'` hoặc `'/dev/video0'`
* **Nguyên nhân**: User hiện tại chưa được cấp quyền truy cập thiết bị phần cứng.
* **Cách khắc phục**:
  ```bash
  sudo usermod -aG dialout,video,plugdev $USER
  ```
  Sau đó đăng xuất (logout) rồi đăng nhập lại hoặc chạy tạm `sudo chmod 666 /dev/ttySTM32 /dev/video*`.

### Lỗi 4: Máy mới có GPU NVIDIA nhưng PyTorch chỉ nhận `CPU`
* **Nguyên nhân**: Cài đặt PyTorch bản CPU hoặc thiếu driver NVIDIA.
* **Cách khắc phục**:
  Kiểm tra lệnh `nvidia-smi`. Nếu driver NVIDIA hoạt động bình thường, cài đặt lại PyTorch CUDA:
  ```bash
  ./.venv-yolo/bin/pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

---

## 7. Cấu Trúc Gói Mã Nguồn Khi Chuyển Sang Máy Mới

```text
RobotNongNghiep/
├── install.sh                     # Script cài đặt tự động 1-click cho máy mới
├── start.sh                       # Script khởi động ứng dụng điều hướng chính (PySide6)
├── start_combined.sh              # Script khởi động Hệ Thống Tích Hợp (Dò Đường + Quét QR)
├── start_nhamang.sh               # Script khởi động Dò đường nhà màng bạt đen
├── start_thamden.sh               # Script khởi động nhanh chế độ thảm đen
├── start_qr.sh                    # Script khởi động Giao diện Quét QR riêng biệt (PyQt5)
├── run.sh                         # Script quét mã QR từ Terminal CLI
├── stop.sh                        # Script dừng ứng dụng
├── main.py                        # Điểm khởi chạy chính (GUI PySide6 & Headless CLI)
├── combined_app.py                # Ứng dụng tích hợp Dò Đường + QR Scanner
├── qr_gui.py                      # Giao diện Quét QR + Đo Kích Thước (PyQt5)
├── main_qr.py                     # Module CLI QR Detector
├── test_orbbec_camera.py          # Script kiểm tra độc lập camera Orbbec
├── requirements.txt               # Danh sách thư viện Python tổng hợp
├── requirements_qr.txt            # Danh sách thư viện phục vụ quét QR
├── HUONG_DAN_CAI_DAT_MAY_MOI.md   # Hướng dẫn chi tiết triển khai máy mới
├── HUONG_DAN_SETUP_CAMERA_ORBBEC.md # Hướng dẫn kỹ thuật và tối ưu Camera
├── README.md                      # Giới thiệu tổng quan dự án
├── README_QR.md                   # Hướng dẫn chi tiết hệ thống nhận diện QR
├── models/                        # Chứa model YOLO Segmentation (.pt)
│   ├── nhamang_batden_seg.pt      # Model YOLOv8 nhận diện bạt đen nhà màng
│   ├── nhamang_batden_seg.json    # Cấu hình bạt đen nhà màng
│   ├── farm_path_bootstrap.pt     # Model nhận diện lối đi chính
│   ├── thamden_seg.pt             # Model nhận diện thảm đen
│   └── pretrained/yolo26n-seg.pt  # Model cơ sở
├── profiles/                      # File lưu thông số cấu hình và màu sắc
│   ├── default_profile.json       # Profile mặc định
│   ├── nhamang_batden_profile.json# Profile bạt đen nhà màng
│   └── thamden_profile.json       # Profile thảm đen
├── qr_scanner/                    # Package giải mã, tiền xử lý và đo đạc QR
├── src/                           # Toàn bộ mã nguồn xử lý thuật toán và giao diện
├── setup/                         # File cấu hình Udev Rule cho phần cứng
└── tests/                         # Bộ unit tests tự động đảm bảo độ ổn định
```

