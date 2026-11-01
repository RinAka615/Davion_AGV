# Hệ Thống Nhận Diện, Nắn Phối Cảnh & Đọc Mã QR Thời Gian Thực
**(Real-Time Classical Computer Vision QR Scanner on Ubuntu/Linux)**

Ứng dụng hoàn chỉnh nhận diện và giải mã mã QR thời gian thực từ Webcam, Video File, hoặc luồng RTSP/HTTP Stream. Được xây dựng hoàn toàn bằng **Thị giác máy tính cổ điển (Classical Computer Vision - OpenCV & ZXing-C++)**, không sử dụng Deep Learning/YOLO, đạt tốc độ xử lý thực tế **46 - 145+ FPS**.

---

## 1. Kết Quả Kiểm Tra Video Mẫu Đầu Vào (`2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4`)

Theo yêu cầu nghiêm ngặt của bài toán: **Báo cáo trung thực nội dung video, không nhầm cây/chậu/khung nhà màng thành QR**:

- **Thông số video**: 720 × 1280 (quay dọc portrait), 30.00 FPS, tổng cộng **1123 frames**, thời lượng **37.43 giây**.
- **Nội dung thực tế**: Camera di chuyển dọc lối đi nhà màng trồng dưa lưới (bạt đen phủ đất ở giữa lối đi, hai bên là các bầu giá thể màu trắng có cây dưa lưới leo giàn dây, hệ thống khung vòm thép mạ kẽm và mái màng nilon).
- **Kết quả quét toàn diện 100% (1123/1123 frames)**:
  - **Không có bất kỳ mã QR nào xuất hiện trong video**.
  - **Không có bất kỳ báo động giả nào (Zero False Positives)**: Hệ thống phân biệt chính xác cấu trúc Finder Pattern của mã QR và hoàn toàn không nhầm lẫn các vật thể như lá dưa, cọc chống, nẹp nhà màng hay bầu giá thể thành mã QR.
- **Tốc độ quét benchmark trên video mẫu**: Đạt **46.6 FPS** (nhanh hơn tốc độ phát 30 FPS của video).

---

## 2. Kết Quả Kiểm Thử Bộ Dữ Liệu QR Độc Lập (Ground Truth)

Do video mẫu trong nhà màng không có mã QR, hệ thống được kiểm thử tự động với bộ dữ liệu chuẩn trong thư mục `test_qr_dataset/`:

| Kịch Bản Thử Nghiệm | Dữ Liệu Đầu Vào | Nội Dung Kỳ Vọng (Ground Truth) | Kết Quả Thực Tế | Trạng Thái |
| :--- | :--- | :--- | :--- | :---: |
| **1. Mã QR Tiêu Chuẩn** | `01_standard_qr.png` | `https://github.com/warlord/RobotNongNghiep` | Đọc chính xác 100% | **PASS** |
| **2. Mã QR Nghiêng Phối Cảnh** | `02_tilted_perspective_qr.png` | `QR-MELON-A3-TILTED` (nghiêng 25°, góc xiên) | Nắn thẳng (Dewarp) & đọc chính xác 100% | **PASS** |
| **3. Đa Mã QR (Multi-QR)** | `03_multi_qr.png` | 3 mã: `QR-BED-ROW-01`, `QR-BED-ROW-02`, `QR-BED-ROW-03` | Phát hiện và đọc đồng thời cả 3 mã | **PASS** |
| **4. Khung Hình Âm Tính (Negative)** | `04_negative_greenhouse.png` | Khung hình thực tế nhà màng dưa lưới | 0 phát hiện, 0 báo động giả | **PASS** |
| **5. Video Tổng Hợp Đa Cảnh** | `test_qr_multiscene.mp4` (10s @ 30FPS) | Cả 4 kịch bản nối tiếp nhau | Đọc chính xác 5/5 mã duy nhất, cooldown 2.0s hoạt động chuẩn | **PASS** |

Chạy lại toàn bộ test suite bất cứ lúc nào bằng lệnh:
```bash
./.venv/bin/python tests/test_qr_pipeline.py
```

---

## 3. Cấu Trúc Mã Nguồn

```
RobotNongNghiep/
├── run.sh                     # Script khởi chạy duy nhất (tự quản lý .venv, hỗ trợ menu và CLI)
├── requirements_qr.txt        # Danh sách thư viện tối thiểu (opencv-python, numpy, zxing-cpp)
├── main_qr.py                 # File thực thi chính (vòng lặp capture, CLI args, phím tắt)
├── qr_scanner/                # Module xử lý cốt lõi
│   ├── __init__.py
│   ├── detector.py            # Động cơ phát hiện đa tầng: ZXing-C++ + OpenCV fallback
│   ├── preprocessor.py        # Nắn phối cảnh (warp_perspective), CLAHE, phân ngưỡng
│   ├── logger.py              # Ghi log JSONL chống trùng lặp theo thời gian cooldown
│   └── visualizer.py          # Vẽ viền xanh/cam, bảng Telemetry HUD, giữ tỉ lệ video dọc
├── generate_test_data.py      # Sinh dữ liệu ảnh & video kiểm thử chuẩn
├── tests/
│   └── test_qr_pipeline.py    # Test suite tự động kiểm tra 5 kịch bản
├── test_qr_dataset/           # Chứa các ảnh và video thử nghiệm
└── output_qr/
    ├── qr_events.jsonl        # File nhật ký sự kiện đọc mã QR thành công
    └── snapshots/             # Lưu ảnh crop các mã đã nắn thẳng
```

---

## 4. Hướng Dẫn Sử Dụng Chi Tiết

Script `run.sh` và `start_qr.sh` đã được cấp quyền thực thi (`chmod +x`) và có thể chạy từ **bất kỳ thư mục nào**:

### 4.1. Mở Giao Diện Đồ Họa Đơn Giản (GUI - Có Thanh Trượt Độ Tin Cậy & Đọc Mã)
Chạy một trong hai lệnh sau:
```bash
./start_qr.sh
```
hoặc:
```bash
./run.sh --gui
```
**Các tính năng nổi bật trên giao diện UI:**
1. **Thanh trượt Độ tin cậy (Confidence Slider: 5% - 95%)**:
   - **Kéo sang Trái (5% - 35%)**: Giảm độ tin cậy xuống mức thấp để kích hoạt quét sâu đa tầng (Multi-pass), tăng tương phản thích nghi (CLAHE), phân ngưỡng và hiển thị cả các vùng mã nghi vấn (viền cam) bị mờ, nghiêng hoặc thiếu sáng.
   - **Kéo sang Phải (70% - 95%)**: Chế độ nghiêm ngặt, chỉ hiển thị mã QR có độ rõ nét cao và giải mã hoàn toàn chính xác.
   - Khi đang **Tạm dừng**, kéo thanh trượt sẽ quét lại frame ngay lập tức để người dùng thấy kết quả tức thì.
2. **Khung Hiển Thị Nội Dung Mã QR (Decoded Content)**:
   - Hiển thị toàn bộ chuỗi text giải mã được bằng phông chữ to rõ.
   - Nút **"📋 Sao Chép Nội Dung"**: Copy nhanh vào Clipboard máy tính chỉ với 1 click.
   - Xem trước ảnh nắn thẳng vuông góc (Dewarped Patch) của mã QR.
3. **Khung Đo Lường Kích Thước & Khoảng Cách (QR Geometry & Sizing)**:
   - **Kích thước Pixel thực tế**: Đo chiều rộng × chiều cao (`width × height px`), độ dài từng cạnh (`Top, Bottom, Left, Right px`), diện tích pixel (`px²`) và góc xoay nghiêng (`Orientation °`).
   - **Quy đổi kích thước thực (mm/cm)**: Cho phép chọn kích thước tem in thực tế (ví dụ: `50 mm`, `80 mm`, `100 mm` hoặc nhập số mm tùy chỉnh).
   - **Độ phân giải & Ước tính khoảng cách**: Tự động tính toán độ phân giải (`px/mm` và `mm/px`), đồng thời ước lượng khoảng cách từ Camera/Webcam đến mã QR theo mô hình Pinhole (ví dụ: `Khoảng cách: ~ 45.2 cm`).
   - **Hiển thị trực quan trên camera**: Nhãn viền quanh mã QR hiển thị trực tiếp kích thước `📐 185×182 px`.
4. **Thanh Tua Video & Phím Duyệt Từng Frame**:
   - Thanh trượt tua video (Timeline Slider) đến bất kỳ giây nào.
   - Hai nút `◀ 1 Frame` và `1 Frame ▶` giúp soi từng khung hình chi tiết.
4. **Bảng Lịch Sử Quét Mã (Scan History)**:
   - Lưu trữ danh sách mã đã đọc kèm timestamp.
   - **Nhấn đúp (Double click)** vào dòng bất kỳ để video tự động nhảy ngay đến khung hình phát hiện mã đó!

### 4.2. Mở Menu Tương Tác Dòng Lệnh
```bash
./run.sh
```
Giao diện dòng lệnh sẽ tự động nhận diện thiết bị và cho phép chọn nhanh:
- `1`: Mở Webcam mặc định (Camera 0)
- `2`: Mở Video nhà màng (`2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4`)
- `3`: Mở Video kiểm thử QR đa cảnh (`test_qr_multiscene.mp4`)
- `4`: Nhập đường dẫn video hoặc URL stream tùy ý

### 4.2. Mở Trực Tiếp Webcam
```bash
./run.sh --source 0
```

### 4.3. Chạy File Video
```bash
./run.sh --source "Quỳnh Robotics [13-10-2026 20_33]/2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4"
```
hoặc chạy video test:
```bash
./run.sh --source "test_qr_dataset/test_qr_multiscene.mp4"
```

### 4.4. Chạy Luồng RTSP/HTTP Stream
```bash
./run.sh --source "rtsp://username:password@192.168.1.100:554/stream1"
```

### 4.5. Chế Độ Headless & Benchmark Tối Đa Tốc Độ
Dành cho máy chủ, môi trường SSH không có màn hình hoặc đo đạc hiệu năng FPS:
```bash
./run.sh --source "test_qr_dataset/test_qr_multiscene.mp4" --headless --benchmark
```

### 4.6. Chạy Từ Ngoài Thư Mục Dự Án
```bash
cd /tmp
/home/warlord/RobotNongNghiep/run.sh --help
```

---

## 5. Phím Tắt Điều Khiển Trong Cửa Sổ Video
- **`Q`** hoặc **`ESC`**: Thoát chương trình sạch sẽ, giải phóng camera và đóng cửa sổ.
- **`SPACE` (Phím cách)**: Tạm dừng / Tiếp tục phát video.
- **`S`**: Chụp và lưu ảnh màn hình ngay lập tức vào thư mục `output_qr/snapshots/`.

---

## 6. Định Dạng Nhật Ký Sự Kiện JSONL

Các sự kiện giải mã thành công được lưu vào `output_qr/qr_events.jsonl`:
```json
{
  "timestamp": "2026-10-25T19:18:47.837859+07:00",
  "video_timestamp_sec": 2.533,
  "text": "https://github.com/warlord/RobotNongNghiep",
  "corners": [[244.0, 524.0], [476.0, 524.0], [476.0, 756.0], [244.0, 756.0]],
  "backend": "zxing-cpp",
  "format": "QR Code"
}
```
- **Cơ chế chống ghi lặp (Deduplication)**: Mặc định cooldown là `2.0` giây (`--dedup-cooldown 2.0`). Nếu cùng một mã QR nằm trong tầm nhìn camera suốt 10 giây liên tục, nó chỉ ghi log mỗi 2.0 giây một lần thay vì ghi tràn 300 dòng trùng lặp.

---

## 7. Đo Đạc Hiệu Năng Thực Tế (Benchmark)
- **Môi trường chạy**: Linux (Kernel 6.8 x86_64), Intel Core CPU, Python 3.10.12.
- **Tốc độ xử lý video 720×1280**:
  - Chế độ Benchmark tối đa: **94.0 - 145.6 FPS** (test video) và **46.6 FPS** (video nhà màng).
  - Chế độ Live/Sync: Tự động khóa nhịp ở **29.2 - 30.0 FPS** để người dùng quan sát trực quan mượt mà.
- **Ghi chú môi trường**: Máy chủ hiện tại không cắm webcam vật lý (`/dev/video0`), khi chạy `./run.sh --source 0` hệ thống bắt lỗi chuẩn xác và hiển thị hướng dẫn cắm camera rõ ràng.

