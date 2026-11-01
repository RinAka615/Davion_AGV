# RobotNongNghiep - Hệ thống nhận diện lối đi nhà màng (Greenhouse Path Detection)

Ứng dụng thị giác máy tính truyền thống (Traditional Computer Vision) hỗ trợ robot nông nghiệp di chuyển bám luống/lối đi phủ bạt trong nhà màng trồng dưa.

---

## 1. Kiến trúc & Cấu trúc thư mục

Hệ thống được thiết kế theo kiến trúc module hóa độc lập, không hardcode đường dẫn, hỗ trợ cả giao diện đồ họa Desktop (PySide6 / PyQt5) và chế độ dòng lệnh/headless:

```
RobotNongNghiep/
├── src/
│   ├── __init__.py
│   ├── video_reader.py      # Đọc video đa luồng, seek, step frame, quản lý FPS & timestamp
│   ├── color_model.py       # Mô hình phân bố màu CIELAB 3D, lấy mẫu đa điểm, scoring LUT
│   ├── path_detector.py     # Quét lát cắt scanlines, lọc biên, kiểm tra clipping, fit đường tâm
│   ├── profile_manager.py   # Quản lý lưu/nạp cấu hình farm dạng JSON
│   ├── ui_compat.py         # Lớp tương thích Qt (PySide6 với fallback PyQt5)
│   └── main_window.py       # Giao diện Desktop hiển thị 4 view đồng thời và bảng điều khiển
├── profiles/
│   └── default_profile.json # Profile mặc định đã học sẵn mẫu bạt và nền
├── output/                  # Thư mục lưu snapshot và các frame minh chứng kiểm thử
├── tests/
│   ├── test_video_reader.py
│   ├── test_color_model.py
│   └── test_path_detector.py
├── main.py                  # Điểm khởi chạy ứng dụng (GUI / Headless)
├── requirements.txt         # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 2. Cài đặt môi trường trên Máy tính mới

> [!TIP]
> 📖 **Xem các tài liệu hướng dẫn chuyên sâu đi kèm:**
> * 🚀 **[Hướng dẫn cài đặt trên máy mới](HUONG_DAN_CAI_DAT_MAY_MOI.md)**: Chi tiết từ A-Z cách triển khai trên PC/Laptop/Mini PC mới.
> * 📷 **[Hướng dẫn cấu hình Camera Orbbec Astra Pro](HUONG_DAN_SETUP_CAMERA_ORBBEC.md)**: Cách kết nối, tối ưu 30 FPS MJPG và lấy luồng ảnh.

### Cách 1: Cài đặt tự động 1-Click (Khuyên dùng cho máy mới)
Chạy script tự động thiết lập toàn bộ hệ thống (gói APT, virtualenv, PyTorch CUDA/CPU, thư viện và udev rules):
```bash
chmod +x install.sh
./install.sh
```

### Cách 2: Cài đặt thủ công
```bash
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
pip install -r requirements.txt
./setup/setup_udev.sh
```
*(Nếu máy có GPU NVIDIA, hãy cài PyTorch CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`).*

---

## 3. Hướng dẫn sử dụng

### 3.1. Khởi động & Dừng nhanh bằng script (Khuyên dùng)
* **Khởi động ứng dụng:**
  ```bash
  ./start.sh
  ```
  *(Script sẽ tự động tìm nạp sẵn video mẫu từ thư mục `Quỳnh Robotics` nếu bạn không truyền đối số).*

* **Dừng ứng dụng lập tức:**
  ```bash
  ./stop.sh
  ```
  *(Dừng an toàn mọi tiến trình RobotNongNghiep đang chạy ngầm).*

Hoặc truyền đường dẫn video bất kỳ:
```bash
./start.sh --video "duong/dan/video.mp4"
```

### 3.2. Chạy trực tiếp qua Python
```bash
python3 main.py
```
Hoặc:
```bash
python3 main.py --video "Quỳnh Robotics [13-10-2026 20_33]/2aobor4lg6hsdvvdyhctp71bl3bxane6x8p3swey2.mp4"
```

### 3.2. Chạy kiểm thử tự động (Headless Mode)
Nếu muốn chạy kiểm thử xử lý hàng loạt frame và xuất ảnh kết quả mà không cần mở cửa sổ đồ họa:
```bash
python3 main.py --headless --video "Quỳnh Robotics [13-10-2026 20_33]/2aobor4lg6hsdvvdyhctp71bl3bxane6x8p3swey2.mp4" --frames 60
```

### 3.3. Chạy toàn bộ Unit Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Quy trình thao tác trên Giao diện

1. **Mở Video & Điều khiển:**
   * Bấm nút **"Mở File Video"** để chọn video camera hành trình từ robot.
   * Dùng các nút **Phát / Tạm dừng**, **Tiến 1F / Lùi 1F** hoặc kéo thanh trượt để tua đến frame mong muốn.
   * Có thể chọn tốc độ phát từ `0.25x` đến `2.0x`.
2. **Học màu trực tiếp trong ROI (Color Sampling):**
   * Tạm dừng video tại một khung hình rõ ràng.
   * **Lấy mẫu tự động:** Bấm nút **"Lấy mẫu từ ROI"** (hoặc bấm **"Học lại màu"** khi bảng mẫu trống), hệ thống sẽ tự động trích xuất vùng bạt ở giữa khung viền vàng ROI và học ngay lập tức.
   * **Lấy mẫu thủ công:** Giữ chuột trái kéo chọn các hình chữ nhật trên các vùng bạt (vùng sáng, vùng tối, vùng phản quang) nằm bên trong khung viền vàng ROI trên ảnh số 1.
   * Tất cả mẫu đều tập trung vào nhận diện **Bạt / Lối đi (PATH)** mà không cần lấy mẫu nền.
   * Có thể bấm **"Xóa"** từng mẫu hoặc **"Xóa tất cả mẫu"** bất cứ lúc nào.
3. **Cân chỉnh tham số:**
   * **ROI (Vùng xử lý):** Kéo các thanh trượt vị trí đỉnh, chiều cao, chiều rộng và tâm ngang để căn chỉnh khung ROI khớp với luống/lối đi.
   * **Ngưỡng phân loại bạt:** Điều chỉnh độ nhạy tách bạt (mặc định 100–128).
   * **Lọc nhiễu Opening / Closing:** Khử đốm nhỏ và lấp nếp nhăn bạt.
   * **Làm mượt đường tâm:** Điều chỉnh quán tính đường tâm (EMA) giữa các frame liên tiếp.
4. **Lưu & Nạp Profile:**
   * Nhập tên trang trại (ví dụ: `Farm_XuanLoc_01`) và bấm **"Lưu Profile"** để lưu lại toàn bộ mẫu màu và tham số thành file JSON.
   * Khi mang robot sang farm khác, chỉ cần bấm **"Nạp Profile"** hoặc bấm **"Học lại màu"** để tạo cấu hình mới.
5. **Xuất kết quả:**
   * Bấm nút **"📷 Xuất Frame minh chứng"** để lưu ảnh gốc, overlay, binary mask và heatmap vào thư mục `output/`.

---

## 5. Các lưu ý kỹ thuật quan trọng

* **Trục tham chiếu & Độ lệch:** Mốc giữa ảnh (`x = W / 2`) là trục quang học của camera. Độ lệch (`Lateral Offset`) được tính bằng **pixel** trên ảnh tại lát cắt tham chiếu (phía trước đầu xe robot). Chưa hiệu chuẩn hệ mét thực tế nên hệ thống không báo cm hay góc lái bánh xe.
* **Đường nét liền vs Nét đứt:** Đoạn đường tâm được tính toán trực tiếp từ các lát cắt đo được sẽ hiển thị **nét liền màu đỏ**. Đoạn kéo dài suy đoán về phía trước (hướng điểm tụ) hiển thị **nét đứt**.
* **Phát hiện Clipping:** Nếu mép bạt chạm sát viền ảnh/ROI, thuật toán sẽ đánh dấu là biên bị cắt (`Clipped`), không coi mép ảnh là ranh giới bạt thật.
* **Cơ chế an toàn khi mất đường:** Khi số lát cắt hợp lệ giảm dưới 25% quá 3 frame liên tiếp, trạng thái chuyển sang `Mất đường (LOST)`, toàn bộ kết quả bám đường cũ sẽ bị hủy bỏ ngay lập tức để tránh robot bám theo tín hiệu rác.

## 6. Model bạt đen trong `thamden.mp4`

Chạy app với model và ROI dành riêng cho video bạt đen trong nhà:

```bash
./start_thamden.sh
```

Lệnh này dùng `models/thamden_seg.pt`, video `thamden.mp4` và
`profiles/thamden_profile.json`. Profile đặt đỉnh ROI ở 10%, chiều cao 40%
để giữ phần còn thấy mép bạt; ROI nhà kính mặc định nhìn quá gần trong video này.
Model giữ lớp `bat_loi_di` nên tương thích với chế độ AI của app.

Có thể kiểm tra qua dòng lệnh:

```bash
./start_thamden.sh --headless --frames 60 --output output/thamden_check
```

Tham số mới `--profile <file.json>` nạp profile khi mở GUI; trong chế độ
headless, tham số này nạp cấu hình detector/ROI, còn model và inference dùng
các tham số CLI. Model cũ và profile mặc định được giữ riêng.

Dữ liệu/nhãn, script train và báo cáo nằm trong `output/thamden_training/`.
Model được fine-tune 70 epoch bằng CUDA. Các đoạn train, validation và test
của video mới được tách theo thời gian có khoảng đệm, nhưng vẫn là cùng một
cảnh quay. Nhãn là polygon ước lượng bằng mắt; kết quả trên video này chưa
chứng minh khả năng tổng quát ở farm hoặc điều kiện ánh sáng khác.

---

## 7. Model bạt đen bóng góc quay thấp trên robot (`nhamang_20261025_030235.mp4`)

Model `models/nhamang_batden_seg.pt` (kiến trúc `YOLO26n-seg`, lớp `0: bat_loi_di`)
được huấn luyện cho tấm bạt đen bóng trên nền sáng quay từ camera Orbbec Astra Pro
(`1280x720`, gắn thấp trên robot), đồng thời giữ lại dữ liệu replay của `thamden.mp4`
và lối đi nhà màng cũ. Model này đã được đặt làm **model mặc định** khi mở ứng dụng.

* **Khởi chạy với cấu hình ROI tối ưu cho bạt đen (`ROI 28%..95%`):**
  ```bash
  ./start_nhamang.sh
  ```
* **Hoặc mở giao diện bình thường (dùng trực tiếp với Camera Orbbec hoặc chọn file video):**
  ```bash
  ./start.sh
  ```
* **Kiểm tra chế độ Headless:**
  ```bash
  ./start_nhamang.sh --headless --frames 120 --output output/nhamang_check
  ```

Toàn bộ dữ liệu, nhãn polygon, script huấn luyện (`prepare_dataset.py`, `train.py`,
`evaluate.py`) và báo cáo kiểm định nằm tại `output/nhamang_20261025_training/`.


