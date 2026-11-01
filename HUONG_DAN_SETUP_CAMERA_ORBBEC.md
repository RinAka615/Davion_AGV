# HƯỚNG DẪN CẤU HÌNH VÀ LẤY HÌNH ẢNH CAMERA ORBBEC

Tài liệu này cung cấp toàn bộ thông tin chi tiết về dòng camera hiện tại đang sử dụng trong dự án RobotNongNghiep, cấu hình kỹ thuật tối ưu và các bước để kết nối, lấy luồng hình ảnh phục vụ nhận diện luống cây / dẫn hướng robot.

---

## 1. Thông tin Model Camera

* **Tên Model chính xác**: **Orbbec Astra Pro HD Camera** (thuộc series camera 3D Orbbec Astra).
* **Nhà sản xuất**: Orbbec 3D Technology International, Inc.
* **Thông số nhận diện phần cứng (USB Hardware IDs)**:
  * **Cảm biến hình ảnh RGB (Màu)**: `ID 2bc5:0501` (`Orbbec 3D Technology International, Inc Astra Pro HD Camera`).
  * **Cảm biến đo độ sâu (Depth / IR Sensor)**: `ID 2bc5:0403` (`Orbbec Astra Depth Sensor`).
* **Chuẩn giao tiếp RGB**: **UVC (USB Video Class)** tiêu chuẩn.
* **Cổng giao tiếp hệ điều hành Linux**: `/dev/video0` (hoặc `/dev/video1`, `/dev/video2` tùy số lượng camera trên máy).
* **Độ phân giải & Tốc độ khung hình tối ưu**: **1280x720 (HD 720p) @ 30 FPS**.
* **Định dạng nén phần cứng**: **`MJPG`** (Motion JPEG).

---

## 2. Nguyên lý hoạt động & Bí quyết tối ưu quan trọng

Camera **Orbbec Astra Pro** được thiết kế tách biệt 2 phần cứng:
1. **Cảm biến đo sâu (Depth sensor)**: Cần OpenNI2/Orbbec SDK để giải mã đám mây điểm.
2. **Cảm biến hình ảnh màu (RGB Camera HD)**: Chạy hoàn toàn theo chuẩn **UVC tiêu chuẩn**.

> [!IMPORTANT]
> **Không cần cài đặt Orbbec SDK cồng kềnh**: Vì bài toán dẫn hướng robot trong nhà màng sử dụng hình ảnh màu RGB để chạy mạng YOLO Segmentation và mô hình màu CIELAB, bạn có thể đọc trực tiếp luồng camera qua OpenCV tiêu chuẩn mà **không cần cài thêm bất kỳ driver hay SDK bên thứ ba nào**.

> [!WARNING]
> **Bí quyết bắt buộc để đạt chuẩn 30 FPS mượt mà (Tránh bị giật lag / timeout)**:
> Mặc định OpenCV trên Linux sẽ yêu cầu định dạng pixel thô `YUYV`. Ở độ phân giải 1280x720, luồng `YUYV` sẽ làm nghẽn băng thông bus USB 2.0, khiến tốc độ tụt xuống chỉ còn **5–10 FPS** hoặc bị lỗi `select timeout / No space left on device`.
> **BẮT BUỘC** phải thiết lập FourCC của camera sang **`MJPG`**:
> ```python
> cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
> cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
> cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
> cap.set(cv2.CAP_PROP_FPS, 30)
> ```
> Khi kích hoạt `MJPG`, chip phần cứng tích hợp trong camera sẽ tự nén hình ảnh trước khi truyền qua USB, đảm bảo hình ảnh cực kỳ sắc nét và đạt chuẩn **30 FPS ổn định**.

---

## 3. Các bước cài đặt và cấp quyền trên Linux

### Bước 1: Cắm camera vào máy
* Cắm giắc USB của camera vào cổng USB máy tính (khuyến nghị cổng USB 3.0 màu xanh hoặc cổng USB gắn trực tiếp trên bo mạch máy tính/robot, tránh cắm qua hub USB không có nguồn phụ).

### Bước 2: Kiểm tra nhận diện thiết bị
Mở Terminal và chạy lệnh:
```bash
lsusb | grep -i 2bc5
```
Kết quả hiển thị tương tự:
```text
Bus 003 Device 016: ID 2bc5:0501 Orbbec 3D Technology International, Inc Astra Pro HD Camera
```

Kiểm tra cổng `/dev/video*`:
```bash
ls -l /dev/video*
```
Thường camera Orbbec sẽ xuất hiện tại `/dev/video0` hoặc `/dev/video1`.

### Bước 3: Cài đặt Udev Rule (Không cần chạy sudo)
Trong thư mục dự án đã tạo sẵn script cài đặt udev rules:
```bash
chmod +x setup/setup_udev.sh
./setup/setup_udev.sh
```
Script này sẽ:
1. Đăng ký udev rule cho Orbbec Astra Pro (`/etc/udev/rules.d/99-orbbec-astra.rules`).
2. Tự động thêm user hiện tại vào các nhóm `video`, `dialout`, `plugdev`.
3. Cho phép đọc/ghi dữ liệu camera trực tiếp mà không bị lỗi `Permission Denied`.

---

## 4. Hướng dẫn lấy hình ảnh Camera

### Cách 1: Sử dụng trực tiếp trên Ứng dụng RobotNongNghiep (Khuyên dùng)

1. Khởi động ứng dụng:
   ```bash
   ./start.sh
   ```
2. Trên thanh điều khiển bên phải, tại mục **"Điều khiển Video"**, bấm vào nút:
   **`📷 Mở Camera Orbbec`**
3. Ứng dụng sẽ ngay lập tức kết nối tới Camera Orbbec, hiển thị luồng video 1280x720 trực tiếp, chạy nhận diện lối đi (AI hoặc Xử lý ảnh màu) và gửi lệnh điều khiển xuống STM32 theo thời gian thực.

### Cách 2: Khởi chạy trực tiếp Camera bằng Dòng lệnh (CLI)

* **Mở giao diện UI phát trực tiếp từ camera:**
  ```bash
  ./.venv-yolo/bin/python main.py --video 0 --mode ai
  ```
  *(Nếu máy có webcam tích hợp, thay `0` bằng `1` hoặc `/dev/video1`)*

* **Chạy kiểm tra tự động không cần màn hình (Headless Mode):**
  ```bash
  ./.venv-yolo/bin/python main.py --headless --video 0 --mode ai --frames 30 --uart
  ```

### Cách 3: Chạy script kiểm tra độc lập `test_orbbec_camera.py`

File [test_orbbec_camera.py](file:///home/warlord/RobotNongNghiep/test_orbbec_camera.py) được cấu hình sẵn chuẩn MJPG 1280x720:

1. **Xem trực tiếp luồng camera trên cửa sổ test:**
   ```bash
   ./.venv-yolo/bin/python test_orbbec_camera.py
   ```
   *(Bấm phím `q` hoặc `ESC` trên cửa sổ để tắt)*

2. **Chụp thử 1 khung hình và đo benchmark FPS (Chế độ không mở GUI):**
   ```bash
   ./.venv-yolo/bin/python test_orbbec_camera.py --no-gui
   ```
   *Ảnh chụp mẫu sẽ được lưu tại `output/orbbec_live_snapshot.jpg` và in tốc độ FPS thực tế đo được.*

---

## 5. Đoạn mã Python mẫu độc lập (Copy & Chạy ngay)

Nếu bạn muốn viết một script Python riêng để lấy hình ảnh từ Camera Orbbec Astra Pro:

```python
import cv2
import time

def capture_orbbec(device_id=0):
    # 1. Khởi tạo đối tượng VideoCapture
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ Không thể kết nối tới camera tại index {device_id}!")
        return

    # 2. Cấu hình định dạng tối ưu bắt buộc cho Orbbec Astra Pro
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Đọc lại thông số thực tế camera phản hồi
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Đã kết nối Camera Orbbec Astra Pro: {width}x{height} @ {fps:.1f} FPS")

    # 3. Vòng lặp đọc khung hình
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Mất tín hiệu camera...")
                break

            # Hiển thị khung hình lên màn hình
            cv2.imshow("Orbbec Astra Pro HD Stream (Nhan 'q' de thoat)", frame)

            # Thoát khi nhấn phím 'q' hoặc ESC
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        # Giải phóng thiết bị khi tắt
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Đã giải phóng camera.")

if __name__ == "__main__":
    capture_orbbec(0)
```

---

## 6. Xử lý các lỗi thường gặp (Troubleshooting)

| Hiện tượng | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `Không thể mở thiết bị camera: 0` | Index camera sai (do laptop có webcam tích hợp) | Thử đổi index camera sang `1`, `2` hoặc truyền thẳng đường dẫn: `python test_orbbec_camera.py /dev/video1`. |
| `Permission Denied` trên `/dev/video*` | User chưa có quyền truy cập video | Chạy `./setup/setup_udev.sh` hoặc tạm thời chạy: `sudo chmod 666 /dev/video*`. |
| Tốc độ camera chỉ đạt `5–10 FPS`, bị lag | Chưa đặt định dạng FourCC `MJPG` | Đảm bảo có dòng lệnh: `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))`. |
| `VIDIOC_STREAMON: No space left on device` | Băng thông cổng USB bị quá tải | Cắm camera vào cổng USB controller khác; không cắm chung bộ chia USB không nguồn với các thiết bị nặng khác. |
| Màn hình bị đen vài giây đầu khi mới mở | Cảm biến tự động phơi sáng (AE) đang thích ứng | Bỏ qua 2–3 frame đầu tiên sau khi khởi tạo để cảm biến đạt độ sáng chuẩn. |

