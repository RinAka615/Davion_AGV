# HƯỚNG DẪN DÀNH CHO AI ANTIGRAVITY: HUẤN LUYỆN YOLO-FASTEST V2 NHẬN DIỆN MÃ QR (NCNN INT8)

Tài liệu này cung cấp quy trình vận hành tiêu chuẩn (SOP) và kịch bản prompt để **AI Antigravity** tự động thực hiện từ A-Z: Nhận video thực tế quay mã QR -> Tự động trích xuất & gán nhãn (kèm nhòe chuyển động) -> Huấn luyện mô hình YOLO-Fastest V2 chuyên nhận diện QR -> Lượng tử hóa NCNN INT8 -> Kiểm thử & Đồng bộ sang Raspberry Pi 5.

---

## 1. MẪU PROMPT DÀNH CHO NGƯỜI DÙNG (USER COPY & DÁN)

Khi người dùng quay xong video mới có mã QR (ví dụ khi robot xoay đầu nhanh hoặc ánh sáng mới), người dùng chỉ cần copy prompt sau và gửi cho Antigravity:

> **Prompt mẫu:**
> ```text
> Tôi mới thêm video [TÊN_VIDEO.mp4] chứa cảnh quay mã QR (bao gồm cả các đoạn quay nhanh bị nhòe chuyển động). 
> Hãy đọc tài liệu HUONG_DAN_TRAIN_QR_YOLOFASTESTV2.md và thực hiện tự động toàn bộ quy trình:
> 1. Trích xuất frame và gán nhãn tự động bảng mã QR từ video này (bổ sung hiệu ứng Horizontal Motion Blur mô phỏng lúc xe quay nhanh).
> 2. Huấn luyện mô hình YOLO-Fastest V2 chuyên nhận diện QR (1 class) ở chế độ Low-CPU.
> 3. Lượng tử hóa sang NCNN INT8 (kích thước ~100 KB).
> 4. Chạy kiểm thử độ nhạy phát hiện trên video và đồng bộ SSH sang Raspberry Pi 5.
> ```

---

## 2. QUY TRÌNH THỰC HIỆN DÀNH CHO AI ANTIGRAVITY (AGENT SOP)

Khi nhận được prompt trên, Antigravity phải tuân thủ nghiêm ngặt các bước kỹ thuật sau:

### Bước 1: Khảo sát và Chuẩn bị Môi trường
1. Kiểm tra video đầu vào trong thư mục gốc dự án:
   ```bash
   ls -lh [TÊN_VIDEO.mp4]
   ```
2. Đảm bảo sử dụng môi trường Python `.venv-yolo` có đầy đủ PyTorch, Torchvision, OpenCV, và công cụ PNNX:
   ```bash
   .venv-yolo/bin/python -c "import torch, cv2, zxingcpp; print('Environment OK')"
   ```
3. Đặt cờ giới hạn luồng CPU để máy người dùng không bị đơ giật trong lúc xử lý:
   ```python
   import os, torch
   os.nice(15)  # Mức ưu tiên thấp, nhường CPU cho giao diện người dùng
   torch.set_num_threads(2)
   ```

---

### Bước 2: Tự Động Trích Xuất & Gán Nhãn Pseudo-Labeling (Không cần gán tay)
1. **Đọc frame video:** Lấy các frame cách nhau 1-2 frame để đa dạng góc nhìn.
2. **Loại bỏ HUD cũ (nếu video quay từ màn hình):** Nếu frame có vạch vẽ màu xanh lá hoặc viền vàng cũ, dùng phép nội suy inpaint hoặc xóa overlay trước khi đưa vào tập train.
3. **Thuật toán tạo Mask QR tự động:**
   - Dùng thư viện `zxingcpp.read_barcodes(frame)` để tìm 4 đỉnh tọa độ `position` của mã QR.
   - Nếu mã bị nhòe nặng không đọc được text, dùng **Template Matching** hoặc mô hình checkpoint trước đó để bắt hộp QR (`x1, y1, x2, y2`) với ngưỡng tin cậy `conf >= 0.48`.
   - Lưu ảnh vào `datasets/batden_multi_angle_v3/images/{train,val}` (kích thước chuẩn `320x320`).
   - Tạo ảnh mask nhị phân tương ứng vào `datasets/batden_multi_angle_v3/qr_masks/{train,val}` (Pixel 255 = QR, Pixel 0 = Nền).

4. **Tăng Cường Dữ Liệu Chống Nhòe Khi Xe Quay Tốc Độ Cao (Motion Blur Augmentation):**
   Trong `Dataset.__getitem__`, khi bốc ảnh có QR (hoặc copy-paste patch QR vào ảnh):
   ```python
   # Mô phỏng nhòe quang học ngang do robot quay đầu gắt (Horizontal Motion Blur)
   if random.random() < 0.75:
       ksize = random.choice([3, 5, 7, 9, 11])
       kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
       kernel_mb[ksize // 2, :] = 1.0 / ksize
       img = cv2.filter2D(img, -1, kernel_mb)
   ```

---

### Bước 3: Kiến Trúc Mạng YOLO-Fastest V2 (QR Segmentation)
Sử dụng kiến trúc ShuffleNetV2 Backbone siêu nhẹ kết hợp Lightweight PAN Head:
- **Input:** `(1, 3, 320, 320)` RGB chuẩn hóa `[0, 1]`.
- **Output:** `(1, 1, 320, 320)` Sigmoid xác suất vùng QR.
- **Hàm Loss:** Kết hợp `BCEWithLogitsLoss` và `DiceLoss`:
  $$\mathcal{L} = 0.4 \times \text{BCE} + 0.6 \times \text{Dice}$$
- **Tối ưu:** Optimizer `AdamW(lr=2.8e-3)`, CosineAnnealingLR trong 20-24 epochs.
- **Transfer Learning:** Nạp lại backbone từ `/tmp/yolo_fastestv2_mt_fp32.pt` hoặc checkpoint cũ để hội tụ trong vòng dưới 5 phút.

---

### Bước 4: Lượng Tử Hóa NCNN INT8 (Chuẩn Phần Cứng Pi 5)
1. **Trace TorchScript FP32:**
   ```python
   dummy = torch.randn(1, 3, 320, 320, dtype=torch.float32)
   traced = torch.jit.trace(export_m, dummy)
   traced.save("/tmp/yolo_fastestv2_qr_fp32.pt")
   ```
2. **Chuyển đổi sang NCNN qua PNNX:**
   ```bash
   .venv-yolo/bin/pnnx /tmp/yolo_fastestv2_qr_fp32.pt inputshape=[1,3,320,320] fp16=0
   ```
3. **Hiệu chuẩn bảng lượng tử hóa (KL Divergence):**
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2table \
       /tmp/yolo_fastestv2_qr_fp32.ncnn.param \
       /tmp/yolo_fastestv2_qr_fp32.ncnn.bin \
       /tmp/calib_qr_list.txt \
       /tmp/yolo_fastestv2_qr.table \
       mean=[0.0,0.0,0.0] norm=[0.00392157,0.00392157,0.00392157] \
       shape=[320,320,3] pixel=RGB thread=2 method=kl
   ```
4. **Tạo model NCNN INT8 siêu nhẹ:**
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2int8 \
       /tmp/yolo_fastestv2_qr_fp32.ncnn.param \
       /tmp/yolo_fastestv2_qr_fp32.ncnn.bin \
       models/yolo_fastestv2_qr_int8.param \
       models/yolo_fastestv2_qr_int8.bin \
       /tmp/yolo_fastestv2_qr.table
   ```
   *(Kích thước file kết quả đạt ~105 - 110 KB)*.

---

### Bước 5: Kiểm Thử & Tự Động Đồng Bộ Lên Raspberry Pi 5
1. **Chạy kiểm thử độ nhạy phát hiện:**
   Quét thử trên các video quay tốc độ cao (ví dụ `record2.mp4`, `record3.mp4`) để đảm bảo tỷ lệ phát hiện khi quay xe đạt ít nhất 30-50% số khung hình (thay vì 0% như các thư viện barcode truyền thống).
2. **Commit và đẩy lên GitHub:**
   ```bash
   git add models/yolo_fastestv2_qr_int8.* qr_scanner/detector.py
   git commit -m "feat(qr): retrain yolo-fastest v2 qr ncnn int8 for fast rotation"
   git push origin main
   ```
3. **Đồng bộ qua SSH sang Raspberry Pi 5:**
   ```bash
   ssh -o BatchMode=yes davion@192.168.2.119 "cd ~/Davion_AGV && git fetch origin main && git reset --hard origin/main"
   ```
4. **Báo cáo kết quả rõ ràng:** Thông báo lại cho người dùng về mIoU của mô hình, số frame bắt được khi quay tốc độ cao, và lệnh chạy trên Pi 5.

