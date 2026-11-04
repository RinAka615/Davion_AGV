# HƯỚNG DẪN DÀNH CHO AI ANTIGRAVITY: HUẤN LUYỆN YOLO-FASTEST V2 ĐA NHIỆM TỔNG HỢP (DÒ ĐƯỜNG + QUÉT QR) NCNN INT8

Tài liệu này cung cấp quy trình vận hành tiêu chuẩn (SOP) và kịch bản prompt để **AI Antigravity** tự động thực hiện từ A-Z: Nhận các video thực tế (cả video dò luống bạt đen lẫn video quay gắt có mã QR) -> Tự động trích xuất & gán nhãn đa nhiệm -> Huấn luyện mô hình YOLO-Fastest V2 2-trong-1 (Kênh 0: Đường bạt, Kênh 1: Bảng mã QR) -> Lượng tử hóa NCNN INT8 -> Kiểm thử toàn diện -> Tự động đồng bộ lên Raspberry Pi 5.

Đây là **mô hình chính thức (Production Model)** đang vận hành trên xe tự hành AGV nhà màng.

---

## 1. MẪU PROMPT DÀNH CHO NGƯỜI DÙNG (USER COPY & DÁN)

Khi người dùng quay xong video mới (có thể là 1 video bao gồm cả đi trên luống và quay đầu gặp QR, hoặc 2-3 video riêng lẻ), người dùng chỉ cần copy prompt sau và gửi cho Antigravity:

> **Prompt mẫu (Dành cho 1 video):**
> ```text
> Tôi mới thêm video [TÊN_VIDEO.mp4] tại khu vực thử nghiệm mới (bao gồm cả bạt lối đi và bảng mã QR khi quay đầu).
> Hãy đọc tài liệu HUONG_DAN_TRAIN_TONG_HOP_YOLOFASTESTV2.md và thực hiện tự động toàn bộ quy trình:
> 1. Trích xuất frame, tự động gán nhãn đường bạt đen và mã QR (kèm hiệu ứng Horizontal Motion Blur mô phỏng lúc xe quay gắt).
> 2. Huấn luyện mô hình YOLO-Fastest V2 đa nhiệm 2-trong-1 (Channel 0: bat_loi_di, Channel 1: qr_code) ở chế độ Low-CPU.
> 3. Lượng tử hóa sang NCNN INT8 đè trực tiếp vào models/yolo_fastestv2_int8.* (kích thước ~109 KB).
> 4. Chạy toàn bộ 11 bài unit test điều hướng và đồng bộ SSH sang Raspberry Pi 5.
> ```

> **Prompt mẫu (Dành cho nhiều video):**
> ```text
> Tôi mới thêm các video [video1.mp4, video2.mp4...]. Hãy đọc HUONG_DAN_TRAIN_TONG_HOP_YOLOFASTESTV2.md để tự động trích xuất nhãn bạt và nhãn QR, train lại model YOLO-Fastest V2 đa nhiệm NCNN INT8, đè vào models/yolo_fastestv2_int8.* rồi push sang Pi 5 giúp tôi.
> ```

---

## 2. QUY TRÌNH THỰC HIỆN DÀNH CHO AI ANTIGRAVITY (AGENT SOP)

Khi nhận được prompt trên, Antigravity phải tuân thủ nghiêm ngặt các bước kỹ thuật sau:

### Bước 1: Khảo sát Video & Kiểm tra Môi trường
1. Kiểm tra sự tồn tại của các file video:
   ```bash
   ls -lh [TÊN_VIDEO.mp4]
   ```
2. Đảm bảo chạy bằng môi trường Python `.venv-yolo` và giới hạn tài nguyên CPU để máy tính người dùng hoàn toàn êm ái:
   ```python
   import os, torch
   try:
       os.nice(15)  # Mức ưu tiên thấp
   except Exception:
       pass
   torch.set_num_threads(2)
   ```

---

### Bước 2: Tự Động Trích Xuất & Gán Nhãn Đa Nhiệm (Pseudo-Labeling)
1. **Làm sạch khung hình:** Nếu video là bản ghi màn hình cũ có vẽ khung vàng/cyan hoặc thông số HUD, gọi hàm `clean_hud_and_green_overlay(frame)` để trả về ảnh sạch trước khi lưu.
2. **Kênh 0 - Đường Bạt Đen (`bat_loi_di`):**
   - Trích xuất nhãn polygon bạt từ các frame luống đi, lưu vào:
     `datasets/batden_multi_angle_v3/labels/{train,val}/[tên_frame].txt`.
3. **Kênh 1 - Bảng Mã QR (`qr_code`):**
   - Dùng `zxingcpp.read_barcodes()` kết hợp Bounding Box và Template Matching để xác định vị trí bảng QR.
   - Lưu mask nhị phân QR vào:
     `datasets/batden_multi_angle_v3/qr_masks/{train,val}/[tên_frame].png`.
4. **Tăng Cường Dữ Liệu Copy-Paste + Horizontal Motion Blur:**
   Để xe không bị mất dấu QR khi quay đầu tốc độ cao:
   - Cắt các patch bảng QR chuẩn (`qr_patches_bgr`).
   - Với các frame không có QR, áp dụng ngẫu nhiên tỷ lệ 45% paste patch QR vào với kích thước ngẫu nhiên (tương đương 60px..180px trên ảnh thực tế).
   - Mô phỏng vệt nhòe quang học ngang bằng bộ lọc tích chập:
     ```python
     if random.random() < 0.80:
         ksize = random.choice([3, 5, 7, 9, 11])
         kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
         kernel_mb[ksize // 2, :] = 1.0 / ksize
         patch = cv2.filter2D(patch, -1, kernel_mb)
     ```

---

### Bước 3: Kiến Trúc Mạng YOLO-Fastest V2 Đa Nhiệm (Multi-Task)
Sử dụng kiến trúc chuẩn được triển khai trong script [`train_yolofastestv2_qr_ncnn.py`](file:///home/rinaka/Downloads/RobotNongNghiep_Deploy0/train_yolofastestv2_qr_ncnn.py):
- **Backbone:** ShuffleNetV2 siêu nhẹ:
  - `stem`: ConvBNReLU (3->24) + DWConv (24->24, s=2)
  - `stage2`: 3 khối ShuffleInvertedUnit (out_c=48, s=2)
  - `stage3`: 4 khối ShuffleInvertedUnit (out_c=96, s=2)
  - `stage4`: 2 khối ShuffleInvertedUnit (out_c=128, s=2)
- **Neck (FPN / PAN):** Lateral 1x1 convs đưa về 48 kênh + nội suy song tuyến (Bilinear Upsample).
- **Head:** `DWConvBlock(48, 32, k=3)` -> `Conv2d(32, 2, 1)` -> `Sigmoid` (kích thước `320x320x2`).
  - **Channel 0:** Xác suất bạt lối đi (`bat_loi_di`).
  - **Channel 1:** Xác suất bảng mã QR (`qr_code`).

- **Hàm Loss Đa Nhiệm Cân Bằng Trọng Số:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{path}} + 1.2 \times \mathcal{L}_{\text{qr}}$$
  Trong đó:
  $$\mathcal{L}_{\text{path}} = 0.5 \times \text{BCE} + 0.5 \times \text{Dice}_{\text{path}}$$
  $$\mathcal{L}_{\text{qr}} = 0.4 \times \text{BCE} + 0.6 \times \text{Dice}_{\text{qr}}$$
  *(Hệ số 1.2 và Dice trọng số 0.6 cho QR giúp bù đắp diện tích QR nhỏ hơn nhiều so với diện tích bạt)*.

- **Transfer Learning Cực Nhanh:**
  Nạp lại toàn bộ trọng số từ checkpoint `/tmp/yolo_fastestv2_mt_fp32.pt` (kế thừa 212 tầng trọng số), huấn luyện nhanh trong **20 - 24 epochs** (khoảng 5 - 8 phút trên CPU).
  - Mục tiêu: `Path mIoU >= 93.0%`, `QR mIoU >= 40.0%`.

---

### Bước 4: Lượng Tử Hóa NCNN INT8 (Ghi Đè Trực Tiếp)
1. **Trace TorchScript FP32:**
   ```python
   traced = torch.jit.trace(export_m, dummy_320)
   traced.save("/tmp/yolo_fastestv2_mt_fp32.pt")
   ```
2. **PNNX Chuyển sang NCNN:**
   ```bash
   .venv-yolo/bin/pnnx /tmp/yolo_fastestv2_mt_fp32.pt inputshape=[1,3,320,320] fp16=0
   ```
3. **Hiệu chuẩn bảng lượng tử hóa (KL Divergence):**
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2table \
       /tmp/yolo_fastestv2_mt_fp32.ncnn.param \
       /tmp/yolo_fastestv2_mt_fp32.ncnn.bin \
       /tmp/calib_yolofastestv2_mt.txt \
       /tmp/yolo_fastestv2_mt.table \
       mean=[0.0,0.0,0.0] norm=[0.0039215686,0.0039215686,0.0039215686] \
       shape=[320,320,3] pixel=RGB thread=2 method=kl
   ```
4. **Lượng tử hóa INT8 ghi đè trực tiếp:**
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2int8 \
       /tmp/yolo_fastestv2_mt_fp32.ncnn.param \
       /tmp/yolo_fastestv2_mt_fp32.ncnn.bin \
       models/yolo_fastestv2_int8.param \
       models/yolo_fastestv2_int8.bin \
       /tmp/yolo_fastestv2_mt.table
   ```
   *(Kích thước: ~**109 KB**! Model nén INT8 nhẹ hơn 70 lần so với PyTorch FP32 ban đầu)*.

---

### Bước 5: Kiểm Thử Toàn Diện & Đồng Bộ Lên Raspberry Pi 5
1. **Chạy bộ 11 bài Unit Test tự động:**
   ```bash
   .venv-yolo/bin/python tests/test_navigation_fusion.py
   ```
   Phải đạt kết quả `ALL 11 NAVIGATION FUSION TESTS PASSED 100%!`.

2. **Kiểm tra độ trễ suy luận trên Pi 5:**
   Đảm bảo pipeline hoạt động ở tốc độ **45 - 55 FPS** (độ trễ ~18 - 22 ms).

3. **Git Commit & Push:**
   ```bash
   git add models/yolo_fastestv2_int8.* qr_scanner/detector.py train_yolofastestv2_qr_ncnn.py
   git commit -m "feat(ai-multitask): retrain yolo-fastest v2 multi-task int8 with new video"
   git push origin main
   ```

4. **Đồng bộ tự động sang Pi 5 qua SSH:**
   ```bash
   ssh -o BatchMode=yes davion@192.168.2.119 "cd ~/Davion_AGV && git fetch origin main && git reset --hard origin/main"
   ```

5. **Báo cáo hoàn tất:** Thông báo kết quả cho người dùng kèm các thông số:
   - Số lượng ảnh Train / Val.
   - Chỉ số `Path mIoU` và `QR mIoU`.
   - Dung lượng file (`109.0 KB`).
   - Sẵn sàng khởi chạy trên Pi 5 bằng lệnh `./start_combined.sh`.

