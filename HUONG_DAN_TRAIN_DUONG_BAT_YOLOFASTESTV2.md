# HƯỚNG DẪN DÀNH CHO AI ANTIGRAVITY: HUẤN LUYỆN YOLO-FASTEST V2 DÒ ĐƯỜNG BẠT ĐEN (NCNN INT8)

Tài liệu này cung cấp quy trình vận hành tiêu chuẩn (SOP) và kịch bản prompt để **AI Antigravity** tự động thực hiện từ A-Z: Nhận video quay luống mới có bạt màu đen -> Tự động trích xuất & gán nhãn đường đi đa điều kiện ánh sáng -> Huấn luyện mô hình YOLO-Fastest V2 phân đoạn bạt đen -> Lượng tử hóa NCNN INT8 -> Kiểm thử & Đồng bộ sang Raspberry Pi 5.

---

## 1. MẪU PROMPT DÀNH CHO NGƯỜI DÙNG (USER COPY & DÁN)

Khi người dùng quay xong video mới tại một luống đi mới (ánh sáng chói, bạt nhăn, bóng râm cây, hoặc góc cua mới), người dùng chỉ cần copy prompt sau và gửi cho Antigravity:

> **Prompt mẫu:**
> ```text
> Tôi mới thêm video [TÊN_VIDEO.mp4] quay luống đi bạt đen tại khu vực thử nghiệm mới.
> Hãy đọc tài liệu HUONG_DAN_TRAIN_DUONG_BAT_YOLOFASTESTV2.md và thực hiện tự động toàn bộ quy trình:
> 1. Trích xuất frame và gán nhãn tự động đường bạt đen từ video này (xử lý đa dạng ánh sáng chói lóa và bóng râm).
> 2. Huấn luyện mô hình YOLO-Fastest V2 phân đoạn bạt đen (1 class: bat_loi_di) ở chế độ Low-CPU.
> 3. Lượng tử hóa sang NCNN INT8 (kích thước ~100 KB).
> 4. Chạy kiểm thử tự động thuật toán bám tâm luống và đồng bộ SSH sang Raspberry Pi 5.
> ```

---

## 2. QUY TRÌNH THỰC HIỆN DÀNH CHO AI ANTIGRAVITY (AGENT SOP)

Khi nhận được prompt trên, Antigravity phải tuân thủ nghiêm ngặt các bước kỹ thuật sau:

### Bước 1: Khảo sát Video & Môi trường Tính toán
1. Kiểm tra video đầu vào:
   ```bash
   ls -lh [TÊN_VIDEO.mp4]
   ```
2. Đảm bảo sử dụng môi trường `.venv-yolo` trên máy trạm:
   ```bash
   .venv-yolo/bin/python -c "import torch, cv2, numpy; print('Environment OK')"
   ```
3. Luôn chạy ở chế độ Low-CPU để giữ máy tính của người dùng mượt mà:
   ```python
   import os, torch
   os.nice(15)
   torch.set_num_threads(2)
   ```

---

### Bước 2: Tự Động Trích Xuất & Gán Nhãn Pseudo-Labeling (Không cần gán tay)
1. **Lọc khung hình thông minh:**
   - Đọc video, lấy các frame cách nhau 2-4 frame.
   - Loại bỏ các frame đứng yên hoặc frame bị che ống kính.
   - Nếu video là bản ghi màn hình có HUD màu vàng/xanh, dùng hàm `clean_hud_and_green_overlay()` để xóa sạch overlay trước khi gán nhãn.

2. **Thuật toán sinh Polygon Bạt Đen tự động:**
   - Sử dụng mô hình bootstrap trước đó (`models/farm_path_bootstrap.pt` hoặc model NCNN hiện có) kết hợp thuật toán phân đoạn ngưỡng thích ứng:
     ```python
     # Lấy vùng nửa dưới ảnh (nơi bạt tiếp giáp bánh xe robot)
     gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
     # Phân vùng bạt màu đen/sẫm so với nền đất sáng
     _, thresh = cv2.threshold(gray[int(H*0.4):, :], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
     # Tìm contour diện tích lớn nhất tiếp giáp đáy ảnh
     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     ```
   - Chuyển contour lớn nhất thành định dạng nhãn chuẩn YOLO Segmentation (`0 x1 y1 x2 y2 ... xn yn` với tọa độ chuẩn hóa `[0, 1]`).
   - Lưu ảnh vào `datasets/batden_multi_angle_v3/images/{train,val}` (`320x320`).
   - Lưu nhãn polygon vào `datasets/batden_multi_angle_v3/labels/{train,val}`.

3. **Tăng Cường Dữ Liệu Đa Ánh Sáng (Robust Lighting Augmentation):**
   Trong `Dataset.__getitem__`:
   - **Mô phỏng nắng gắt (Overexposure):** Tăng độ sáng `alpha = random.uniform(1.1, 1.4)`, `beta = random.uniform(15, 45)`.
   - **Mô phỏng bóng râm (Shadows):** Giảm sáng cục bộ một nửa ảnh `alpha = random.uniform(0.6, 0.85)`.
   - **Lật gương ngang (Horizontal Flip):** Đảm bảo robot bám tốt cả khúc cua trái lẫn cua phải.

---

### Bước 3: Huấn Luyện Mạng YOLO-Fastest V2 (Path Segmentation)
- **Kiến trúc:** ShuffleNetV2 Backbone + Lightweight FPN Head:
  - Input: `(1, 3, 320, 320)` RGB.
  - Output: `(1, 1, 320, 320)` Sigmoid xác suất bạt lối đi (`bat_loi_di`).
- **Hàm Loss:** Kết hợp Binary Cross-Entropy và Dice Loss:
  $$\mathcal{L}_{\text{path}} = 0.5 \times \text{BCE} + 0.5 \times \text{Dice}$$
- **Kế Thừa Trọng Số (Transfer Learning):** Nạp trọng số từ `/tmp/yolo_fastestv2_fp32.pt` hoặc `/tmp/yolo_fastestv2_mt_fp32.pt`.
- **Số Epochs:** 20 - 24 epochs với Cosine Annealing LR (`lr=2.5e-3` xuống `1e-4`).
- **Tiêu chuẩn hoàn thành:** Đạt `Path mIoU >= 93.0%`.

---

### Bước 4: Chuyển Đổi & Lượng Tử Hóa NCNN INT8
1. **Trace TorchScript:**
   ```python
   dummy = torch.randn(1, 3, 320, 320, dtype=torch.float32)
   traced = torch.jit.trace(export_model, dummy)
   traced.save("/tmp/yolo_fastestv2_path_fp32.pt")
   ```
2. **Xuất sang NCNN qua PNNX:**
   ```bash
   .venv-yolo/bin/pnnx /tmp/yolo_fastestv2_path_fp32.pt inputshape=[1,3,320,320] fp16=0
   ```
3. **Hiệu chuẩn bảng lượng tử hóa (ncnn2table):**
   Tập hợp danh sách 100-200 ảnh calib đa dạng ánh sáng trong tập train:
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2table \
       /tmp/yolo_fastestv2_path_fp32.ncnn.param \
       /tmp/yolo_fastestv2_path_fp32.ncnn.bin \
       /tmp/calib_path_list.txt \
       /tmp/yolo_fastestv2_path.table \
       mean=[0.0,0.0,0.0] norm=[0.00392157,0.00392157,0.00392157] \
       shape=[320,320,3] pixel=RGB thread=2 method=kl
   ```
4. **Tạo mô hình NCNN INT8:**
   ```bash
   /tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin/ncnn2int8 \
       /tmp/yolo_fastestv2_path_fp32.ncnn.param \
       /tmp/yolo_fastestv2_path_fp32.ncnn.bin \
       models/yolo_fastestv2_int8.param \
       models/yolo_fastestv2_int8.bin \
       /tmp/yolo_fastestv2_path.table
   ```
   *(Kích thước: ~105 - 109 KB, tốc độ suy luận trên Pi 5 CPU: ~15 - 18 ms / frame)*.

---

### Bước 5: Kiểm Thử Động Lực Học & Đồng Bộ Sang Raspberry Pi 5
1. **Chạy bộ 11 Unit Test điều hướng:**
   ```bash
   .venv-yolo/bin/python tests/test_navigation_fusion.py
   ```
   *(Phải bảo đảm 100% tests PASS, bao gồm độ lệch tâm pixel `offset`, tính toán góc nghiêng luống `angle`, và cờ gói tin UART STM32)*.
2. **Commit và đẩy mã nguồn lên GitHub:**
   ```bash
   git add models/yolo_fastestv2_int8.*
   git commit -m "feat(path): retrain yolo-fastest v2 path int8 with new greenhouse dataset"
   git push origin main
   ```
3. **Đồng bộ trực tiếp sang Raspberry Pi 5 qua SSH:**
   ```bash
   ssh -o BatchMode=yes davion@192.168.2.119 "cd ~/Davion_AGV && git fetch origin main && git reset --hard origin/main"
   ```
4. **Xác nhận trạng thái:** Kiểm tra model nạp tốt trên Pi 5 và báo cáo người dùng sẵn sàng chạy `./start_combined.sh`.

