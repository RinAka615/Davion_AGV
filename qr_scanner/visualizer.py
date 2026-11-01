"""
Visualization Engine for Real-Time QR Scanner.
Renders polygons, status banners, telemetry HUD, and aspect-ratio display.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from .detector import QRDetectionResult


class QRVisualizer:
    """
    Vẽ trực quan hóa trên khung hình:
    - Phân biệt rõ hai trạng thái:
      + Đã giải mã thành công: Viền Xanh lá cây (Green), nhãn thông tin nội dung mã.
      + Vùng QR nghi vấn chưa giải mã: Viền Cam/Hổ phách (Amber/Orange), nhãn cảnh báo.
    - Bảng thông số (Telemetry HUD) thời gian thực: FPS, nguồn video, thời gian, số lượng mã.
    - Giữ đúng tỉ lệ khung hình (Aspect Ratio), tự động điều chỉnh hiển thị cho video dọc (Portrait).
    """

    COLOR_DECODED_BORDER = (0, 255, 0)       # Xanh lá cây (BGR)
    COLOR_DECODED_BG = (0, 180, 0)           # Nền nhãn xanh lá
    COLOR_CANDIDATE_BORDER = (0, 165, 255)   # Cam / Hổ phách (BGR)
    COLOR_CANDIDATE_BG = (0, 120, 220)       # Nền nhãn cam
    COLOR_TEXT_WHITE = (255, 255, 255)       # Trắng
    COLOR_CORNER_DOT = (255, 255, 0)         # Vàng cyan cho điểm góc
    COLOR_CENTER_TARGET = (0, 255, 255)      # Vàng cyan cho tâm ngắm (Crosshair Target)
    COLOR_CENTER_DOT = (0, 0, 255)           # Chấm đỏ rực ở tâm tuyệt đối

    def __init__(self, max_display_w: int = 1280, max_display_h: int = 800):
        self.max_display_w = max_display_w
        self.max_display_h = max_display_h

    def draw_results(
        self,
        frame: np.ndarray,
        results: List[QRDetectionResult],
        hud_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Vẽ toàn bộ kết quả phát hiện và thanh HUD lên một bản sao của frame gốc.
        """
        vis_frame = frame.copy()

        # 1. Vẽ các mã QR
        for idx, res in enumerate(results):
            self._draw_single_qr(vis_frame, res, qr_idx=idx + 1)

        # 2. Vẽ Telemetry HUD trên đầu khung hình
        if hud_info:
            self._draw_hud(vis_frame, hud_info)

        return vis_frame

    def _draw_single_qr(self, img: np.ndarray, res: QRDetectionResult, qr_idx: int):
        pts = res.points.astype(np.int32).reshape((-1, 1, 2))

        # Tính toán tọa độ tâm mã QR (ưu tiên tâm giao điểm chéo xạ ảnh)
        if hasattr(res, "center") and res.center and (res.center[0] != 0.0 or res.center[1] != 0.0):
            cx, cy = int(round(res.center[0])), int(round(res.center[1]))
        else:
            cx = int(round(np.mean(res.points[:, 0])))
            cy = int(round(np.mean(res.points[:, 1])))

        size_str = f"📐 {res.width_px:.0f}×{res.height_px:.0f}px"
        center_str = f"🎯 Tâm:({cx},{cy})"

        if res.is_decoded:
            border_color = self.COLOR_DECODED_BORDER
            bg_color = self.COLOR_DECODED_BG
            label_prefix = f"#{qr_idx} [OK] "
            text_snippet = res.text[:28] + "..." if len(res.text) > 28 else res.text
            label_text = f"{label_prefix}{text_snippet} | {size_str} | {center_str}"
            thickness = 3
        else:
            border_color = self.COLOR_CANDIDATE_BORDER
            bg_color = self.COLOR_CANDIDATE_BG
            err_short = res.error if res.error else "Unreadable"
            label_text = f"#{qr_idx} [CANDIDATE: {err_short}] | {size_str} | {center_str}"
            thickness = 2

        # 1. Vẽ 2 đường chéo định tâm giao nhau tại tâm (Perspective diagonals)
        pt_tl = (int(res.points[0][0]), int(res.points[0][1]))
        pt_tr = (int(res.points[1][0]), int(res.points[1][1]))
        pt_br = (int(res.points[2][0]), int(res.points[2][1]))
        pt_bl = (int(res.points[3][0]), int(res.points[3][1]))
        diag_color = (int(border_color[0] * 0.5), int(border_color[1] * 0.5), int(border_color[2] * 0.5))
        cv2.line(img, pt_tl, pt_br, diag_color, 1, cv2.LINE_AA)
        cv2.line(img, pt_tr, pt_bl, diag_color, 1, cv2.LINE_AA)

        # 2. Vẽ đường bao tứ giác khép kín
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=thickness, lineType=cv2.LINE_AA)

        # 3. Đánh dấu 4 điểm góc
        for pt in pts:
            cv2.circle(img, (int(pt[0][0]), int(pt[0][1])), radius=4, color=self.COLOR_CORNER_DOT, thickness=-1)

        # 4. Vẽ trực quan TÂM MÃ QR (Target Crosshair + Bullseye Dot)
        cv2.circle(img, (cx, cy), radius=8, color=self.COLOR_CENTER_TARGET, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius=3, color=self.COLOR_CENTER_DOT, thickness=-1, lineType=cv2.LINE_AA)
        arm = 12
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), self.COLOR_CENTER_TARGET, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), self.COLOR_CENTER_TARGET, 2, cv2.LINE_AA)

        # Nhãn tọa độ tâm cạnh tâm ngắm
        coord_txt = f"({cx},{cy})"
        font_c = cv2.FONT_HERSHEY_SIMPLEX
        (cw, ch), _ = cv2.getTextSize(coord_txt, font_c, 0.42, 1)
        bx1 = cx + arm + 3
        by1 = cy - ch // 2 - 2
        bx2 = bx1 + cw + 6
        by2 = by1 + ch + 4
        if bx2 < img.shape[1] - 5:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), self.COLOR_CENTER_TARGET, 1)
            cv2.putText(img, coord_txt, (bx1 + 3, by2 - 3), font_c, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            bx1_l = cx - arm - cw - 9
            bx2_l = cx - arm - 3
            cv2.rectangle(img, (bx1_l, by1), (bx2_l, by2), (20, 20, 20), -1)
            cv2.rectangle(img, (bx1_l, by1), (bx2_l, by2), self.COLOR_CENTER_TARGET, 1)
            cv2.putText(img, coord_txt, (bx1_l + 3, by2 - 3), font_c, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # Tính tọa độ đặt nhãn (phía trên cạnh trên cùng)
        min_y = int(np.min(res.points[:, 1]))
        min_x = int(np.min(res.points[:, 0]))

        # Chuẩn bị kích thước hộp nhãn
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Tọa độ hộp nền nhãn
        label_y2 = max(min_y - 8, th + 10)
        label_y1 = label_y2 - th - 8
        label_x1 = max(min_x, 5)
        label_x2 = min(label_x1 + tw + 14, img.shape[1] - 5)

        # Vẽ nền nhãn có bo tròn/vát góc
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)
        cv2.rectangle(img, (label_x1, label_y1), (label_x2, label_y2), border_color, 1)

        # Ghi chữ
        text_origin = (label_x1 + 6, label_y2 - 6)
        cv2.putText(img, label_text, text_origin, font, font_scale, self.COLOR_TEXT_WHITE, font_thick, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, info: Dict[str, Any]):
        """
        Vẽ thanh telemetry HUD dạng semi-transparent overlay ở mép trên ảnh.
        """
        h, w = img.shape[:2]
        hud_h = 70

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        # Pha trộn độ trong suốt alpha = 0.65
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.line(img, (0, hud_h), (w, hud_h), (80, 80, 80), 1)

        fps = info.get("fps", 0.0)
        source = info.get("source", "Unknown")
        time_str = info.get("time_str", "")
        decoded_count = info.get("decoded_count", 0)
        candidate_count = info.get("candidate_count", 0)
        backend_name = info.get("backend", "auto")

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Dòng 1: Nguồn, Độ phân giải, Backend, FPS
        line1 = f"NGUON: {source} ({w}x{h}) | BACKEND: {backend_name} | FPS: {fps:5.1f}"
        cv2.putText(img, line1, (15, 26), font, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

        # Dòng 2: Thời gian, Thống kê, Phím tắt
        line2 = f"{time_str} | GIAI MA: {decoded_count} | CANDIDATES: {candidate_count}  [Q/ESC]: Thoat  [SPACE]: Tam dung  [S]: Chup anh"
        cv2.putText(img, line2, (15, 54), font, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    def fit_to_screen(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Thu phóng frame để vừa với kích thước màn hình máy tính mà vẫn giữ nguyên đúng tỉ lệ (Aspect Ratio).
        Rất quan trọng cho video dọc (như 720x1280).
        
        :return: (display_frame, scale_factor)
        """
        h, w = frame.shape[:2]
        scale_w = self.max_display_w / w
        scale_h = self.max_display_h / h
        scale = min(scale_w, scale_h, 1.0)  # Không phóng to vượt quá kích thước gốc nếu ảnh nhỏ hơn màn hình

        if scale < 0.999:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale
        return frame, 1.0

