"""
Module for agricultural greenhouse path detection, scanline analysis,
boundary clipping detection, robust centerline fitting, and tracking state management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class TrackingState(str, Enum):
    NO_MODEL = "Chưa nạp model"
    NO_COLOR_MODEL = "Chưa nạp model"  # Backward compatibility alias
    TRACKING = "Đang theo dõi"
    LOW_CONFIDENCE = "Độ tin cậy thấp"
    LOST = "Mất đường"


@dataclass
class ScanlineResult:
    """Detection result for a single horizontal scanline slice."""
    y_roi: int                     # y coordinate relative to ROI top
    y_global: int                  # y coordinate in global frame
    x_left: Optional[int] = None   # Left boundary in global frame
    x_right: Optional[int] = None  # Right boundary in global frame
    x_mid: Optional[int] = None    # Midpoint in global frame (only set if valid)
    left_clipped: bool = False     # True if left boundary touches ROI/image border
    right_clipped: bool = False    # True if right boundary touches ROI/image border
    is_valid: bool = False         # True ONLY if both boundaries are real, unclipped, and plausible
    width_px: int = 0              # Measured width in pixels
    reject_reason: str = ""        # Explanation if invalid


@dataclass
class DetectionResult:
    """Full path detection output for a single frame."""
    state: TrackingState
    confidence: float                    # Geometric path quality: 0.0 to 100.0%
    lateral_offset_px: Optional[float]   # Deviation in pixels relative to camera center at reference scanline
    offset_direction: str                # 'Lệch trái XX px', 'Lệch phải XX px', 'Đúng tâm (0 px)', 'N/A'
    ref_y_global: Optional[int]          # Reference scanline y in global frame (within valid observed range)
    scanlines: List[ScanlineResult] = field(default_factory=list)
    centerline_points: List[Tuple[int, int]] = field(default_factory=list)      # Measured range: SOLID line
    extrapolated_points: List[Tuple[int, int]] = field(default_factory=list)    # Extrapolated range: DASHED line
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (bx, by, bw, bh) in global coords
    box_center: Optional[Tuple[int, int]] = None              # (cx, cy) in global coords
    path_center: Optional[Tuple[int, int]] = None             # (px, py) on fitted centerline at ref_y
    center_method: str = "box"                                # "box" or "path"
    roi_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    binary_mask: Optional[np.ndarray] = None
    safe_interior_mask: Optional[np.ndarray] = None
    valid_scanlines_count: int = 0
    total_scanlines_count: int = 0
    poly_coeffs: Optional[np.ndarray] = None


class PathDetector:
    """Analyzes mulch corridor binary masks, identifies genuine unclipped boundaries,
    fits smooth centerlines, and manages tracking telemetry."""

    def __init__(self):
        # Centering method: 'box' (Bounding Box Midpoint) or 'path' (Centerline Path)
        self.center_method: str = "box"

        # ROI parameters (percentages of image height & width)
        self.roi_top_pct: float = 0.45       # Top of ROI at 45% from top of image
        self.roi_height_pct: float = 0.50    # Height of ROI is 50% of image height
        self.roi_width_pct: float = 1.00     # Width of ROI (30% to 100%)
        self.roi_center_x_pct: float = 0.50  # Center X position (default 50% - centered)

        # Binary mask parameters
        self.threshold: int = 128            # Score threshold if input is continuous [0..255]
        self.k_open: int = 5                 # Morphological opening kernel size
        self.k_close: int = 9                # Morphological closing kernel size

        # Scanline parameters
        self.num_scanlines: int = 24         # Number of horizontal evaluation slices
        self.min_path_width_px: int = 40     # Minimum allowed width in pixels to prevent false edges
        self.clipping_margin_px: int = 3     # Pixel margin from ROI/image border to flag clipping

        # Temporal smoothing & history
        self.smoothing_factor: float = 0.35  # Weight of new measurement (0.0 = freeze, 1.0 = instant)
        self.prev_poly_coeffs: Optional[np.ndarray] = None
        self.prev_centerline_points: Optional[List[Tuple[int, int]]] = None
        self.prev_ref_x: Optional[float] = None
        self.prev_box_center: Optional[Tuple[int, int]] = None
        self.lost_counter: int = 0
        self.max_lost_frames: int = 8

    def reset_tracking(self):
        """Resets temporal tracking history (must be called on seek, video change, or model change)."""
        self.prev_poly_coeffs = None
        self.prev_centerline_points = None
        self.prev_ref_x = None
        self.prev_box_center = None
        self.lost_counter = 0

    def compute_roi(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Calculates (rx, ry, rw, rh) from percentage settings and frame dimensions."""
        h_img, w_img = frame_shape[:2]
        ry = int(np.clip(self.roi_top_pct * h_img, 0, h_img - 20))
        rh = int(np.clip(self.roi_height_pct * h_img, 20, h_img - ry))

        rw = int(np.clip(self.roi_width_pct * w_img, 40, w_img))
        cx = int(np.clip(self.roi_center_x_pct * w_img, rw // 2, w_img - rw // 2))
        rx = int(np.clip(cx - rw // 2, 0, w_img - rw))
        return rx, ry, rw, rh

    def process(
        self,
        frame_bgr: np.ndarray,
        mask_or_score: Optional[np.ndarray],
        is_source_valid: bool = True,
        is_color_model_trained: Optional[bool] = None,
        center_method: Optional[str] = None
    ) -> DetectionResult:
        """Processes a frame and its binary mask (from YOLO or color model) to detect boundaries and centerline.
        
        Args:
            frame_bgr: Full BGR frame (H, W, 3).
            mask_or_score: 2D uint8 array (either full-frame mask or ROI-sized mask).
            is_source_valid: False if model is not loaded or inference failed.
            is_color_model_trained: Alias for is_source_valid for backward compatibility.
        """
        if is_color_model_trained is not None:
            is_source_valid = is_color_model_trained

        h_img, w_img = frame_bgr.shape[:2]
        rx, ry, rw, rh = self.compute_roi((h_img, w_img))
        camera_center_x = w_img / 2.0

        if not is_source_valid or mask_or_score is None or mask_or_score.size == 0:
            self.reset_tracking()
            return DetectionResult(
                state=TrackingState.NO_MODEL,
                confidence=0.0,
                lateral_offset_px=None,
                offset_direction="Chưa nạp model",
                ref_y_global=None,
                roi_rect=(rx, ry, rw, rh),
                binary_mask=np.zeros((rh, rw), dtype=np.uint8)
            )

        # Extract mask for ROI region
        if mask_or_score.shape == (h_img, w_img):
            raw_roi = mask_or_score[ry:ry + rh, rx:rx + rw]
        elif mask_or_score.shape == (rh, rw):
            raw_roi = mask_or_score
        else:
            raw_roi = cv2.resize(mask_or_score, (rw, rh), interpolation=cv2.INTER_NEAREST)

        # Binarize if continuous score map
        if raw_roi.dtype != np.uint8:
            raw_roi = raw_roi.astype(np.uint8)
        if np.max(raw_roi) <= 1:
            raw_roi = raw_roi * 255
        _, binary_roi = cv2.threshold(raw_roi, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological filtering to clean pepper noise and close minor tarp wrinkles
        k_o = max(1, self.k_open | 1)
        k_c = max(1, self.k_close | 1)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_o, k_o))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_c, k_c))
        clean_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, k_open)
        clean_roi = cv2.morphologyEx(clean_roi, cv2.MORPH_CLOSE, k_close)

        # Select primary corridor component within ROI
        path_mask = self._select_path_component(clean_roi)

        # Step 1: Scanline extraction with strict border clipping checks
        scanlines = self._extract_scanlines(path_mask, ry, rx, rw, w_img)
        total_scans = len(scanlines)

        # Step 2: Outlier rejection and geometric consistency
        valid_scans = self._filter_outliers([s for s in scanlines if s.is_valid])
        valid_count = len(valid_scans)
        valid_ratio = valid_count / float(max(1, total_scans))

        # Step 3: Bounding Box & Centerpoint Detection (Direct midpoint logic, no curve fitting)
        nz_count = int(cv2.countNonZero(path_mask))
        bbox_global = None
        box_center_global = None

        if nz_count >= 100:
            bx_r, by_r, bw_r, bh_r = cv2.boundingRect(path_mask)
            if bw_r >= self.min_path_width_px:
                raw_bx = rx + bx_r
                raw_by = ry + by_r
                raw_bw = bw_r
                raw_bh = bh_r
                bbox_global = (raw_bx, raw_by, raw_bw, raw_bh)
                raw_cx = raw_bx + raw_bw // 2
                raw_cy = raw_by + raw_bh // 2

                # Temporal smoothing (EMA) for box center
                beta = self.smoothing_factor
                if self.prev_box_center is not None:
                    smooth_cx = int(round((1.0 - beta) * self.prev_box_center[0] + beta * raw_cx))
                    smooth_cy = int(round((1.0 - beta) * self.prev_box_center[1] + beta * raw_cy))
                else:
                    smooth_cx, smooth_cy = raw_cx, raw_cy
                self.prev_box_center = (smooth_cx, smooth_cy)
                box_center_global = (smooth_cx, smooth_cy)

        # Clipping safety check (mask touches both borders like all-white)
        clipped_scans = [s for s in scanlines if s.left_clipped and s.right_clipped]
        is_both_clipped = (len(clipped_scans) >= max(3, int(total_scans * 0.6)))

        # Determine centering mode: 'path' (Centerline curve) vs 'box' (Bounding box midpoint)
        active_method = center_method if center_method is not None else self.center_method

        centerline_points: List[Tuple[int, int]] = []
        extrapolated_points: List[Tuple[int, int]] = []
        poly_coeffs: Optional[np.ndarray] = None
        path_center_global: Optional[Tuple[int, int]] = None
        ref_y_global: Optional[int] = None
        lateral_offset_px: Optional[float] = None
        offset_dir = "N/A"

        if is_both_clipped:
            # Mask touches borders on both sides (e.g. all white): do NOT report high confidence
            current_state = TrackingState.LOW_CONFIDENCE
            confidence = 10.0
            if box_center_global is not None:
                smooth_cx, smooth_cy = box_center_global
                ref_y_global = smooth_cy
                lateral_offset_px = float(smooth_cx - camera_center_x)
                lateral_offset_px = float(np.clip(lateral_offset_px, -w_img / 2.0, w_img / 2.0))
            offset_dir = "Cảnh báo: Bị cắt 2 biên"

        elif active_method == "path" and valid_count >= 2:
            # Mode: Centerline Polynomial Path Fitting
            pts_x = np.array([s.x_mid for s in valid_scans], dtype=np.float64)
            pts_y = np.array([s.y_global for s in valid_scans], dtype=np.float64)

            # Fit 1st or 2nd order polynomial x = f(y)
            deg = 2 if valid_count >= 6 else 1
            try:
                fit_coeffs = np.polyfit(pts_y, pts_x, deg)
                if deg == 2 and abs(fit_coeffs[0]) > 0.002:
                    fit_coeffs = np.polyfit(pts_y, pts_x, 1)
            except Exception:
                fit_coeffs = np.polyfit(pts_y, pts_x, 1)

            # Apply temporal smoothing
            if self.prev_poly_coeffs is not None and len(self.prev_poly_coeffs) == len(fit_coeffs):
                beta = self.smoothing_factor
                smoothed_coeffs = (1.0 - beta) * self.prev_poly_coeffs + beta * fit_coeffs
            else:
                smoothed_coeffs = fit_coeffs
            self.prev_poly_coeffs = smoothed_coeffs.copy()
            poly_coeffs = smoothed_coeffs

            min_meas_y = int(min(pts_y))
            max_meas_y = int(max(pts_y))

            # Sample measured points (Solid Line)
            eval_meas_ys = np.arange(min_meas_y, max_meas_y + 1, 8)
            for y_val in eval_meas_ys:
                x_val = int(np.clip(np.polyval(smoothed_coeffs, y_val), 0, w_img - 1))
                centerline_points.append((x_val, int(y_val)))

            # Sample extrapolated points (Dashed Line)
            eval_extra_top = np.arange(ry, min_meas_y, 8) if min_meas_y > ry else []
            eval_extra_bottom = np.arange(max_meas_y + 8, ry + rh, 8) if max_meas_y < ry + rh else []
            for y_val in list(eval_extra_top) + list(eval_extra_bottom):
                x_val = int(np.clip(np.polyval(smoothed_coeffs, y_val), 0, w_img - 1))
                extrapolated_points.append((x_val, int(y_val)))

            # Select reference scanline: LOWEST valid scanline with two observed edges
            ref_scan = max(valid_scans, key=lambda s: s.y_global)
            ref_y_global = ref_scan.y_global
            path_x_at_ref = float(np.polyval(smoothed_coeffs, ref_y_global))

            # Smooth lateral position
            if self.prev_ref_x is not None:
                path_x_at_ref = (1.0 - self.smoothing_factor) * self.prev_ref_x + self.smoothing_factor * path_x_at_ref
            self.prev_ref_x = path_x_at_ref

            path_center_global = (int(round(path_x_at_ref)), ref_y_global)
            lateral_offset_px = float(path_x_at_ref - camera_center_x)
            lateral_offset_px = float(np.clip(lateral_offset_px, -w_img / 2.0, w_img / 2.0))

            if abs(lateral_offset_px) <= 3.0:
                offset_dir = "Đúng tâm đường dẫn (0 px)"
            elif lateral_offset_px < -3.0:
                offset_dir = f"Lệch trái {abs(lateral_offset_px):.1f} px (Đường dẫn)"
            else:
                offset_dir = f"Lệch phải {lateral_offset_px:.1f} px (Đường dẫn)"

            if valid_count >= 5 and (valid_scans[-1].y_global - valid_scans[0].y_global) >= int(rh * 0.35):
                current_state = TrackingState.TRACKING
                self.lost_counter = 0
            else:
                current_state = TrackingState.LOW_CONFIDENCE
                self.lost_counter = 0

            span_ratio = min(1.0, (max_meas_y - min_meas_y) / float(rh * 0.70))
            confidence = float(np.clip(valid_ratio * 60.0 + span_ratio * 35.0, 10.0, 95.0))

        elif bbox_global is not None and box_center_global is not None:
            # Mode: Bounding Box Midpoint (or fallback when path fit lacks slices)
            smooth_cx, smooth_cy = box_center_global
            ref_y_global = smooth_cy
            lateral_offset_px = float(smooth_cx - camera_center_x)
            lateral_offset_px = float(np.clip(lateral_offset_px, -w_img / 2.0, w_img / 2.0))

            if abs(lateral_offset_px) <= 3.0:
                offset_dir = "Đúng tâm (0 px)"
            elif lateral_offset_px < -3.0:
                offset_dir = f"Lệch trái {abs(lateral_offset_px):.1f} px"
            else:
                offset_dir = f"Lệch phải {lateral_offset_px:.1f} px"

            raw_bh = bbox_global[3]
            raw_bw = bbox_global[2]
            if raw_bh >= int(rh * 0.20):
                current_state = TrackingState.TRACKING
                self.lost_counter = 0
            else:
                current_state = TrackingState.LOW_CONFIDENCE
                self.lost_counter = 0

            # Confidence based on box coverage and aspect
            area_ratio = min(1.0, float(nz_count) / float(max(1, raw_bw * raw_bh)))
            height_ratio = min(1.0, float(raw_bh) / float(max(1, rh)))
            confidence = float(np.clip(height_ratio * 60.0 + area_ratio * 35.0, 15.0, 95.0))
        else:
            self.lost_counter += 1
            if self.lost_counter >= self.max_lost_frames or nz_count == 0:
                current_state = TrackingState.LOST
                self.prev_box_center = None
                self.prev_poly_coeffs = None
                self.prev_centerline_points = None
                self.prev_ref_x = None
            else:
                current_state = TrackingState.LOW_CONFIDENCE
            confidence = 0.0
            ref_y_global = None
            lateral_offset_px = None
            offset_dir = "N/A (Mất đường)"

        # Safe interior mask for online adaptation
        safe_mask = self._generate_safe_interior_mask(path_mask, valid_scans, rx)

        return DetectionResult(
            state=current_state,
            confidence=confidence,
            lateral_offset_px=lateral_offset_px,
            offset_direction=offset_dir,
            ref_y_global=ref_y_global,
            scanlines=scanlines,
            centerline_points=centerline_points,
            extrapolated_points=extrapolated_points,
            bounding_box=bbox_global,
            box_center=box_center_global,
            path_center=path_center_global,
            center_method=active_method,
            roi_rect=(rx, ry, rw, rh),
            binary_mask=path_mask,
            safe_interior_mask=safe_mask,
            valid_scanlines_count=valid_count,
            total_scanlines_count=total_scans,
            poly_coeffs=poly_coeffs
        )

    def _select_path_component(self, binary_mask: np.ndarray) -> np.ndarray:
        """Selects the connected component representing the path corridor."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        if num_labels <= 1:
            return binary_mask
        if num_labels == 2:
            if stats[1, cv2.CC_STAT_AREA] >= 100:
                return binary_mask
            return np.zeros_like(binary_mask)

        h, w = binary_mask.shape
        center_x = w / 2.0
        best_label = 0
        best_score = -1.0

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 100:
                continue

            cx, cy = centroids[label]
            top = stats[label, cv2.CC_STAT_TOP]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            bottom = top + height

            dist_center = abs(cx - center_x) / (w / 2.0)
            reaches_bottom = 1.0 if bottom >= h - 10 else (bottom / float(h))
            score = (area * (reaches_bottom ** 2)) / (1.0 + dist_center * 1.5)

            if score > best_score:
                best_score = score
                best_label = label

        if best_label > 0:
            out_mask = np.zeros_like(binary_mask)
            out_mask[labels == best_label] = 255
            return out_mask
        return binary_mask

    def _extract_scanlines(
        self,
        mask_roi: np.ndarray,
        ry_global: int,
        rx_global: int,
        rw: int,
        w_img: int
    ) -> List[ScanlineResult]:
        """Extracts horizontal slices and applies strict edge clipping detection."""
        h_roi, _ = mask_roi.shape
        scanlines: List[ScanlineResult] = []
        ys = np.linspace(8, h_roi - 8, self.num_scanlines, dtype=int)

        margin = self.clipping_margin_px

        for y_r in ys:
            row = mask_roi[y_r, :] > 0
            y_g = ry_global + y_r
            indices = np.where(row)[0]

            if len(indices) == 0:
                scanlines.append(ScanlineResult(
                    y_roi=int(y_r),
                    y_global=int(y_g),
                    is_valid=False,
                    reject_reason="Không có điểm bạt"
                ))
                continue

            # Identify contiguous runs of path pixels via vectorized diff
            splits = np.where(np.diff(indices) > 1)[0]
            if len(splits) == 0:
                runs = [(int(indices[0]), int(indices[-1]))]
            else:
                starts = np.insert(indices[splits + 1], 0, indices[0])
                ends = np.append(indices[splits], indices[-1])
                runs = list(zip(starts.tolist(), ends.tolist()))

            # Select run closest to middle of ROI
            mid_roi = rw / 2.0
            best_run = min(runs, key=lambda r: abs((r[0] + r[1]) / 2.0 - mid_roi))
            x_start, x_end = best_run
            width = x_end - x_start

            # Strict clipping detection: touching ROI or image borders
            left_clipped = bool(x_start <= margin or (rx_global + x_start) <= margin)
            right_clipped = bool(x_end >= (rw - 1 - margin) or (rx_global + x_end) >= (w_img - 1 - margin))

            x_left_g = rx_global + x_start
            x_right_g = rx_global + x_end

            # A scanline is valid ONLY IF both borders are genuine unclipped edges
            if left_clipped and right_clipped:
                is_valid = False
                reject_reason = "Cắt cả hai biên"
                x_mid = None
            elif left_clipped:
                is_valid = False
                reject_reason = "Cắt biên trái"
                x_mid = None
            elif right_clipped:
                is_valid = False
                reject_reason = "Cắt biên phải"
                x_mid = None
            elif width < self.min_path_width_px:
                is_valid = False
                reject_reason = f"Bề rộng quá hẹp ({width} px)"
                x_mid = None
            else:
                is_valid = True
                reject_reason = ""
                x_mid = int((x_left_g + x_right_g) / 2)

            scanlines.append(ScanlineResult(
                y_roi=int(y_r),
                y_global=int(y_g),
                x_left=int(x_left_g),
                x_right=int(x_right_g),
                x_mid=x_mid,
                left_clipped=left_clipped,
                right_clipped=right_clipped,
                is_valid=is_valid,
                width_px=int(width),
                reject_reason=reject_reason
            ))

        return scanlines

    def _filter_outliers(self, valid_scans: List[ScanlineResult]) -> List[ScanlineResult]:
        """Filters sudden jumps or width anomalies among scanlines with two observed edges."""
        if len(valid_scans) <= 2:
            return valid_scans

        # Sort by y coordinate
        sorted_scans = sorted(valid_scans, key=lambda s: s.y_global)
        mids = [s.x_mid for s in sorted_scans if s.x_mid is not None]
        if not mids:
            return []

        # Check for abrupt lateral jumps: corridor center cannot jump > 80px between adjacent slices
        filtered: List[ScanlineResult] = [sorted_scans[0]]
        for i in range(1, len(sorted_scans)):
            curr = sorted_scans[i]
            prev = filtered[-1]
            if curr.x_mid is not None and prev.x_mid is not None:
                dy = max(1, curr.y_global - prev.y_global)
                dx = abs(curr.x_mid - prev.x_mid)
                # Maximum plausible slope dx/dy is ~1.5
                if (dx / float(dy)) > 1.8:
                    curr.is_valid = False
                    curr.reject_reason = "Điểm ngoại lai (bước nhảy quá lớn)"
                    continue
            filtered.append(curr)

        return [s for s in filtered if s.is_valid]

    def _generate_safe_interior_mask(
        self,
        path_mask: np.ndarray,
        valid_scans: List[ScanlineResult],
        rx_global: int
    ) -> Optional[np.ndarray]:
        """Generates an eroded interior mask safely inside valid boundaries for online tuning."""
        if not valid_scans or len(valid_scans) < 3:
            return None

        h, w = path_mask.shape
        safe = np.zeros_like(path_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        eroded = cv2.erode(path_mask, kernel)

        # Restrict eroded mask strictly between min and max observed scanlines
        min_y = min(s.y_roi for s in valid_scans)
        max_y = max(s.y_roi for s in valid_scans)
        safe[min_y:max_y, :] = eroded[min_y:max_y, :]
        return safe if np.count_nonzero(safe) > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes detector parameters for farm profile persistence."""
        return {
            "center_method": self.center_method,
            "roi_top_pct": self.roi_top_pct,
            "roi_height_pct": self.roi_height_pct,
            "roi_width_pct": self.roi_width_pct,
            "roi_center_x_pct": self.roi_center_x_pct,
            "threshold": self.threshold,
            "k_open": self.k_open,
            "k_close": self.k_close,
            "num_scanlines": self.num_scanlines,
            "min_path_width_px": self.min_path_width_px,
            "smoothing_factor": self.smoothing_factor
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserializes detector parameters from farm profile dictionary."""
        self.center_method = str(data.get("center_method", self.center_method))
        self.roi_top_pct = float(data.get("roi_top_pct", self.roi_top_pct))
        self.roi_height_pct = float(data.get("roi_height_pct", self.roi_height_pct))
        self.roi_width_pct = float(data.get("roi_width_pct", self.roi_width_pct))
        self.roi_center_x_pct = float(data.get("roi_center_x_pct", self.roi_center_x_pct))
        self.threshold = int(data.get("threshold", self.threshold))
        self.k_open = int(data.get("k_open", self.k_open))
        self.k_close = int(data.get("k_close", self.k_close))
        self.num_scanlines = int(data.get("num_scanlines", self.num_scanlines))
        self.min_path_width_px = int(data.get("min_path_width_px", self.min_path_width_px))
        self.smoothing_factor = float(data.get("smoothing_factor", self.smoothing_factor))
        self.reset_tracking()
