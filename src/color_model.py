"""
Module for Lab color space distribution modeling and classification scoring.
Supports multi-patch sampling, 3D histogram smoothing, and slow online adaptation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except (ImportError, Exception):
    HAS_SCIPY = False


@dataclass
class ColorPatch:
    """Represents a user-sampled color patch on a frame."""
    patch_id: int
    category: str  # 'path' (Bạt) or 'background' (Nền)
    label: str     # e.g., 'Bạt sáng', 'Bạt tối', 'Lá cây', 'Đất'
    rect: Tuple[int, int, int, int]  # (x, y, w, h)
    mean_lab: Tuple[float, float, float]
    sample_count: int


class ColorModel:
    """Models multi-modal color distribution in CIELAB space."""

    BINS = 32
    STEP = 256 // BINS  # 8

    def __init__(self):
        self.patches: List[ColorPatch] = []
        self._next_patch_id: int = 1
        
        # 3D Histograms of shape (32, 32, 32)
        self.hist_path = np.zeros((self.BINS, self.BINS, self.BINS), dtype=np.float32)
        self.hist_bg = np.zeros((self.BINS, self.BINS, self.BINS), dtype=np.float32)
        
        # Lookup table for fast scoring, shape (32, 32, 32) float32 in [0.0, 1.0]
        self.lut = np.zeros((self.BINS, self.BINS, self.BINS), dtype=np.float32)
        
        self.is_trained: bool = False
        self.online_update_enabled: bool = False
        self.online_learning_rate: float = 0.01

    def add_patch(
        self,
        frame_bgr: np.ndarray,
        rect: Tuple[int, int, int, int],
        category: str,
        label: str = ""
    ) -> Optional[ColorPatch]:
        """Extracts a patch from frame_bgr and registers it.
        
        Args:
            frame_bgr: BGR frame.
            rect: (x, y, w, h) in frame coordinates.
            category: 'path' or 'background'.
            label: Descriptive label.
        """
        x, y, w, h = rect
        H_img, W_img = frame_bgr.shape[:2]
        x1 = max(0, min(x, W_img - 1))
        y1 = max(0, min(y, H_img - 1))
        x2 = max(x1 + 1, min(x + w, W_img))
        y2 = max(y1 + 1, min(y + h, H_img))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        pixels = lab_crop.reshape(-1, 3)

        mean_l = float(np.mean(pixels[:, 0]))
        mean_a = float(np.mean(pixels[:, 1]))
        mean_b = float(np.mean(pixels[:, 2]))

        if not label:
            cat_vn = "Bạt" if category == "path" else "Nền"
            label = f"{cat_vn} #{self._next_patch_id}"

        patch = ColorPatch(
            patch_id=self._next_patch_id,
            category=category,
            label=label,
            rect=(x1, y1, x2 - x1, y2 - y1),
            mean_lab=(mean_l, mean_a, mean_b),
            sample_count=len(pixels)
        )
        self._next_patch_id += 1
        self.patches.append(patch)
        self.train(frame_bgr)
        return patch

    def remove_patch(self, patch_id: int, current_frame_bgr: Optional[np.ndarray] = None) -> bool:
        """Removes a patch by ID and re-trains if possible."""
        initial_count = len(self.patches)
        self.patches = [p for p in self.patches if p.patch_id != patch_id]
        if len(self.patches) != initial_count:
            if current_frame_bgr is not None:
                self.train(current_frame_bgr)
            else:
                self.rebuild_from_patches()
            return True
        return False

    def clear_patches(self, category: Optional[str] = None):
        """Clears all patches or patches of a specific category."""
        if category is None:
            self.patches.clear()
            self.hist_path.fill(0)
            self.hist_bg.fill(0)
            self.lut.fill(0)
            self.is_trained = False
        else:
            self.patches = [p for p in self.patches if p.category != category]
            self.rebuild_from_patches()

    def train(self, frame_bgr: np.ndarray):
        """Builds 3D Lab histograms from all registered patches using the provided frame."""
        self.hist_path.fill(0)
        self.hist_bg.fill(0)

        lab_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        H_img, W_img = frame_bgr.shape[:2]

        path_samples_total = 0
        bg_samples_total = 0

        for p in self.patches:
            x, y, w, h = p.rect
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + w), min(H_img, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_lab = lab_full[y1:y2, x1:x2]
            pixels = crop_lab.reshape(-1, 3)
            
            # Map [0..255] to [0..BINS-1]
            idx_L = np.clip(pixels[:, 0] // self.STEP, 0, self.BINS - 1)
            idx_a = np.clip(pixels[:, 1] // self.STEP, 0, self.BINS - 1)
            idx_b = np.clip(pixels[:, 2] // self.STEP, 0, self.BINS - 1)

            if p.category == "path":
                np.add.at(self.hist_path, (idx_L, idx_a, idx_b), 1.0)
                path_samples_total += len(pixels)
            else:
                np.add.at(self.hist_bg, (idx_L, idx_a, idx_b), 1.0)
                bg_samples_total += len(pixels)

        self._smooth_and_update_lut(path_samples_total, bg_samples_total)

    def rebuild_from_patches(self):
        """Recalculates LUT if patches changed without frame access."""
        path_count = sum(p.sample_count for p in self.patches if p.category == "path")
        bg_count = sum(p.sample_count for p in self.patches if p.category == "background")
        if path_count == 0:
            self.lut.fill(0)
            self.is_trained = False
        else:
            self._smooth_and_update_lut(path_count, bg_count)

    def _smooth_and_update_lut(self, path_samples: int, bg_samples: int):
        """Applies 3D smoothing to histograms and builds the score LUT."""
        if path_samples == 0:
            self.lut.fill(0)
            self.is_trained = False
            return

        # 3D smoothing to generalize to neighboring color bins
        if HAS_SCIPY:
            h_p = gaussian_filter(self.hist_path, sigma=1.0)
            h_b = gaussian_filter(self.hist_bg, sigma=1.0) if bg_samples > 0 else np.zeros_like(self.hist_bg)
        else:
            h_p = self.hist_path.copy()
            h_b = self.hist_bg.copy()

        # Normalize to probability mass
        sum_p = np.sum(h_p)
        if sum_p > 0:
            h_p /= sum_p

        sum_b = np.sum(h_b)
        if sum_b > 0:
            h_b /= sum_b

        eps = 1e-6
        if bg_samples > 0:
            # Classification matching score: Likelihood ratio / Posterior score
            # Score near 1.0 indicates strong path match; near 0.0 indicates background
            self.lut = (h_p + eps) / (h_p + h_b + 2 * eps)
        else:
            # Only path samples provided: relative similarity to path distribution
            max_p = np.max(h_p)
            self.lut = h_p / (max_p + eps)

        self.is_trained = True

    def score_frame(self, frame_bgr: np.ndarray, roi_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Computes continuous classification score map in [0, 255].
        
        Args:
            frame_bgr: BGR frame.
            roi_rect: (x, y, w, h) optional region to compute.
            
        Returns:
            score_map: 2D uint8 array with scores in [0, 255].
        """
        if not self.is_trained:
            if roi_rect is not None:
                _, _, w, h = roi_rect
                return np.zeros((h, w), dtype=np.uint8)
            return np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)

        if roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            sub_bgr = frame_bgr[ry:ry + rh, rx:rx + rw]
        else:
            sub_bgr = frame_bgr

        lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB)
        idx_L = np.clip(lab[:, :, 0] // self.STEP, 0, self.BINS - 1)
        idx_a = np.clip(lab[:, :, 1] // self.STEP, 0, self.BINS - 1)
        idx_b = np.clip(lab[:, :, 2] // self.STEP, 0, self.BINS - 1)

        raw_score = self.lut[idx_L, idx_a, idx_b]
        score_uint8 = np.clip(raw_score * 255.0, 0, 255).astype(np.uint8)
        return score_uint8

    def update_online(self, frame_bgr: np.ndarray, safe_mask: np.ndarray, roi_offset: Tuple[int, int]):
        """Slow online adaptation from confident interior path region.
        
        Safeguards:
        - Only updates from safe_mask (eroded interior region of path).
        - Frozen if tracking is lost or confidence is low.
        """
        if not self.online_update_enabled or not self.is_trained:
            return

        if safe_mask is None or np.count_nonzero(safe_mask) < 200:
            return

        rx, ry = roi_offset
        h_mask, w_mask = safe_mask.shape
        sub_bgr = frame_bgr[ry:ry + h_mask, rx:rx + w_mask]

        lab = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[safe_mask > 0]
        if len(pixels) == 0:
            return

        # Build batch histogram
        idx_L = np.clip(pixels[:, 0] // self.STEP, 0, self.BINS - 1)
        idx_a = np.clip(pixels[:, 1] // self.STEP, 0, self.BINS - 1)
        idx_b = np.clip(pixels[:, 2] // self.STEP, 0, self.BINS - 1)

        new_hist = np.zeros_like(self.hist_path)
        np.add.at(new_hist, (idx_L, idx_a, idx_b), 1.0)
        
        sum_new = np.sum(new_hist)
        if sum_new > 0:
            new_hist /= sum_new

        # Exponential smoothing update
        alpha = self.online_learning_rate
        self.hist_path = (1.0 - alpha) * self.hist_path + alpha * new_hist

        path_count = sum(p.sample_count for p in self.patches if p.category == "path")
        bg_count = sum(p.sample_count for p in self.patches if p.category == "background")
        self._smooth_and_update_lut(path_count, bg_count)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes color model configuration and trained distribution to dictionary."""
        import base64
        lut_b64 = ""
        if self.is_trained:
            lut_b64 = base64.b64encode(self.lut.astype(np.float32).tobytes()).decode("ascii")

        return {
            "online_update_enabled": self.online_update_enabled,
            "online_learning_rate": self.online_learning_rate,
            "lut_b64": lut_b64,
            "patches": [
                {
                    "patch_id": p.patch_id,
                    "category": p.category,
                    "label": p.label,
                    "rect": list(p.rect),
                    "mean_lab": list(p.mean_lab),
                    "sample_count": p.sample_count
                }
                for p in self.patches
            ]
        }

    def from_dict(self, data: Dict[str, Any], current_frame_bgr: Optional[np.ndarray] = None):
        """Deserializes color model configuration from dictionary."""
        import base64
        self.clear_patches()
        self.online_update_enabled = data.get("online_update_enabled", False)
        self.online_learning_rate = data.get("online_learning_rate", 0.01)

        # Restore LUT if available
        lut_b64 = data.get("lut_b64", "")
        if lut_b64:
            try:
                raw_bytes = base64.b64decode(lut_b64.encode("ascii"))
                arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape((self.BINS, self.BINS, self.BINS))
                self.lut = arr.copy()
                self.is_trained = True
            except Exception as e:
                print(f"Warning: could not restore LUT from base64: {e}")

        for pdata in data.get("patches", []):
            patch = ColorPatch(
                patch_id=pdata["patch_id"],
                category=pdata["category"],
                label=pdata.get("label", ""),
                rect=tuple(pdata["rect"]),
                mean_lab=tuple(pdata.get("mean_lab", (0, 0, 0))),
                sample_count=pdata.get("sample_count", 100)
            )
            self.patches.append(patch)
            self._next_patch_id = max(self._next_patch_id, patch.patch_id + 1)

        if current_frame_bgr is not None and self.patches:
            self.train(current_frame_bgr)
        elif not self.is_trained and self.patches:
            self.rebuild_from_patches()
