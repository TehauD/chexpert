from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
import cv2

@dataclass(frozen=True)
class LungMaskResult:
    mask: np.ndarray          # uint8 [H,W] values {0,1}
    method: str               # "heuristic" or "model"
    notes: str                # debug notes

class LungMaskProvider(Protocol):
    def predict_mask(self, img_rgb: np.ndarray) -> LungMaskResult:
        ...

def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def _largest_components(binary: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Keep k largest connected components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return binary
    # stats: [label, x, y, w, h, area]
    areas = stats[1:, cv2.CC_STAT_AREA]
    idxs = np.argsort(-areas)[:k] + 1
    out = np.zeros_like(binary, dtype=np.uint8)
    for lab in idxs:
        out[labels == lab] = 1
    return out

class HeuristicLungMaskProvider:
    """
    Fast baseline "lung-ish" mask for gating CAMs:
    - convert to grayscale
    - CLAHE contrast normalize
    - threshold to separate body vs background
    - invert/cleanup to approximate thorax region
    - morphological operations to fill holes
    - keep largest connected components
    - optional ellipse prior (soft anatomical bias)

    This is NOT a medical segmentation model, but it is excellent at preventing
    CAM heatmaps from floating in background/air, and is a clean stepping stone
    to a real U-Net.
    """
    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int,int] = (8,8),
        morph_kernel: int = 9,
        keep_components: int = 2,
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.morph_kernel = morph_kernel
        self.keep_components = keep_components

    def predict_mask(self, img_rgb: np.ndarray) -> LungMaskResult:
        gray = _to_gray(img_rgb)

        # CLAHE to reduce scanner / exposure variation
        g = self.clahe.apply(gray)

        # Otsu threshold to separate foreground/body from background
        # Often background is near-black; body is brighter
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Heuristic: body region is the bright component; lungs are inside body.
        # We'll build a "thorax allowed region" by taking the body mask,
        # then eroding away edges to avoid borders/text, then filling holes.
        body = (th > 0).astype(np.uint8)

        # Remove thin borders / text regions by eroding then dilating (opening)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, k, iterations=1)

        # Fill holes (lungs are darker, so body mask may have holes) -> close
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k, iterations=2)

        # Keep largest component (patient body)
        body = _largest_components(body, k=1)

        # Build an "interior thorax region" by eroding body a bit
        interior = cv2.erode(body, k, iterations=2)

        # Optional: apply an ellipse prior centered in image (anatomical bias)
        h, w = interior.shape
        ell = np.zeros_like(interior, dtype=np.uint8)
        center = (w//2, int(h*0.52))
        axes = (int(w*0.33), int(h*0.28))
        cv2.ellipse(ell, center, axes, 0, 0, 360, 1, -1)

        allowed = (interior & ell).astype(np.uint8)

        # As a final cleanup: keep two biggest components (approx left/right lung regions)
        allowed = _largest_components(allowed, k=self.keep_components)

        notes = "clahe+otsu+open/close+erode+ellipse_prior+keep_components"
        return LungMaskResult(mask=allowed, method="heuristic", notes=notes)


# --- Hook for a real segmentation model later -----------------------------------------

class ModelLungMaskProvider:
    """
    Placeholder for a true segmentation model (U-Net).
    You can plug in weights later with the same interface.

    For now it raises unless you implement `self._predict`.
    """
    def __init__(self, weights_path: Optional[Path] = None, device: str = "auto"):
        self.weights_path = weights_path
        self.device = device

    def predict_mask(self, img_rgb: np.ndarray) -> LungMaskResult:
        raise NotImplementedError(
            "ModelLungMaskProvider is a hook. Use HeuristicLungMaskProvider for now, "
            "or implement a U-Net predictor here."
        )
