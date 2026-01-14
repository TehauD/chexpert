from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

@dataclass(frozen=True)
class CamViz:
    cam: np.ndarray             # float32 [H,W] in [0,1]
    heat_rgb: np.ndarray        # uint8 [H,W,3]
    overlay_rgb: np.ndarray     # uint8 [H,W,3]
    energy_in_mask: float       # 0..1 proportion of CAM energy inside mask
    notes: str

def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.min(x)
    mx = float(np.max(x))
    if mx < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx

def apply_colormap(cam01: np.ndarray) -> np.ndarray:
    """
    cam01: float [H,W] in [0,1]
    returns RGB uint8 heatmap
    """
    cam8 = np.clip(cam01 * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def resize_cam(cam01: np.ndarray, size_hw: Tuple[int,int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(cam01, (w, h), interpolation=cv2.INTER_LINEAR)

def gate_cam_to_mask(cam01: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """
    Multiply CAM by mask without renormalizing.
    mask01: {0,1} float or uint8
    """
    gated = cam01 * (mask01.astype(np.float32))
    return gated.astype(np.float32)

def cam_energy_inside_mask(cam01: np.ndarray, mask01: np.ndarray) -> float:
    cam = cam01.astype(np.float32)
    m = (mask01 > 0).astype(np.float32)
    total = float(cam.sum())
    if total < 1e-8:
        return 0.0
    inside = float((cam * m).sum())
    return inside / total

def threshold_cam(cam01: np.ndarray, thr: float) -> np.ndarray:
    thr = float(np.clip(thr, 0.0, 1.0))
    out = cam01.copy()
    out[out < thr] = 0.0
    return normalize_01(out)

def overlay(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    out = base_rgb.astype(np.float32) * (1-a) + heat_rgb.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
