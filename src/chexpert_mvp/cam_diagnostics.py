from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class CamDiagnostics:
    energy_inside_lungs: float
    energy_outside_lungs: float
    entropy: float
    max_activation: float
    mean_activation: float

def cam_entropy(cam01: np.ndarray) -> float:
    c = cam01.astype(np.float32)
    s = c.sum()
    if s < 1e-8:
        return 0.0
    p = (c / s).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))

def compute_diagnostics(cam01: np.ndarray, lung_mask01: np.ndarray) -> CamDiagnostics:
    cam = cam01.astype(np.float32)
    mask = (lung_mask01 > 0).astype(np.float32)

    total = cam.sum() + 1e-8
    inside = (cam * mask).sum()
    outside = total - inside

    return CamDiagnostics(
        energy_inside_lungs=float(inside / total),
        energy_outside_lungs=float(outside / total),
        entropy=cam_entropy(cam),
        max_activation=float(cam.max()),
        mean_activation=float(cam.mean()),
    )
