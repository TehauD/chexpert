from __future__ import annotations
import numpy as np
from typing import Callable, List
import cv2

def _augments(img: np.ndarray) -> List[np.ndarray]:
    h, w = img.shape[:2]
    outs = [img]
    outs.append(cv2.flip(img, 1))  # horizontal
    outs.append(cv2.GaussianBlur(img, (5,5), 0))
    outs.append(cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA))
    outs[-1] = cv2.resize(outs[-1], (w, h), interpolation=cv2.INTER_LINEAR)
    return outs

def cam_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.2) -> float:
    A = (a > thr)
    B = (b > thr)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum() + 1e-8
    return float(inter / union)

def stability_score(
    base_img: np.ndarray,
    cam_fn: Callable[[np.ndarray], np.ndarray],
    thr: float = 0.2
) -> float:
    cams = []
    for img in _augments(base_img):
        cam = cam_fn(img)
        cams.append(cam)

    # compare all cams to first
    ref = cams[0]
    ious = [cam_iou(ref, c, thr=thr) for c in cams[1:]]
    return float(np.mean(ious))
