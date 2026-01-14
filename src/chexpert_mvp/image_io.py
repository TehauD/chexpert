from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image

def load_image_rgb(path: Path) -> np.ndarray:
    """
    Loads an image and returns RGB uint8 array [H,W,3].
    CheXpert JPGs may be grayscale; we convert to RGB for consistent overlay.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def resize_rgb(img_rgb: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    img = Image.fromarray(img_rgb)
    img = img.resize(size, resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

def to_tensor_rgb_imagenet(img_rgb: np.ndarray, img_size: int):
    """
    Convert RGB uint8 [H,W,3] to normalized torch tensor [1,3,S,S].
    """
    import torch
    import torchvision.transforms as T

    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = tf(img_rgb).unsqueeze(0)
    return x

def alpha_blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """
    Alpha blend overlay on base: out = base*(1-a) + overlay*a
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (base_rgb.astype(np.float32) * (1-a) + overlay_rgb.astype(np.float32) * a)
    return np.clip(out, 0, 255).astype(np.uint8)
