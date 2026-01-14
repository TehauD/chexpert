from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2

from .model import build_densenet121, pick_device
from .logging_utils import setup_logger

def _to_uint8(img_rgb: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return img

def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb: HxWx3 uint8
    cam: HxW float in [0,1]
    """
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (img_rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out

@torch.no_grad()
def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    labels = ckpt["labels"]
    model = build_densenet121(num_classes=len(labels))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, labels

def gradcam_single(
    ckpt_path: Path,
    image_path: Path,
    class_name: Optional[str] = None,
    class_index: Optional[int] = None,
    img_size: int = 320,
    device_str: str = "auto",
    out_dir: Path = Path("outputs/gradcam"),
):
    """
    Produces:
      - overlay PNG
      - raw heatmap PNG
    """
    logger = setup_logger("chexpert.gradcam")
    device = pick_device(device_str)

    out_dir.mkdir(parents=True, exist_ok=True)
    model, labels = load_checkpoint(ckpt_path, device)

    if class_name is not None:
        if class_name not in labels:
            raise ValueError(f"class_name='{class_name}' not in labels={labels}")
        target_idx = labels.index(class_name)
    elif class_index is not None:
        target_idx = int(class_index)
        if target_idx < 0 or target_idx >= len(labels):
            raise ValueError(f"class_index out of range: {target_idx}")
    else:
        # default: use top predicted class for that image
        target_idx = None

    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil = Image.open(image_path).convert("RGB")
    pil_resized = pil.resize((img_size, img_size))
    img_rgb = np.array(pil_resized).astype(np.uint8)

    x = tf(pil).unsqueeze(0).to(device)

    # Hook last DenseNet conv features (modern + stable-ish)
    # densenet.features is a sequential; last feature map comes from model.features
    feats = None
    grads = None

    def fwd_hook(_m, _in, out):
        nonlocal feats
        feats = out

    def bwd_hook(_m, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    h1 = model.features.register_forward_hook(fwd_hook)
    h2 = model.features.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)  # [1,C]
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    if target_idx is None:
        target_idx = int(np.argmax(probs))

    score = logits[0, target_idx]
    score.backward()

    h1.remove()
    h2.remove()

    if feats is None or grads is None:
        raise RuntimeError("Grad-CAM hooks did not capture features/grads (unexpected).")

    # Grad-CAM: weights = global-average-pool grads over spatial dims
    g = grads.detach()[0]         # [C,H,W]
    f = feats.detach()[0]         # [C,H,W]
    w = g.mean(dim=(1, 2))        # [C]
    cam = (w[:, None, None] * f).sum(dim=0)  # [H,W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max().clamp(min=1e-6))
    cam = cam.detach().cpu().numpy()

    # Resize cam to image
    cam_resized = cv2.resize(cam, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    overlay = overlay_heatmap(img_rgb, cam_resized, alpha=0.45)

    pred_label = labels[target_idx]
    base = f"{image_path.stem}__{pred_label}"
    out_overlay = out_dir / f"{base}__overlay.png"
    out_heat = out_dir / f"{base}__heat.png"

    Image.fromarray(overlay).save(out_overlay)
    Image.fromarray((cam_resized * 255).astype(np.uint8)).save(out_heat)

    logger.info(f"[gradcam] image={image_path}")
    logger.info(f"[gradcam] target={pred_label} | prob={probs[target_idx]:.4f}")
    logger.info(f"[gradcam] saved={out_overlay}")
    logger.info(f"[gradcam] saved={out_heat}")

    # Also log top-5 predictions for “clinical insight practice”
    top = np.argsort(-probs)[: min(5, len(labels))]
    logger.info("[pred] top:")
    for i in top:
        logger.info(f"  - {labels[i]}: {probs[i]:.4f}")

    return out_overlay, out_heat
