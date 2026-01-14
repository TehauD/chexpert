from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, List

import numpy as np
import torch

from .image_io import load_image_rgb, to_tensor_rgb_imagenet
from .cam_methods import gradcam_pp, scorecam, CamOutput
from .cam_utils import (
    normalize_01, apply_colormap, resize_cam, gate_cam_to_mask,
    cam_energy_inside_mask, threshold_cam, overlay, CamViz
)
from .lung_mask import HeuristicLungMaskProvider, LungMaskProvider, LungMaskResult

CamMethod = Literal["gradcam++", "scorecam"]

@dataclass(frozen=True)
class ExplainRequest:
    image_path: Path
    ckpt_path: Path
    labels: List[str]
    method: CamMethod = "scorecam"
    target_label: Optional[str] = None
    target_index: Optional[int] = None

    use_lung_gating: bool = True
    cam_threshold: float = 0.15
    alpha: float = 0.45

    img_size: int = 384
    max_scorecam_channels: int = 64

@dataclass(frozen=True)
class ExplainResult:
    base_rgb: np.ndarray
    lung_mask: LungMaskResult
    cam_out: CamOutput
    viz_gated: CamViz
    viz_raw: CamViz
    probs: Optional[np.ndarray] = None
    prob_labels: Optional[List[str]] = None

def load_checkpoint_model(ckpt_path: Path):
    import torch
    from .model import build_model, pick_device

    ckpt = torch.load(ckpt_path, map_location="cpu")
    labels = ckpt["labels"]
    backbone = ckpt.get("backbone", ckpt["config"].get("backbone", "densenet121"))
    model = build_model(backbone, num_classes=len(labels))
    model.load_state_dict(ckpt["model"], strict=True)
    device = pick_device(ckpt["config"].get("device", "auto"))
    model.to(device).eval()
    return model, labels, device

def explain_one(req: ExplainRequest, lung_provider: Optional[LungMaskProvider] = None) -> ExplainResult:
    lung_provider = lung_provider or HeuristicLungMaskProvider()

    model, ckpt_labels, device = load_checkpoint_model(req.ckpt_path)
    labels = req.labels or ckpt_labels

    base_rgb = load_image_rgb(req.image_path)
    lung_mask_res = lung_provider.predict_mask(base_rgb)

    # model input tensor (resized + normalized)
    x = to_tensor_rgb_imagenet(base_rgb, img_size=req.img_size).to(device)

    # Full prediction vector for display/diagnostics
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    # choose target
    if req.target_label is not None:
        if req.target_label not in labels:
            raise ValueError(f"target_label '{req.target_label}' not in labels={labels}")
        target_index = labels.index(req.target_label)
    elif req.target_index is not None:
        target_index = int(req.target_index)
    else:
        target_index = None  # auto (top)

    # CAM method
    if req.method == "gradcam++":
        cam_out = gradcam_pp(model, x, labels, target_index=target_index, device=device)
    elif req.method == "scorecam":
        cam_out = scorecam(model, x, labels, target_index=target_index, max_channels=req.max_scorecam_channels, device=device)
    else:
        raise ValueError(f"Unknown method: {req.method}")

    # CAM is on input size already for Score-CAM (H,W), for Grad-CAM++ it may be feature-size
    # Normalize and resize to original image size
    cam01 = normalize_01(cam_out.cam)
    cam01 = resize_cam(cam01, size_hw=(base_rgb.shape[0], base_rgb.shape[1]))

    # Raw visualization
    cam_raw_thr = threshold_cam(cam01, req.cam_threshold)
    heat_raw = apply_colormap(cam_raw_thr)
    overlay_raw = overlay(base_rgb, heat_raw, alpha=req.alpha)
    energy_raw = cam_energy_inside_mask(cam_raw_thr, lung_mask_res.mask)

    viz_raw = CamViz(
        cam=cam_raw_thr,
        heat_rgb=heat_raw,
        overlay_rgb=overlay_raw,
        energy_in_mask=energy_raw,
        notes=f"raw thr={req.cam_threshold}",
    )

    # Gated visualization
    if req.use_lung_gating:
        gated = gate_cam_to_mask(cam01, lung_mask_res.mask)
    else:
        gated = cam01

    gated_thr = threshold_cam(gated, req.cam_threshold)
    heat_g = apply_colormap(gated_thr)
    overlay_g = overlay(base_rgb, heat_g, alpha=req.alpha)
    energy_g = cam_energy_inside_mask(gated_thr, lung_mask_res.mask)

    viz_gated = CamViz(
        cam=gated_thr,
        heat_rgb=heat_g,
        overlay_rgb=overlay_g,
        energy_in_mask=energy_g,
        notes=f"gated={req.use_lung_gating} thr={req.cam_threshold} mask={lung_mask_res.method}",
    )

    return ExplainResult(
        base_rgb=base_rgb,
        lung_mask=lung_mask_res,
        cam_out=cam_out,
        viz_gated=viz_gated,
        viz_raw=viz_raw,
        probs=probs,
        prob_labels=labels,
    )
