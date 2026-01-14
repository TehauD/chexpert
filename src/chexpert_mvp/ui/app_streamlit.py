from __future__ import annotations

from pathlib import Path
import io
import sys
from typing import Optional

# Ensure local package imports work when running Streamlit directly.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import cv2
from PIL import Image

from chexpert_mvp.explain import load_checkpoint_model
from chexpert_mvp.image_io import to_tensor_rgb_imagenet
from chexpert_mvp.cam_methods import gradcam_pp, scorecam
from chexpert_mvp.cam_utils import (
    normalize_01,
    resize_cam,
    threshold_cam,
    overlay,
)
from chexpert_mvp.config import TrainConfig

st.set_page_config(
    page_title="CheXpert Explainability Studio",
    layout="wide",
)

st.title("CheXpert Explainability Studio")
st.caption("Research-grade visualization. Not for clinical diagnosis.")

_COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "gray": None,
}


@st.cache_resource
def _load_model_cached(ckpt_path: str):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model, labels, device = load_checkpoint_model(ckpt)
    return model, labels, device


def _load_image_from_upload(upload) -> Optional[np.ndarray]:
    if upload is None:
        return None
    img = Image.open(upload).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_image_from_path(path_str: str) -> Optional[np.ndarray]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _img_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()



def _apply_colormap(cam01: np.ndarray, cmap_name: str) -> np.ndarray:
    cam8 = np.clip(cam01 * 255.0, 0, 255).astype(np.uint8)
    if cmap_name == "gray":
        return np.stack([cam8, cam8, cam8], axis=2)
    cmap = _COLORMAPS[cmap_name]
    heat = cv2.applyColorMap(cam8, cmap)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _normalize_cam(cam01: np.ndarray, gamma: float, focus_pct: float, blur_px: int) -> np.ndarray:
    cam = normalize_01(cam01)
    if gamma != 1.0:
        cam = np.power(cam, gamma)
    if focus_pct > 0.0:
        thr = np.percentile(cam, 100.0 - focus_pct)
        cam = np.clip((cam - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
    if blur_px > 0:
        k = 2 * int(blur_px) + 1
        cam = cv2.GaussianBlur(cam, (k, k), 0)
        cam = normalize_01(cam)
    return cam.astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    largest = int(np.argmax(counts))
    return (labels == largest).astype(np.uint8) * 255


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def _body_masks(img_rgb: np.ndarray, dilate_px: int, soft_blur_px: int) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _fill_holes(mask)
    mask = _largest_component(mask)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    hard = (mask > 0).astype(np.float32)
    if soft_blur_px > 0:
        k = 2 * int(soft_blur_px) + 1
        soft = cv2.GaussianBlur(hard, (k, k), 0)
        soft = soft / max(float(soft.max()), 1e-6)
    else:
        soft = hard
    return hard, soft


def _label_options(model_labels: list[str]) -> tuple[list[str], dict[str, Optional[str]]]:
    all_labels = TrainConfig().labels
    options: list[str] = ["Auto (top)"]
    option_to_label: dict[str, Optional[str]] = {"Auto (top)": None}

    for lab in all_labels:
        if lab in model_labels:
            options.append(lab)
            option_to_label[lab] = lab
        else:
            display = f"{lab} (not in model)"
            options.append(display)
            option_to_label[display] = None

    for lab in model_labels:
        if lab not in all_labels:
            options.append(lab)
            option_to_label[lab] = lab

    return options, option_to_label

def _run_analysis(
    base_rgb: np.ndarray,
    model,
    labels,
    device,
    method: str,
    target_index: Optional[int],
    thr: float,
    alpha: float,
    img_size: int,
    max_scorecam_channels: int,
    invert_heatmap: bool,
    gate_mode: str,
    body_dilate_px: int,
    body_blur_px: int,
    colormap: str,
    cam_gamma: float,
    cam_focus_pct: float,
    cam_blur_px: int,
    auto_tune: bool,
):
    x = to_tensor_rgb_imagenet(base_rgb, img_size=img_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    if target_index is None:
        target_index = int(np.argmax(probs))

    if method == "gradcam++":
        cam_out = gradcam_pp(model, x, labels, target_index=target_index, device=device)
    elif method == "scorecam":
        cam_out = scorecam(
            model,
            x,
            labels,
            target_index=target_index,
            max_channels=max_scorecam_channels,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if auto_tune:
        strength = float(np.clip(cam_out.prob, 0.0, 1.0))
        cam_gamma = float(np.clip(cam_gamma + 0.6 * strength, 0.3, 3.0))
        cam_focus_pct = float(np.clip(cam_focus_pct + 10.0 * strength, 0.0, 40.0))
        cam_blur_px = int(np.clip(cam_blur_px + int(round(2.0 * strength)), 0, 15))
        thr = float(np.clip(thr + 0.08 * strength, 0.0, 1.0))
        alpha = float(np.clip(alpha + 0.12 * strength, 0.0, 1.0))

    cam01 = _normalize_cam(cam_out.cam, cam_gamma, cam_focus_pct, cam_blur_px)
    if invert_heatmap:
        cam01 = 1.0 - cam01
    cam01 = resize_cam(cam01, size_hw=(base_rgb.shape[0], base_rgb.shape[1]))

    body_hard, body_soft = _body_masks(base_rgb, body_dilate_px, body_blur_px)
    gates = {
        "Raw": cam01,
        "Body hard": cam01 * body_hard,
        "Body soft": cam01 * body_soft,
    }
    if gate_mode not in gates:
        gate_mode = "Raw"

    variants: dict[str, dict[str, np.ndarray]] = {}
    for name, gated_cam in gates.items():
        cam_thr = threshold_cam(gated_cam, thr)
        heat = _apply_colormap(cam_thr, colormap)
        over = overlay(base_rgb, heat, alpha=alpha)
        variants[name] = {"heat": heat, "overlay": over}

    return {
        "cam_out": cam_out,
        "probs": probs,
        "labels": labels,
        "variants": variants,
        "gate_mode": gate_mode,
        "body_mask": body_hard,
    }


with st.sidebar:
    st.header("Case Controls")
    ckpt_path = st.text_input(
        "Checkpoint path",
        value="outputs/runs/run_20260111_111317/best.pt",
        help="Path to a trained CheXpert checkpoint (best.pt or last.pt).",
    )

    image_source = st.radio("Image source", ["Upload", "Path"], horizontal=True)
    upload = None
    image_path = ""
    if image_source == "Upload":
        upload = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
    else:
        image_path = st.text_input("Image path", value="")

    st.subheader("Explainability")
    method = st.selectbox("Method", ["scorecam", "gradcam++"], index=0)
    thr = st.slider("Threshold", 0.0, 1.0, 0.08, 0.01)
    alpha = st.slider("Opacity", 0.0, 1.0, 0.5, 0.01)

    st.subheader("Targets")
    target_label = "Auto (top)"
    topk = st.slider("Top-K likelihoods", 3, 10, 5, 1)

    with st.expander("Advanced settings", expanded=False):
        img_size = st.slider("Model input size", 224, 512, 384, 16)
        max_scorecam_channels = st.slider("Score-CAM max channels", 16, 256, 128, 16)
        invert_heatmap = st.checkbox("Invert heatmap", value=False)
        colormap = st.selectbox(
            "Colormap",
            options=list(_COLORMAPS.keys()),
            index=list(_COLORMAPS.keys()).index("inferno"),
        )
        st.markdown("---")
        st.caption("Gating")
        gate_mode = st.selectbox("Gate mode", ["Raw", "Body hard", "Body soft"], index=1)
        show_all_gates = st.checkbox("Show all gate modes", value=True)
        body_dilate_px = st.slider("Body mask dilation (px)", 0, 40, 8, 2)
        body_blur_px = st.slider("Body mask blur (px)", 0, 25, 5, 2)
        show_body_mask = st.checkbox("Show body mask", value=False)
        st.markdown("---")
        st.caption("Heatmap shaping")
        cam_gamma = st.slider("Heatmap gamma", 0.3, 3.0, 2.2, 0.1)
        cam_focus_pct = st.slider("Focus top %", 0.0, 40.0, 20.0, 1.0)
        cam_blur_px = st.slider("Heatmap blur (px)", 0, 15, 5, 1)
        auto_tune = st.checkbox("Auto-tune by probability", value=True)
        st.markdown("---")
        st.caption("Display")
        show_overlay = st.checkbox("Show overlay", value=True)
        show_heat_raw = st.checkbox("Show heatmap", value=True)

    run = st.button("Run analysis", type="primary")

model = labels = device = None
load_error = None
if ckpt_path:
    try:
        model, labels, device = _load_model_cached(ckpt_path)
    except Exception as exc:
        load_error = exc

if labels:
    st.sidebar.caption(f"Model labels detected: {len(labels)}")
    label_options, option_to_label = _label_options(labels)
    target_label = st.sidebar.selectbox(
        "Target label",
        options=label_options,
        index=0,
    )

if load_error:
    st.error(str(load_error))

base_rgb = None
if image_source == "Upload":
    base_rgb = _load_image_from_upload(upload)
else:
    base_rgb = _load_image_from_path(image_path)

if base_rgb is None:
    st.info("Load an image to begin analysis.")
    st.stop()

if model is None:
    st.warning("Model not loaded yet. Check the checkpoint path.")
    st.stop()

selected_label = None
if labels:
    selected_label = option_to_label.get(target_label, None)
    if selected_label is None and target_label != "Auto (top)":
        st.sidebar.warning("Selected label is not in this checkpoint; using Auto.")
if selected_label and labels:
    target_index = labels.index(selected_label)
else:
    target_index = None

if run:
    with st.spinner("Running model and generating CAMs..."):
        result = _run_analysis(
            base_rgb=base_rgb,
            model=model,
            labels=labels,
            device=device,
            method=method,
            target_index=target_index,
            thr=thr,
            alpha=alpha,
            img_size=img_size,
            max_scorecam_channels=max_scorecam_channels,
            invert_heatmap=invert_heatmap,
            gate_mode=gate_mode,
            body_dilate_px=body_dilate_px,
            body_blur_px=body_blur_px,
            colormap=colormap,
            cam_gamma=cam_gamma,
            cam_focus_pct=cam_focus_pct,
            cam_blur_px=cam_blur_px,
            auto_tune=auto_tune,
        )
    st.session_state["last_result"] = result
    st.session_state["last_base"] = base_rgb

if "last_result" not in st.session_state:
    st.info("Press 'Run analysis' to generate a report.")
    st.stop()

result = st.session_state["last_result"]
base_rgb = st.session_state["last_base"]

cam_out = result["cam_out"]
probs = result["probs"]
labels = result["labels"]

st.subheader("Clinical likelihood overview")

probs_sorted = np.argsort(-probs)
rows = []
for idx in probs_sorted[:topk]:
    rows.append({"label": labels[idx], "prob": float(probs[idx])})

tab_overview, tab_heatmap, tab_exports = st.tabs(
    ["Overview", "Heatmap", "Exports"]
)

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with right:
        st.metric("Primary finding", cam_out.label_name)
        st.metric("Primary probability", f"{cam_out.prob:.3f}")
        st.metric("Target label", cam_out.label_name if selected_label else "Auto")

with tab_heatmap:
    panels = []
    panels.append((base_rgb, "Original radiograph"))
    variants = result["variants"]
    if show_all_gates:
        gate_names = list(variants.keys())
    else:
        gate_names = [result["gate_mode"]]
    for name in gate_names:
        if show_overlay:
            panels.append((variants[name]["overlay"], f"{name} overlay"))
        if show_heat_raw:
            panels.append((variants[name]["heat"], f"{name} heatmap"))
    if show_body_mask:
        mask_vis = (result["body_mask"] * 255).astype(np.uint8)
        panels.append((mask_vis, "Body mask"))

    if panels:
        cols = st.columns(min(3, len(panels)))
        for i, (img, caption) in enumerate(panels):
            cols[i % len(cols)].image(img, caption=caption, use_container_width=True)

with tab_exports:
    export_gate = st.selectbox(
        "Export gate mode",
        options=list(result["variants"].keys()),
        index=list(result["variants"].keys()).index(result["gate_mode"]),
    )
    export_cols = st.columns(3)
    export_cols[0].download_button(
        "Download base image",
        data=_img_bytes(base_rgb),
        file_name="base.png",
        mime="image/png",
    )
    export_cols[1].download_button(
        "Download heatmap overlay",
        data=_img_bytes(result["variants"][export_gate]["overlay"]),
        file_name="overlay.png",
        mime="image/png",
    )
    export_cols[2].download_button(
        "Download heatmap",
        data=_img_bytes(result["variants"][export_gate]["heat"]),
        file_name="heatmap.png",
        mime="image/png",
    )
