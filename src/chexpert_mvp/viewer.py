from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import json
import cv2
import numpy as np

from .config_io import load_config
from .logging_utils import setup_logger
from .explain import ExplainRequest, explain_one
from .lung_mask import HeuristicLungMaskProvider
from .cam_utils import normalize_01

def _collect_images(chexpert_root: Path, split: str = "valid", limit: int = 4000) -> List[Path]:
    """
    Collect image paths from CheXpert folder structure.
    split: "train" or "valid"
    """
    base = chexpert_root / split
    if not base.exists():
        raise FileNotFoundError(f"Split folder not found: {base}")

    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    # CheXpert is nested deep; rglob is ok for viewer
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
            if len(paths) >= limit:
                break
    return sorted(paths)

def _fit_to_screen(img: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def _put_hud(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    x, y = 12, 28
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,10,10), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
    return out

def _save_outputs(out_dir: Path, img_path: Path, result, cfg_labels: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    lab = result.cam_out.label_name
    meth = result.cam_out.method

    # Save overlay / heat / mask as PNG
    overlay = cv2.cvtColor(result.viz_gated.overlay_rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.cvtColor(result.viz_gated.heat_rgb, cv2.COLOR_RGB2BGR)
    base = cv2.cvtColor(result.base_rgb, cv2.COLOR_RGB2BGR)

    mask = (result.lung_mask.mask * 255).astype(np.uint8)

    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__heat.png"), heat)
    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__base.png"), base)

    meta = {
        "image": str(img_path),
        "label": lab,
        "method": meth,
        "prob": result.cam_out.prob,
        "gating": True,
        "threshold": float(result.viz_gated.notes.split("thr=")[-1].split()[0]) if "thr=" in result.viz_gated.notes else None,
        "energy_in_mask": result.viz_gated.energy_in_mask,
        "mask_method": result.lung_mask.method,
        "mask_notes": result.lung_mask.notes,
        "labels": cfg_labels,
    }
    (out_dir / f"{stem}__{lab}__{meth}__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/multitask.yaml"))
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--image", type=Path, default=None, help="Optional single image path")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gradcam"))
    args = parser.parse_args()

    logger = setup_logger("chexpert.viewer")
    cfg = load_config(args.config)

    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        imgs = [args.image]
    else:
        imgs = _collect_images(cfg.chexpert_root, split=args.split, limit=args.limit)
    if not imgs:
        raise RuntimeError("No images found. Check dataset path and split selection.")

    logger.info(f"[viewer] images={len(imgs)} split={args.split}")
    logger.info("[viewer] Controls: N/P next/prev | G gating | M method | [ ] threshold | - = opacity | 1..9 label | T auto-target | S save | Q quit")

    idx = 0
    gating = True
    method = "scorecam"
    thr = 0.15
    alpha = 0.45
    target_index: Optional[int] = None  # None means auto-top

    lung_provider = HeuristicLungMaskProvider()

    cv2.namedWindow("CheXpert Lung-Aware CAM Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CheXpert Lung-Aware CAM Viewer", 1400, 900)

    while True:
        img_path = imgs[idx]

        try:
            req = ExplainRequest(
                image_path=img_path,
                ckpt_path=args.ckpt,
                labels=cfg.labels,
                method=method,
                target_index=target_index,
                use_lung_gating=gating,
                cam_threshold=thr,
                alpha=alpha,
                img_size=cfg.img_size,
            )
            result = explain_one(req, lung_provider=lung_provider)
        except Exception as e:
            logger.warning(f"[warn] failed on {img_path}: {type(e).__name__}: {e}")
            idx = (idx + 1) % len(imgs)
            continue

        # Compose a side-by-side view: base | raw overlay | gated overlay | mask
        base = cv2.cvtColor(result.base_rgb, cv2.COLOR_RGB2BGR)
        raw = cv2.cvtColor(result.viz_raw.overlay_rgb, cv2.COLOR_RGB2BGR)
        gated_img = cv2.cvtColor(result.viz_gated.overlay_rgb, cv2.COLOR_RGB2BGR)
        mask = (result.lung_mask.mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # match heights
        h = base.shape[0]
        def _resize_h(img):
            if img.shape[0] == h:
                return img
            scale = h / img.shape[0]
            w = int(img.shape[1] * scale)
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        raw = _resize_h(raw)
        gated_img = _resize_h(gated_img)
        mask_bgr = _resize_h(mask_bgr)

        panel = np.concatenate([base, raw, gated_img, mask_bgr], axis=1)
        panel = _fit_to_screen(panel)

        hud = [
            f"Image {idx+1}/{len(imgs)}: {img_path.name}",
            f"Method: {result.cam_out.method} | Target: {result.cam_out.label_name} | Prob: {result.cam_out.prob:.3f}",
            f"Gating: {gating} | thr={thr:.2f} | alpha={alpha:.2f}",
            f"Energy inside lung-mask (raw/gated): {result.viz_raw.energy_in_mask:.3f} / {result.viz_gated.energy_in_mask:.3f}",
            f"Mask: {result.lung_mask.method} ({result.lung_mask.notes})",
            "Keys: N/P | G | M | [ ] | - = | 1..9 | T | S | Q",
        ]
        if result.probs is not None and result.prob_labels is not None:
            topk = min(5, len(result.prob_labels))
            idxs = np.argsort(-result.probs)[:topk]
            top_str = " | ".join(
                f"{result.prob_labels[i]}={result.probs[i]:.3f}" for i in idxs
            )
            hud.insert(2, f"Top-{topk} likelihoods: {top_str}")
        panel = _put_hud(panel, hud)

        cv2.imshow("CheXpert Lung-Aware CAM Viewer", panel)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # ESC
            break
        if key in (ord('n'), ord('N')):
            idx = (idx + 1) % len(imgs)
        elif key in (ord('p'), ord('P')):
            idx = (idx - 1) % len(imgs)
        elif key in (ord('g'), ord('G')):
            gating = not gating
        elif key in (ord('m'), ord('M')):
            method = "gradcam++" if method == "scorecam" else "scorecam"
        elif key == ord('['):
            thr = max(0.0, thr - 0.02)
        elif key == ord(']'):
            thr = min(1.0, thr + 0.02)
        elif key == ord('-'):
            alpha = max(0.0, alpha - 0.05)
        elif key == ord('='):
            alpha = min(1.0, alpha + 0.05)
        elif key in (ord('t'), ord('T')):
            target_index = None
        elif key in (ord('s'), ord('S')):
            _save_outputs(args.out_dir, img_path, result, cfg.labels)
            logger.info(f"[save] -> {args.out_dir}")
        elif ord('1') <= key <= ord('9'):
            k = key - ord('1')
            if k < len(cfg.labels):
                target_index = k

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
