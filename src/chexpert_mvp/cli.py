from __future__ import annotations
from pathlib import Path
import typer

from .config_io import load_config
from .train import train
from .explain import ExplainRequest, explain_one

app = typer.Typer(add_completion=False)

@app.command()
def fit(
    config: Path = typer.Option(Path("configs/multitask.yaml"), help="YAML config path"),
):
    cfg = load_config(config)
    run_dir = train(cfg)
    typer.echo(str(run_dir))

@app.command()
def cam(
    ckpt: Path = typer.Option(..., help="Path to best.pt or last.pt"),
    image: Path = typer.Option(..., help="Path to a CheXpert image"),
    config: Path = typer.Option(Path("configs/multitask.yaml"), help="YAML config path"),
    method: str = typer.Option("scorecam", help="scorecam | gradcam++"),
    label: str = typer.Option(None, help="Target label name (optional)"),
    index: int = typer.Option(None, help="Target label index (optional)"),
    gating: bool = typer.Option(True, help="Apply lung gating"),
    threshold: float = typer.Option(0.15, help="CAM threshold 0..1"),
    alpha: float = typer.Option(0.45, help="Overlay alpha 0..1"),
    out_dir: Path = typer.Option(Path("outputs/gradcam"), help="Output folder"),
):
    cfg = load_config(config)

    req = ExplainRequest(
        image_path=image,
        ckpt_path=ckpt,
        labels=cfg.labels,
        method=method,  # type: ignore
        target_label=label,
        target_index=index,
        use_lung_gating=gating,
        cam_threshold=threshold,
        alpha=alpha,
        img_size=cfg.img_size,
    )
    res = explain_one(req)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    lab = res.cam_out.label_name
    meth = res.cam_out.method

    import cv2
    import numpy as np

    overlay = cv2.cvtColor(res.viz_gated.overlay_rgb, cv2.COLOR_RGB2BGR)
    heat = cv2.cvtColor(res.viz_gated.heat_rgb, cv2.COLOR_RGB2BGR)
    mask = (res.lung_mask.mask * 255).astype(np.uint8)

    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__heat.png"), heat)
    cv2.imwrite(str(out_dir / f"{stem}__{lab}__{meth}__mask.png"), mask)

    typer.echo(f"Saved to {out_dir}")

@app.command()
def viewer(
    ckpt: Path = typer.Option(..., help="Path to best.pt or last.pt"),
    config: Path = typer.Option(Path("configs/multitask.yaml"), help="YAML config path"),
    split: str = typer.Option("valid", help="train|valid"),
    limit: int = typer.Option(1500, help="max images to scan"),
):
    # call module entrypoint for viewer
    import subprocess, sys
    cmd = [sys.executable, "-m", "chexpert_mvp.viewer",
           "--config", str(config),
           "--ckpt", str(ckpt),
           "--split", split,
           "--limit", str(limit)]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    app()
