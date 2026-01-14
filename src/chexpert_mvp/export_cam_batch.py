from __future__ import annotations
from pathlib import Path
import csv
import json
from typing import List
import numpy as np
import cv2

from .explain import ExplainRequest, explain_one
from .cam_diagnostics import compute_diagnostics
from .lung_mask import HeuristicLungMaskProvider

HTML_TMPL = """<!doctype html>
<html><head><meta charset="utf-8">
<title>CheXpert CAM Export</title>
<style>
body{font-family:Arial;background:#111;color:#eee}
.grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(320px,1fr));gap:12px}
.card{background:#1c1c1c;padding:8px;border-radius:8px}
img{max-width:100%;border-radius:6px}
.small{font-size:12px;color:#bbb}
</style>
</head><body>
<h1>CheXpert Lung-Aware CAM Export</h1>
<div class="grid">
{cards}
</div>
</body></html>
"""

CARD = """<div class="card">
<img src="{overlay}">
<div class="small">
<b>{label}</b> prob={prob:.3f}<br/>
energy_in_lungs={eil:.3f} entropy={ent:.3f}
</div>
</div>
"""

def export_batch(
    images: List[Path],
    ckpt: Path,
    labels: List[str],
    out_dir: Path,
    method: str = "scorecam",
    limit: int = 100
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []

    lung = HeuristicLungMaskProvider()

    for i, img in enumerate(images[:limit]):
        req = ExplainRequest(
            image_path=img,
            ckpt_path=ckpt,
            labels=labels,
            method=method,
            use_lung_gating=True,
        )
        res = explain_one(req, lung_provider=lung)
        diag = compute_diagnostics(res.viz_gated.cam, res.lung_mask.mask)

        stem = img.stem
        overlay_p = out_dir / f"{stem}__overlay.png"
        cv2.imwrite(str(overlay_p), cv2.cvtColor(res.viz_gated.overlay_rgb, cv2.COLOR_RGB2BGR))

        rows.append({
            "image": str(img),
            "label": res.cam_out.label_name,
            "prob": res.cam_out.prob,
            "energy_inside_lungs": diag.energy_inside_lungs,
            "energy_outside_lungs": diag.energy_outside_lungs,
            "entropy": diag.entropy,
            "method": method,
        })

        cards.append(CARD.format(
            overlay=overlay_p.name,
            label=res.cam_out.label_name,
            prob=res.cam_out.prob,
            eil=diag.energy_inside_lungs,
            ent=diag.entropy,
        ))

    # CSV
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # HTML
    html = HTML_TMPL.format(cards="\n".join(cards))
    (out_dir / "index.html").write_text(html, encoding="utf-8")
