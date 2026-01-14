from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .config import TrainConfig
from .config_io import load_config
from .data import make_loaders, compute_pos_weight
from .logging_utils import setup_logger
from .model import build_model, pick_device
from .metrics import multilabel_auroc

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # speed-oriented defaults (ok for MVP). Set deterministic=True if you need exact reproducibility.
    torch.backends.cudnn.benchmark = True

def masked_bce_with_logits(logits, targets, mask, pos_weight: torch.Tensor | None = None):
    # elementwise BCE then masked average
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight
    )
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

def freeze_backbone(model: nn.Module):
    # freeze everything except last classification layer(s)
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze known classifier blocks per backbone types
    # DenseNet: classifier
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    # ConvNeXt/EfficientNet: classifier is sequential
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True

@torch.no_grad()
def evaluate(model, dl_valid, device, labels, amp: bool, pos_weight: torch.Tensor | None):
    model.eval()
    probs_all, y_all, m_all = [], [], []
    loss_sum = 0.0
    n_batches = 0

    for x, y, m, _paths in tqdm(dl_valid, desc="valid", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = masked_bce_with_logits(logits, y, m, pos_weight=pos_weight)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy())
        m_all.append(m.detach().cpu().numpy())
        loss_sum += float(loss.item())
        n_batches += 1

    probs_np = np.concatenate(probs_all, axis=0)
    y_np = np.concatenate(y_all, axis=0)
    m_np = np.concatenate(m_all, axis=0)

    aurocs, mean_auc = multilabel_auroc(probs_np, y_np, m_np, labels)
    return {
        "val_loss": loss_sum / max(n_batches, 1),
        "aurocs": aurocs,
        "mean_auc": mean_auc
    }

def train(cfg: TrainConfig):
    logger = setup_logger("chexpert")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    logger.info(f"[run] dir={run_dir}")
    logger.info(f"[env] device={device.type} | amp={cfg.amp} | seed={cfg.seed}")
    logger.info(f"[data] root={cfg.chexpert_root}")

    dl_train, dl_valid = make_loaders(
        root=cfg.chexpert_root,
        train_csv=cfg.train_csv,
        valid_csv=cfg.valid_csv,
        labels=cfg.labels,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        uncertain=cfg.uncertain,
        limit_train=cfg.limit_train,
        limit_valid=cfg.limit_valid,
        logger=logger
    )

    model = build_model(cfg.backbone, num_classes=len(cfg.labels)).to(device)
    pos_weight = None
    if cfg.use_pos_weight:
        pos_weight = compute_pos_weight(
            cfg.train_csv,
            cfg.labels,
            cfg.uncertain,
            limit=cfg.limit_train,
            logger=logger
        ).to(device)

    if cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
        logger.info(f"[model] freezing backbone for {cfg.freeze_backbone_epochs} epoch(s)")
        freeze_backbone(model)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))

    history_path = run_dir / "history.jsonl"
    best_auc = -1.0

    logger.info("[train] starting")

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        if cfg.freeze_backbone_epochs and epoch == cfg.freeze_backbone_epochs + 1:
            logger.info("[model] unfreezing backbone")
            unfreeze_all(model)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(dl_train, desc=f"train e{epoch}/{cfg.epochs}")
        for x, y, m, paths in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            try:
                with autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                    logits = model(x)
                    loss = masked_bce_with_logits(logits, y, m, pos_weight=pos_weight)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            except RuntimeError as e:
                logger.warning(f"[warn] batch failed: {type(e).__name__}: {e}")
                logger.warning(f"[warn] example path={paths[0] if paths else 'N/A'}")
                continue

            loss_sum += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=(loss_sum / max(n_batches, 1)))

        metrics = evaluate(model, dl_valid, device, cfg.labels, amp=cfg.amp, pos_weight=pos_weight)

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            **metrics
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"[epoch {epoch}] train_loss={record['train_loss']:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} mean_auc={metrics['mean_auc']}")

        ckpt = {
            "model": model.state_dict(),
            "labels": cfg.labels,
            "epoch": epoch,
            "config": asdict(cfg),
            "backbone": cfg.backbone,
        }
        torch.save(ckpt, run_dir / "last.pt")

        if metrics["mean_auc"] is not None and metrics["mean_auc"] > best_auc:
            best_auc = metrics["mean_auc"]
            torch.save(ckpt, run_dir / "best.pt")
            logger.info(f"[ckpt] new best mean_auc={best_auc:.4f}")

    logger.info(f"[done] best_mean_auc={best_auc if best_auc >= 0 else None} | artifacts={run_dir}")
    return run_dir

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == "__main__":
    main()
