from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

def multilabel_auroc(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray, labels: list[str]):
    """
    probs/targets/masks: [N,C]
    masks: 1=use, 0=ignore
    Returns (aurocs: dict, mean_auc: float|None)
    """
    aurocs = {}
    for i, lab in enumerate(labels):
        keep = masks[:, i] > 0.5
        if keep.sum() < 10:
            aurocs[lab] = None
            continue
        y = targets[keep, i]
        p = probs[keep, i]
        if len(np.unique(y)) < 2:
            aurocs[lab] = None
            continue
        aurocs[lab] = float(roc_auc_score(y, p))

    vals = [v for v in aurocs.values() if v is not None]
    mean_auc = float(np.mean(vals)) if vals else None
    return aurocs, mean_auc
