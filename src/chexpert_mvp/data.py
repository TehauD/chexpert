from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, Optional

import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

UNCERTAIN_STRATEGY = Literal["zero", "one", "ignore"]

@dataclass
class CheXpertSample:
    img_path: Path
    y: torch.Tensor      # [C]
    y_mask: torch.Tensor # [C] (1=use label in loss, 0=ignore)

class CheXpertDataset(Dataset):
    """
    CheXpert labels:
      1   positive
      0   negative
     -1   uncertain
     NaN  missing

    uncertain strategies:
      - zero:   -1 -> 0, included in loss
      - one:    -1 -> 1, included in loss
      - ignore: -1 masked out of loss
    """

    def __init__(
        self,
        root: Path,
        csv_path: Path,
        labels: List[str],
        img_size: int = 384,
        uncertain: UNCERTAIN_STRATEGY = "zero",
        limit: int = 0,
        logger=None,
    ):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.labels = labels
        self.uncertain = uncertain
        self.logger = logger

        if not self.root.exists():
            raise FileNotFoundError(
                f"CheXpert root not found: {self.root}\n"
                f"Expected: data/chexpert/CheXpert-v1.0-small/"
            )
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if "Path" not in df.columns:
            raise ValueError(f"CSV missing 'Path' column: {self.csv_path}")

        missing_cols = [c for c in labels if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing label columns: {missing_cols}")

        if limit and limit > 0:
            df = df.head(limit).copy()

        self.df = df.reset_index(drop=True)

        # standard ImageNet normalization (works well for pretrained backbones)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        if self.logger:
            self.logger.info(f"[data] Loaded {len(self.df)} rows from {self.csv_path.name}")
            self.logger.info(f"[data] Labels: {labels} | uncertain='{uncertain}'")
            self.logger.info(f"[data] Example CSV Path: {self.df.iloc[0]['Path']}")

    def __len__(self) -> int:
        return len(self.df)

    def _encode_labels(self, row) -> Tuple[torch.Tensor, torch.Tensor]:
        y, mask = [], []
        for lab in self.labels:
            v = row[lab]
            if pd.isna(v):
                y.append(0.0); mask.append(0.0)
                continue
            v = float(v)
            if v == -1.0:
                if self.uncertain == "zero":
                    y.append(0.0); mask.append(1.0)
                elif self.uncertain == "one":
                    y.append(1.0); mask.append(1.0)
                else:
                    y.append(0.0); mask.append(0.0)
            elif v == 1.0:
                y.append(1.0); mask.append(1.0)
            else:
                y.append(0.0); mask.append(1.0)
        return torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def _resolve_image_path(self, path_value: str) -> Path:
        """
        CheXpert CSVs can store paths like:
          - CheXpert-v1.0-small/train/patient.../view1_frontal.jpg
          - train/patient.../view1_frontal.jpg
        We normalize so self.root / rel is correct in both cases.
        """
        raw = Path(str(path_value)).as_posix()  # normalize separators
        raw_path = Path(raw)

        # if CSV begins with dataset folder name, strip it
        if len(raw_path.parts) > 0 and raw_path.parts[0] == self.root.name:
            rel = Path(*raw_path.parts[1:])
        else:
            rel = raw_path

        return (self.root / rel)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["Path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y, y_mask = self._encode_labels(row)
        return CheXpertSample(img_path=img_path, y=y, y_mask=y_mask), x


def compute_pos_weight(
    csv_path: Path,
    labels: List[str],
    uncertain: UNCERTAIN_STRATEGY,
    limit: int = 0,
    logger=None
) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogits: neg/pos for each label.
    Uses the same uncertain handling as the dataset label encoding.
    """
    df = pd.read_csv(csv_path)
    if limit and limit > 0:
        df = df.head(limit).copy()

    pos = np.zeros(len(labels), dtype=np.float64)
    neg = np.zeros(len(labels), dtype=np.float64)

    for i, lab in enumerate(labels):
        if lab not in df.columns:
            raise ValueError(f"CSV missing label column: {lab}")
        col = df[lab]
        for v in col.values:
            if pd.isna(v):
                continue
            v = float(v)
            if v == -1.0:
                if uncertain == "zero":
                    neg[i] += 1.0
                elif uncertain == "one":
                    pos[i] += 1.0
                else:
                    continue
            elif v == 1.0:
                pos[i] += 1.0
            else:
                neg[i] += 1.0

    # Avoid divide-by-zero: if pos is 0, set weight to 1.0
    pos_weight = np.ones_like(pos)
    nz = pos > 0
    pos_weight[nz] = neg[nz] / pos[nz]

    if logger:
        logger.info(f"[data] pos_weight={pos_weight.tolist()}")

    return torch.tensor(pos_weight, dtype=torch.float32)

# MUST be top-level for Windows multiprocessing pickling
def chexpert_collate(batch):
    samples, xs = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack([s.y for s in samples], dim=0)
    m = torch.stack([s.y_mask for s in samples], dim=0)
    paths = [s.img_path for s in samples]
    return x, y, m, paths

def make_loaders(
    root: Path,
    train_csv: Path,
    valid_csv: Path,
    labels: List[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    uncertain: UNCERTAIN_STRATEGY,
    limit_train: int = 0,
    limit_valid: int = 0,
    logger=None
):
    ds_train = CheXpertDataset(root, train_csv, labels, img_size, uncertain, limit_train, logger)
    ds_valid = CheXpertDataset(root, valid_csv, labels, img_size, uncertain, limit_valid, logger)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=chexpert_collate,
        persistent_workers=(num_workers > 0),
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=chexpert_collate,
        persistent_workers=(num_workers > 0),
    )
    return dl_train, dl_valid
