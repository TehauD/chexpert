from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List

UNCERTAIN_STRATEGY = Literal["zero", "one", "ignore"]
DEVICE_STR = Literal["auto", "cuda", "cpu"]

# repo-root discovery: src/chexpert_mvp/config.py -> parents[2] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class TrainConfig:
    # paths
    chexpert_root: Path = PROJECT_ROOT / "data/chexpert/CheXpert-v1.0-small"
    output_dir: Path = PROJECT_ROOT / "outputs/runs"

    train_csv: Path | None = None
    valid_csv: Path | None = None

    # labels & semantics
    labels: List[str] | None = None
    uncertain: UNCERTAIN_STRATEGY = "zero"

    # model / training
    backbone: str = "densenet121"
    img_size: int = 384
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 0
    use_pos_weight: bool = True

    amp: bool = True
    seed: int = 1337
    device: DEVICE_STR = "auto"

    limit_train: int = 0
    limit_valid: int = 0

    def __post_init__(self):
        if self.labels is None:
            object.__setattr__(self, "labels", [
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ])
        if self.train_csv is None:
            object.__setattr__(self, "train_csv", self.chexpert_root / "train.csv")
        if self.valid_csv is None:
            object.__setattr__(self, "valid_csv", self.chexpert_root / "valid.csv")
