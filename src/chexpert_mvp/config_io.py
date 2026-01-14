from __future__ import annotations
from pathlib import Path
import yaml
from .config import TrainConfig

def _as_path(v) -> Path:
    return v if isinstance(v, Path) else Path(str(v))

def load_config(path: Path) -> TrainConfig:
    """
    Load YAML -> TrainConfig.
    Paths in YAML are treated as repo-relative if not absolute.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Normalize relative paths relative to repo root (config file location is repo-root/configs/...)
    base = path.parent.parent if (path.parent.name == "configs") else path.parent

    if "chexpert_root" in raw:
        p = _as_path(raw["chexpert_root"])
        raw["chexpert_root"] = p if p.is_absolute() else (base / p)

    if "output_dir" in raw:
        p = _as_path(raw["output_dir"])
        raw["output_dir"] = p if p.is_absolute() else (base / p)

    if "train_csv" in raw:
        p = _as_path(raw["train_csv"])
        raw["train_csv"] = p if p.is_absolute() else (base / p)

    if "valid_csv" in raw:
        p = _as_path(raw["valid_csv"])
        raw["valid_csv"] = p if p.is_absolute() else (base / p)

    return TrainConfig(**raw)
