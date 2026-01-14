from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as tvm

BackboneName = Literal[
    "densenet121", "densenet169", "densenet201",
    "convnext_tiny", "convnext_small",
    "efficientnet_v2_s",
]

def pick_device(device: str = "auto") -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(backbone: str, num_classes: int) -> nn.Module:
    b = backbone.lower()

    if b == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, num_classes)
        return m

    if b == "densenet169":
        m = tvm.densenet169(weights=tvm.DenseNet169_Weights.IMAGENET1K_V1)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, num_classes)
        return m

    if b == "densenet201":
        m = tvm.densenet201(weights=tvm.DenseNet201_Weights.IMAGENET1K_V1)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, num_classes)
        return m

    if b == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m

    if b == "convnext_small":
        m = tvm.convnext_small(weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m

    if b == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m

    raise ValueError(f"Unknown backbone: {backbone}")
