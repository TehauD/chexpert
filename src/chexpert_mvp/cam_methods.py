from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CamOutput:
    cam: np.ndarray            # 2D CAM in [0,1]
    prob: float
    label_index: int
    label_name: str
    method: str
    notes: Optional[str] = None


class _Hook:
    def __init__(self) -> None:
        self.fmap = None
        self.grad = None

    def fwd(self, _module, _inputs, output):
        self.fmap = output

    def bwd(self, _module, _grad_in, grad_out):
        if isinstance(grad_out, (tuple, list)) and grad_out:
            self.grad = grad_out[0]
        else:
            self.grad = grad_out


def _find_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "features"):
        return model.features

    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for CAM target.")

    return last_conv


def disable_inplace_relu_(model: nn.Module) -> int:
    """
    Sets all nn.ReLU(inplace=True) to inplace=False in-place.
    Returns number of modules changed.
    """
    changed = 0
    for m in model.modules():
        if isinstance(m, nn.ReLU) and getattr(m, "inplace", False):
            m.inplace = False
            changed += 1
    return changed


def gradcam(
    model: nn.Module,
    x: torch.Tensor,
    labels: List[str],
    target_index: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> CamOutput:
    """
    Hookless Grad-CAM using torch.autograd.grad (no backward hooks).
    Much more stable under PyTorch >=2.x view/inplace constraints.

    Returns CAM at feature-map spatial resolution (caller resizes).
    """
    model.eval()
    device = device or next(model.parameters()).device
    x = x.to(device)

    # Avoid in-place relu issues (best effort)
    _ = disable_inplace_relu_(model)

    target_layer = _find_target_layer(model)

    # Forward through target_layer manually:
    # We need both:
    #  - feature maps (A)
    #  - logits output
    #
    # For torchvision DenseNet/ConvNeXt/EfficientNet, .features gives feature maps.
    # Then classifier head produces logits.
    #
    # We'll try to detect model type by attributes.
    with torch.enable_grad():
        x = x.requires_grad_(True)

        # Get feature maps
        A = target_layer(x)  # [1,K,Hf,Wf] typically
        if isinstance(A, (tuple, list)):
            A = A[0]
        if A.ndim != 4:
            raise RuntimeError(f"Expected feature map [1,K,Hf,Wf], got {tuple(A.shape)}")

        # Build logits depending on model architecture
        logits = None

        # DenseNet: features -> relu -> adaptive avg pool -> flatten -> classifier
        if hasattr(model, "classifier") and hasattr(model, "features"):
            z = A
            z = torch.relu(z)  # NOT inplace
            z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        # ConvNeXt / EfficientNet / others: easiest is to run full model forward too,
        # but that would recompute features and might reintroduce inplace issues.
        # We'll handle common torchvision patterns:
        elif hasattr(model, "avgpool") and hasattr(model, "classifier"):
            z = A
            # some models have avgpool module
            z = model.avgpool(z)
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        elif hasattr(model, "classifier"):
            # last-resort: try global pool then classifier
            z = torch.nn.functional.adaptive_avg_pool2d(A, (1, 1))
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        else:
            raise RuntimeError("Unsupported model head for hookless gradcam; model lacks expected classifier attributes.")

        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        if target_index is None:
            target_index = int(np.argmax(probs))
        if target_index < 0 or target_index >= len(labels):
            raise ValueError(f"target_index out of range: {target_index}")

        score = logits[0, target_index]

        # Compute gradient d(score)/dA without backward hooks
        dA = torch.autograd.grad(
            outputs=score,
            inputs=A,
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True
        )[0]  # [1,K,Hf,Wf]

        # Channel weights = global average pool of gradients
        weights = dA.mean(dim=(2, 3), keepdim=True)  # [1,K,1,1]

        cam = (weights * A).sum(dim=1, keepdim=False)  # [1,Hf,Wf]
        cam = torch.relu(cam)  # NOT inplace
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)

        cam_np = cam[0].detach().cpu().numpy().astype(np.float32)

        return CamOutput(
            cam=cam_np,
            prob=float(probs[target_index]),
            label_index=int(target_index),
            label_name=labels[target_index],
            method="gradcam",
            notes="hookless autograd.grad on .features",
        )


def gradcam_pp(
    model: nn.Module,
    x: torch.Tensor,
    labels: List[str],
    target_index: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> CamOutput:
    """
    Grad-CAM++ using torch.autograd.grad (hookless, safer for view/inplace rules).
    """
    model.eval()
    device = device or next(model.parameters()).device
    x = x.to(device)

    # Clear any old grads
    model.zero_grad(set_to_none=True)
    _ = disable_inplace_relu_(model)

    target_layer = _find_target_layer(model)

    with torch.enable_grad():
        x = x.requires_grad_(True)

        # Get feature maps
        A = target_layer(x)  # [1,K,Hf,Wf] typically
        if isinstance(A, (tuple, list)):
            A = A[0]
        if A.ndim != 4:
            raise RuntimeError(f"Expected feature map [1,K,Hf,Wf], got {tuple(A.shape)}")

        # Build logits depending on model architecture
        logits = None

        # DenseNet: features -> relu -> adaptive avg pool -> flatten -> classifier
        if hasattr(model, "classifier") and hasattr(model, "features"):
            z = A
            z = torch.relu(z)  # NOT inplace
            z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1))
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        # ConvNeXt / EfficientNet / others
        elif hasattr(model, "avgpool") and hasattr(model, "classifier"):
            z = A
            z = model.avgpool(z)
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        elif hasattr(model, "classifier"):
            z = torch.nn.functional.adaptive_avg_pool2d(A, (1, 1))
            z = torch.flatten(z, 1)
            logits = model.classifier(z)

        else:
            raise RuntimeError("Unsupported model head for gradcam++; model lacks expected classifier attributes.")

        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        # Choose class
        if target_index is None:
            target_index = int(np.argmax(probs))
        if target_index < 0 or target_index >= len(labels):
            raise ValueError(f"target_index out of range: {target_index} (labels={len(labels)})")

        # Backprop score to feature maps
        score = logits[0, target_index]
        dA = torch.autograd.grad(
            outputs=score,
            inputs=A,
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]  # [1,K,Hf,Wf]

        fmap = A[0]
        grad = dA[0]

        # Grad-CAM++ weights
        # alpha_ij = grad^2 / (2*grad^2 + sum_{a,b}( fmap * grad^3 ))
        grad2 = grad.pow(2)
        grad3 = grad.pow(3)

        # Denominator term: 2*grad^2 + fmap * grad^3 summed over spatial dims
        denom = 2.0 * grad2 + (fmap * grad3).sum(dim=(1, 2), keepdim=True)

        # Avoid divide-by-zero
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        alpha = grad2 / denom

        # Positive gradients only
        relu_grad = F.relu(grad)

        # Channel weights
        weights = (alpha * relu_grad).sum(dim=(1, 2))  # [K]

        # Weighted combination of feature maps
        cam = (weights[:, None, None] * fmap).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)

        cam_np = cam.detach().cpu().numpy().astype(np.float32)

        return CamOutput(
            cam=cam_np,
            prob=float(probs[target_index]),
            label_index=int(target_index),
            label_name=labels[target_index],
            method="gradcam++",
            notes="hookless autograd.grad on .features",
        )


def scorecam(
    model: nn.Module,
    x: torch.Tensor,
    labels: List[str],
    target_index: Optional[int] = None,
    max_channels: int = 64,
    device: Optional[torch.device] = None,
) -> CamOutput:
    model.eval()
    device = device or next(model.parameters()).device
    x = x.to(device)

    target_layer = _find_target_layer(model)
    hook = _Hook()
    h_fwd = target_layer.register_forward_hook(hook.fwd)

    try:
        _ = disable_inplace_relu_(model)

        with torch.no_grad():
            logits = model(x)  # [1,C]
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

            if target_index is None:
                target_index = int(np.argmax(probs))
            if target_index < 0 or target_index >= len(labels):
                raise ValueError(f"target_index out of range: {target_index} (labels={len(labels)})")

            if hook.fmap is None:
                raise RuntimeError("Score-CAM hook failed to capture fmap (None).")

            fmap = hook.fmap.detach()
            if isinstance(fmap, (tuple, list)):
                fmap = fmap[0]
            if fmap.ndim != 4:
                raise RuntimeError(f"Unexpected fmap shape: fmap={tuple(fmap.shape)}")

            # Upsample feature maps to input size for masking
            fmap_up = F.interpolate(
                fmap,
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )[0]  # [K,H,W]

            if max_channels is not None and max_channels > 0 and fmap_up.shape[0] > max_channels:
                scores = fmap_up.mean(dim=(1, 2))
                _, idxs = torch.topk(scores, k=max_channels, largest=True)
                fmap_sel = fmap_up[idxs]
            else:
                fmap_sel = fmap_up

            weights = []
            for k in range(fmap_sel.shape[0]):
                m = fmap_sel[k]
                m = m - m.min()
                m = m / (m.max().clamp(min=1e-6))
                masked = x * m[None, None, :, :]
                score_k = torch.sigmoid(model(masked))[0, target_index]
                weights.append(score_k.item())

            weights_t = torch.tensor(weights, device=fmap_sel.device, dtype=fmap_sel.dtype)
            cam = (weights_t[:, None, None] * fmap_sel).sum(dim=0)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / cam.max().clamp(min=1e-8)

            cam_np = cam.detach().cpu().numpy().astype(np.float32)

            return CamOutput(
                cam=cam_np,
                prob=float(probs[target_index]),
                label_index=int(target_index),
                label_name=labels[target_index],
                method="scorecam",
                notes=f"scorecam channels={fmap_sel.shape[0]}",
            )

    finally:
        h_fwd.remove()
