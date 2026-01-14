from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal U-Net skeleton (encoder-decoder). Replace with your preferred impl if you like.
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        def C(in_c, out_c): 
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
            )
        self.e1 = C(in_ch, base)
        self.e2 = C(base, base*2)
        self.e3 = C(base*2, base*4)
        self.e4 = C(base*4, base*8)
        self.p = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.d3 = C(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.d2 = C(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.d1 = C(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p(e1))
        e3 = self.e3(self.p(e2))
        e4 = self.e4(self.p(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)

class UNetLungMaskProvider:
    """
    True lung segmentation provider.
    Expects weights trained on JSRT/Montgomery/Shenzhen etc.
    """
    def __init__(self, weights: Path, device: str = "auto", threshold: float = 0.5):
        self.device = torch.device("cuda" if device=="auto" and torch.cuda.is_available() else device)
        self.model = UNet()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict_mask(self, img_rgb: np.ndarray):
        import torchvision.transforms as T
        from .lung_mask import LungMaskResult

        tf = T.Compose([
            T.ToTensor(),
            T.Resize((512,512)),
        ])
        x = tf(img_rgb).unsqueeze(0).to(self.device)
        y = torch.sigmoid(self.model(x))[0,0]
        mask = (y > self.threshold).cpu().numpy().astype(np.uint8)

        return LungMaskResult(
            mask=mask,
            method="unet",
            notes=f"threshold={self.threshold}"
        )
