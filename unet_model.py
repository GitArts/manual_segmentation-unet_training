"""Compact U-Net for 64×64 RGB sky segmentation (3 classes: void, sky, cloud)."""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 3


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet64(nn.Module):
    """Encoder–decoder U-Net sized for 64×64 inputs (4 pooling levels)."""

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES, base: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bot = DoubleConv(c4, c4)

        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec4 = DoubleConv(c3 + c4, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = DoubleConv(c2 + c3, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2 = DoubleConv(c1 + c2, c1)
        self.up1 = nn.ConvTranspose2d(c1, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bot(self.pool(s4))

        x = self.up4(b)
        x = self.dec4(torch.cat([x, s4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.out_conv(x)


def predict_labels(model: nn.Module, image_rgb: torch.Tensor) -> torch.Tensor:
    """image_rgb: (B, 3, H, W) float in [0, 1]. Returns (B, H, W) int64 labels."""
    logits = model(image_rgb)
    return logits.argmax(dim=1)
