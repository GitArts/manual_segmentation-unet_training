"""Run trained U-Net on RGB images at arbitrary resolution (resize to 64 for inference)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from disk_mask import DEFAULT_DISK_MARGIN, mask_labels_outside_disk, zero_outside_fisheye
from seg_dataset import remap_mask_for_training
from unet_model import UNet64, predict_labels

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "output" / "checkpoints" / "unet_best.pt"
INFER_SIZE = 64


@dataclass
class SegPredictor:
    model: UNet64
    device: torch.device
    disk_margin: int
    num_classes: int
    infer_size: int = INFER_SIZE

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
    ) -> SegPredictor:
        checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        base_ch = int(ckpt.get("base_ch", 32))
        num_classes = int(ckpt.get("num_classes", 3))
        disk_margin = int(ckpt.get("disk_margin", DEFAULT_DISK_MARGIN))
        model = UNet64(in_channels=3, num_classes=num_classes, base=base_ch)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return cls(model=model, device=device, disk_margin=disk_margin, num_classes=num_classes)

    def predict_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Predict void/sky/cloud labels for RGB image (HxWx3, uint8 or float 0-1)."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}")

        h, w = rgb.shape[:2]
        if rgb.dtype == np.uint8:
            arr = rgb
        else:
            arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        arr = zero_outside_fisheye(arr)
        small = np.array(
            Image.fromarray(arr).resize(
                (self.infer_size, self.infer_size), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        small = zero_outside_fisheye(small)

        x = torch.from_numpy(small.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        x = x.to(self.device)
        with torch.no_grad():
            labels = predict_labels(self.model, x).squeeze(0).cpu().numpy().astype(np.uint8)

        labels = mask_labels_outside_disk(labels, margin=self.disk_margin)
        labels = remap_mask_for_training(labels).astype(np.uint8)

        if (h, w) != (self.infer_size, self.infer_size):
            labels = np.array(
                Image.fromarray(labels, mode="L").resize((w, h), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            labels = mask_labels_outside_disk(labels, margin=self.disk_margin)

        return labels


def labels_to_mask_channels(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """One-hot class maps as float32 channels: (H, W, num_classes)."""
    h, w = labels.shape[:2]
    out = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        out[..., c] = (labels == c).astype(np.float32)
    return out
