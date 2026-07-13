"""PyTorch dataset for paired images/ and masks/ folders."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from disk_mask import zero_outside_fisheye
from segment_nrbr import LABEL_CLOUD, LABEL_SKY, LABEL_SUN, LABEL_SUN_CLOUDED
from unet_model import NUM_CLASSES


def remap_mask_for_training(mask: np.ndarray) -> np.ndarray:
    """Collapse stored 5-class masks to void / sky / cloud for training.

    Sun disk (class 3) is clear sky in the 3-class setup → remapped to sky.
    Sun behind cloud (class 4) → remapped to cloud.
    """
    out = mask.astype(np.int64, copy=True)
    out[out == LABEL_SUN] = LABEL_SKY
    out[out == LABEL_SUN_CLOUDED] = LABEL_CLOUD
    return np.clip(out, 0, NUM_CLASSES - 1).astype(np.int64)


def list_paired_stems(images_dir: Path, masks_dir: Path) -> list[str]:
    stems: list[str] = []
    for img_path in sorted(images_dir.glob("*.png")):
        mask_path = masks_dir / img_path.name
        if mask_path.is_file():
            stems.append(img_path.stem)
    if not stems:
        raise FileNotFoundError(f"No paired PNGs in {images_dir} and {masks_dir}")
    return stems


def train_val_split(stems: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    order = stems[:]
    rng.shuffle(order)
    if len(order) < 2:
        return order, []
    n_val = max(1, int(round(len(order) * val_frac)))
    n_val = min(n_val, len(order) - 1)
    val = order[:n_val]
    train = order[n_val:]
    return train, val


def train_val_test_split(
    stems: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle with seed and split into train / val / test stem lists."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = random.Random(seed)
    order = stems[:]
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return [], [], []
    if n < 3:
        return order, [], []

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_test - n_val

    if n_train < 1:
        shortfall = 1 - n_train
        take_val = min(shortfall, max(0, n_val - 1))
        n_val -= take_val
        shortfall -= take_val
        if shortfall > 0:
            n_test = max(0, n_test - shortfall)
        n_train = n - n_test - n_val

    test = order[:n_test]
    val = order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    return train, val, test


def save_split_manifests(
    checkpoint_dir: Path,
    train_stems: list[str],
    val_stems: list[str],
    test_stems: list[str],
    *,
    seed: int,
    dataset_dir: Path | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(train_stems) + len(val_stems) + len(test_stems)
    meta = {
        "seed": seed,
        "train_frac": len(train_stems) / max(total, 1),
        "val_frac": len(val_stems) / max(total, 1),
        "test_frac": len(test_stems) / max(total, 1),
        "n_train": len(train_stems),
        "n_val": len(val_stems),
        "n_test": len(test_stems),
    }
    if dataset_dir is not None:
        meta["dataset_dir"] = str(dataset_dir.resolve())
    (checkpoint_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    for name, stems in (("split_train.txt", train_stems), ("split_val.txt", val_stems), ("split_test.txt", test_stems)):
        (checkpoint_dir / name).write_text("\n".join(stems) + ("\n" if stems else ""), encoding="utf-8")


class SegPairDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, stems: list[str]) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img = zero_outside_fisheye(
            np.array(Image.open(self.images_dir / f"{stem}.png").convert("RGB"), dtype=np.uint8)
        )
        mask = np.array(Image.open(self.masks_dir / f"{stem}.png"), dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = remap_mask_for_training(mask)

        x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        y = torch.from_numpy(mask.astype(np.int64))
        return x, y
