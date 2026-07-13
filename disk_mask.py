"""Fisheye disk masks — ignore black corners and lens rim during U-Net training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from edge_filter import DEFAULT_DISK_MARGIN, sky_disk_mask
from segment_nrbr import LABEL_VOID, fisheye_mask


def train_valid_mask(
    shape: tuple[int, int] = (64, 64),
    margin: int = DEFAULT_DISK_MARGIN,
) -> np.ndarray:
    """True inside usable sky disk (inset from fisheye rim by ``margin`` px)."""
    return sky_disk_mask(shape, margin)


def zero_outside_fisheye(rgb: np.ndarray) -> np.ndarray:
    """Zero RGB in black corner pixels outside the fisheye circle."""
    mask = fisheye_mask(rgb.shape[:2])
    out = rgb.copy()
    out[~mask] = 0
    return out


def mask_labels_outside_disk(labels: np.ndarray, margin: int = DEFAULT_DISK_MARGIN) -> np.ndarray:
    """Force void label outside the training-valid region."""
    out = labels.copy()
    out[~train_valid_mask(labels.shape, margin)] = LABEL_VOID
    return out


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy averaged only over ``valid`` pixels (B, H, W) bool."""
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)
    return (per_pixel * valid_f).sum() / denom


def masked_pixel_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    pred = logits.argmax(dim=1)
    ok = (pred == target) & valid
    n = valid.sum().item()
    if n == 0:
        return 0.0
    return ok.sum().item() / n


def masked_mean_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> float:
    pred = logits.argmax(dim=1)
    ious: list[float] = []
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (target == c) & valid
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft multi-class Dice loss averaged over classes present in the batch."""
    probs = F.softmax(logits, dim=1)
    valid_f = valid.float()
    losses: list[torch.Tensor] = []
    for c in range(num_classes):
        pred_c = probs[:, c] * valid_f
        tgt_c = (target == c).float() * valid_f
        inter = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        if denom > 0:
            dice = (2.0 * inter + smooth) / (denom + smooth)
            losses.append(1.0 - dice)
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def masked_seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Cross-entropy plus multi-class Dice loss."""
    ce = masked_cross_entropy(logits, target, valid)
    dice = masked_dice_loss(logits, target, valid, num_classes)
    return ce + dice


def masked_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Macro-averaged Dice, recall, and specificity (one-vs-rest, valid pixels only)."""
    pred = logits.argmax(dim=1)
    dices: list[float] = []
    recalls: list[float] = []
    specificities: list[float] = []
    for c in range(num_classes):
        p_pos = pred == c
        t_pos = target == c
        tp = (p_pos & t_pos & valid).sum().item()
        fn = (~p_pos & t_pos & valid).sum().item()
        fp = (p_pos & ~t_pos & valid).sum().item()
        tn = (~p_pos & ~t_pos & valid).sum().item()

        dice_denom = (p_pos & valid).sum().item() + (t_pos & valid).sum().item()
        if dice_denom > 0:
            dices.append(2.0 * tp / dice_denom)

        if tp + fn > 0:
            recalls.append(tp / (tp + fn))

        if tn + fp > 0:
            specificities.append(tn / (tn + fp))

    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "specificity": float(np.mean(specificities)) if specificities else 0.0,
    }
