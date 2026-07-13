"""
Evaluate trained U-Net on the held-out test split and save prediction visualizations.

  python evaluate_unet.py
  python evaluate_unet.py --dataset-dir output/dataset_round2 --checkpoint output/checkpoints/unet_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from disk_mask import (
    DEFAULT_DISK_MARGIN,
    mask_labels_outside_disk,
    masked_mean_iou,
    masked_pixel_accuracy,
    masked_seg_loss,
    masked_segmentation_metrics,
    train_valid_mask,
    zero_outside_fisheye,
)
from seg_dataset import SegPairDataset, remap_mask_for_training
from segment_nrbr import LABEL_NAMES, colorize_labels, overlay_labels
from unet_model import NUM_CLASSES, UNet64

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "output" / "checkpoints" / "unet_best.pt"
DEFAULT_CHECKPOINT_DIR = ROOT / "output" / "checkpoints"
DEFAULT_DATASET = ROOT / "output" / "dataset"
DEFAULT_OUT_DIR = ROOT / "output" / "checkpoints" / "test_eval"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNet64, int, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_ch = int(ckpt.get("base_ch", 32))
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    disk_margin = int(ckpt.get("disk_margin", DEFAULT_DISK_MARGIN))
    model = UNet64(in_channels=3, num_classes=num_classes, base=base_ch)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, disk_margin, num_classes


def load_test_stems(split_file: Path) -> list[str]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Test split not found: {split_file}")
    stems = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stems:
        raise ValueError(f"No stems in test split: {split_file}")
    return stems


def resolve_dataset_dir(checkpoint_dir: Path, dataset_dir: Path | None) -> Path:
    if dataset_dir is not None:
        return dataset_dir
    meta_path = checkpoint_dir / "split_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = meta.get("dataset_dir")
        if saved:
            return Path(saved)
    return DEFAULT_DATASET


def per_class_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> dict[int, dict[str, int]]:
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in range(num_classes)}
    for c in range(num_classes):
        p_pos = pred == c
        t_pos = target == c
        counts[c]["tp"] = int((p_pos & t_pos & valid).sum().item())
        counts[c]["fn"] = int((~p_pos & t_pos & valid).sum().item())
        counts[c]["fp"] = int((p_pos & ~t_pos & valid).sum().item())
        counts[c]["tn"] = int((~p_pos & ~t_pos & valid).sum().item())
    return counts


def merge_counts(
    total: dict[int, dict[str, int]],
    batch: dict[int, dict[str, int]],
) -> None:
    for c, row in batch.items():
        for key in ("tp", "fp", "fn", "tn"):
            total[c][key] += row[key]


def summarize_per_class(counts: dict[int, dict[str, int]]) -> dict[str, dict[str, float | int | str]]:
    out: dict[str, dict[str, float | int | str]] = {}
    for c, row in counts.items():
        tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
        dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else None
        iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else None
        out[LABEL_NAMES[c]] = {
            "class_id": c,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": dice,
            "recall": recall,
            "specificity": specificity,
            "iou": iou,
        }
    return out


def metrics_from_logits(
    logits: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    seg_metrics = masked_segmentation_metrics(logits, gt, valid, num_classes)
    return {
        "loss": float(masked_seg_loss(logits, gt, valid, num_classes).item()),
        "dice": seg_metrics["dice"],
        "recall": seg_metrics["recall"],
        "specificity": seg_metrics["specificity"],
        "acc": masked_pixel_accuracy(logits, gt, valid),
        "miou": masked_mean_iou(logits, gt, valid, num_classes),
    }


@torch.no_grad()
def compute_single_image_metrics(
    model: UNet64,
    img: np.ndarray,
    gt_labels: np.ndarray,
    valid_mask: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    y = torch.from_numpy(gt_labels.astype(np.int64)).unsqueeze(0)
    valid = valid_mask.unsqueeze(0).to(device)
    logits = model(x.to(device))
    return metrics_from_logits(logits, y.to(device), valid, num_classes)


def format_metrics_block(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f}   dice={metrics['dice']:.3f}   "
        f"recall={metrics['recall']:.3f}   specificity={metrics['specificity']:.3f}   "
        f"acc={metrics['acc']:.3f}   miou={metrics['miou']:.3f}"
    )


@torch.no_grad()
def evaluate_test_set(
    model: UNet64,
    loader: DataLoader,
    valid_mask: torch.Tensor,
    device: torch.device,
    disk_margin: int,
    num_classes: int,
) -> tuple[dict[str, float], dict[int, dict[str, int]]]:
    valid = valid_mask.unsqueeze(0)
    totals = {
        "loss": 0.0,
        "acc": 0.0,
        "miou": 0.0,
        "dice": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }
    class_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in range(num_classes)}
    n = 0

    for images, masks in tqdm(loader, desc="Evaluating", unit="batch"):
        images = images.to(device)
        masks = masks.to(device)
        batch_valid = valid.expand(images.size(0), -1, -1)
        logits = model(images)
        loss = masked_seg_loss(logits, masks, batch_valid, num_classes)
        seg_metrics = masked_segmentation_metrics(logits, masks, batch_valid, num_classes)

        pred = logits.argmax(dim=1)
        merge_counts(class_counts, per_class_counts(pred, masks, batch_valid, num_classes))

        bs = images.size(0)
        totals["loss"] += loss.item() * bs
        totals["acc"] += masked_pixel_accuracy(logits, masks, batch_valid) * bs
        totals["miou"] += masked_mean_iou(logits, masks, batch_valid, num_classes) * bs
        totals["dice"] += seg_metrics["dice"] * bs
        totals["recall"] += seg_metrics["recall"] * bs
        totals["specificity"] += seg_metrics["specificity"] * bs
        n += bs

    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}, class_counts


def save_prediction_figure(
    out_path: Path,
    stem: str,
    img: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    metrics: dict[str, float],
) -> None:
    fig = plt.figure(figsize=(10, 3.6), dpi=120)
    gs = fig.add_gridspec(2, 4, height_ratios=[5, 1.2], hspace=0.35)
    panels = [
        (img, "Image"),
        (colorize_labels(gt_labels), "Ground truth"),
        (colorize_labels(pred_labels), "Prediction"),
        (overlay_labels(img, pred_labels), "Pred overlay"),
    ]
    for col, (data, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.axis("off")
    ax_metrics.text(
        0.5,
        0.65,
        stem,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax_metrics.text(
        0.5,
        0.2,
        format_metrics_block(metrics),
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f4f4f4", edgecolor="#bbbbbb"),
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_overview_grid(
    out_path: Path,
    samples: list[tuple[str, np.ndarray, np.ndarray, dict[str, float]]],
    *,
    cols: int = 4,
) -> None:
    if not samples:
        return
    n = len(samples)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), dpi=120)
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx >= n:
            ax.axis("off")
            continue
        stem, img, pred, metrics = samples[idx]
        ax.imshow(overlay_labels(img, pred))
        ax.set_title(f"{stem}\ndice={metrics['dice']:.3f} miou={metrics['miou']:.3f}", fontsize=6)
        ax.axis("off")
    fig.suptitle("Test set predictions (overlay)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_all_visualizations(
    model: UNet64,
    dataset: SegPairDataset,
    valid_mask: torch.Tensor,
    device: torch.device,
    disk_margin: int,
    viz_dir: Path,
    overview_samples: list[tuple[str, np.ndarray, np.ndarray, dict[str, float]]],
    per_image_metrics: dict[str, dict[str, float]],
    *,
    max_overview: int,
    num_classes: int,
) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(dataset.stems, desc="Saving figures", unit="img"):
        img = zero_outside_fisheye(
            np.array(Image.open(dataset.images_dir / f"{stem}.png").convert("RGB"), dtype=np.uint8)
        )
        gt = np.array(Image.open(dataset.masks_dir / f"{stem}.png"), dtype=np.uint8)
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = remap_mask_for_training(gt).astype(np.uint8)

        x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0)
        valid = valid_mask.unsqueeze(0)
        logits = model(x.to(device))
        metrics = metrics_from_logits(logits, y.to(device), valid.to(device), num_classes)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = mask_labels_outside_disk(pred, margin=disk_margin)

        per_image_metrics[stem] = metrics
        save_prediction_figure(viz_dir / f"{stem}.png", stem, img, gt, pred, metrics)
        if len(overview_samples) < max_overview:
            overview_samples.append((stem, img, pred, metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on held-out test split")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--split-file", type=str, default="", help="Default: <checkpoint-dir>/split_test.txt")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-overview", type=int, default=16, help="Images in overview grid PNG")
    parser.add_argument("--no-viz", action="store_true", help="Skip per-image visualization PNGs")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "visualizations"
    dataset_dir = resolve_dataset_dir(
        checkpoint_dir, Path(args.dataset_dir) if args.dataset_dir else None
    )
    split_file = Path(args.split_file) if args.split_file else checkpoint_dir / "split_test.txt"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Test split: {split_file}")

    model, disk_margin, num_classes = load_model(checkpoint_path, device)
    test_stems = load_test_stems(split_file)
    print(f"Test images: {len(test_stems)}")

    valid_np = train_valid_mask((64, 64), margin=disk_margin)
    valid_mask = torch.from_numpy(valid_np).to(device)

    test_ds = SegPairDataset(images_dir, masks_dir, test_stems)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics, class_counts = evaluate_test_set(
        model, test_loader, valid_mask, device, disk_margin, num_classes
    )
    per_class = summarize_per_class(class_counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "n_test": len(test_stems),
        "disk_margin": disk_margin,
        "metrics": metrics,
        "per_class": per_class,
    }
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overview_samples: list[tuple[str, np.ndarray, np.ndarray, dict[str, float]]] = []
    per_image_metrics: dict[str, dict[str, float]] = {}
    if not args.no_viz:
        save_all_visualizations(
            model,
            test_ds,
            valid_mask,
            device,
            disk_margin,
            viz_dir,
            overview_samples,
            per_image_metrics,
            max_overview=args.max_overview,
            num_classes=num_classes,
        )
        save_overview_grid(out_dir / "overview_grid.png", overview_samples, cols=4)
        per_image_path = out_dir / "per_image_metrics.json"
        per_image_path.write_text(json.dumps(per_image_metrics, indent=2), encoding="utf-8")

    print("\nTest set results:")
    print(f"  loss={metrics['loss']:.4f}")
    print(f"  dice={metrics['dice']:.3f}")
    print(f"  recall={metrics['recall']:.3f}")
    print(f"  specificity={metrics['specificity']:.3f}")
    print(f"  acc={metrics['acc']:.3f}")
    print(f"  miou={metrics['miou']:.3f}")
    print(f"\nMetrics JSON: {metrics_path}")
    if not args.no_viz:
        print(f"Visualizations: {viz_dir} ({len(test_stems)} PNGs)")
        print(f"Per-image metrics: {out_dir / 'per_image_metrics.json'}")
        print(f"Overview grid: {out_dir / 'overview_grid.png'}")


if __name__ == "__main__":
    main()
