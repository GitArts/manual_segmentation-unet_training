"""
Train U-Net on curated dataset (images/ + masks/).

Loss = cross-entropy + Dice. Metrics inside inset fisheye disk only.
Classes: void (0), sky (1), cloud (2). Sun labels in masks are remapped (sun→sky, sun_clouded→cloud).

Default split: 80% train, 10% val, 10% test (seed 42). Test set is saved but not used.

  python train_unet.py --epochs 120
  python train_unet.py --dataset-dir output/dataset_round2 --epochs 120 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from disk_mask import (
    DEFAULT_DISK_MARGIN,
    masked_mean_iou,
    masked_pixel_accuracy,
    masked_seg_loss,
    masked_segmentation_metrics,
    train_valid_mask,
)
from seg_dataset import (
    SegPairDataset,
    list_paired_stems,
    save_split_manifests,
    train_val_test_split,
)
from unet_model import NUM_CLASSES, UNet64

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "output" / "dataset"
DEFAULT_CHECKPOINT_DIR = ROOT / "output" / "checkpoints"


def save_training_plots(out_dir: Path, history: list[dict[str, Any]], has_val: bool) -> Path | None:
    if not history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped training_curves.png (see train_history.json)")
        return None

    epochs = [int(h["epoch"]) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    metrics = [
        ("loss", "Loss (CE + Dice)", "train_loss", "val_loss"),
        ("dice", "Dice", "train_dice", "val_dice"),
        ("recall", "Recall", "train_recall", "val_recall"),
        ("specificity", "Specificity", "train_specificity", "val_specificity"),
    ]
    for ax, (key, title, train_key, val_key) in zip(axes.ravel(), metrics):
        train_vals = [float(h[train_key]) for h in history]
        ax.plot(epochs, train_vals, label="train", color="#1f77b4", linewidth=1.5)
        if has_val:
            val_vals = [float(h[val_key]) for h in history]
            ax.plot(epochs, val_vals, label="val", color="#ff7f0e", linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
        ax.legend()

    fig.suptitle("U-Net sky segmentation — training curves", fontsize=12)
    fig.tight_layout()
    png_path = out_dir / "training_curves.png"
    try:
        fig.savefig(png_path, bbox_inches="tight")
    except OSError as err:
        print(f"Could not write training_curves.png: {err}")
        plt.close(fig)
        return None
    plt.close(fig)
    return png_path


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    valid_mask: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    desc: str = "train",
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    valid = valid_mask.unsqueeze(0)

    totals = {
        "loss": 0.0,
        "acc": 0.0,
        "miou": 0.0,
        "dice": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        batch_valid = valid.expand(images.size(0), -1, -1)

        if train_mode:
            optimizer.zero_grad()
            logits = model(images)
            loss = masked_seg_loss(logits, masks, batch_valid, NUM_CLASSES)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = masked_seg_loss(logits, masks, batch_valid, NUM_CLASSES)

        seg_metrics = masked_segmentation_metrics(logits, masks, batch_valid, NUM_CLASSES)
        bs = images.size(0)
        totals["loss"] += loss.item() * bs
        totals["acc"] += masked_pixel_accuracy(logits, masks, batch_valid) * bs
        totals["miou"] += masked_mean_iou(logits, masks, batch_valid, NUM_CLASSES) * bs
        totals["dice"] += seg_metrics["dice"] * bs
        totals["recall"] += seg_metrics["recall"] * bs
        totals["specificity"] += seg_metrics["specificity"] * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{totals['dice'] / n:.3f}")

    denom = max(n, 1)
    return {k: v / denom for k, v in totals.items()}


def format_metrics(prefix: str, row: dict[str, float]) -> str:
    return (
        f"{prefix} loss={row['loss']:.4f} dice={row['dice']:.3f} "
        f"recall={row['recall']:.3f} spec={row['specificity']:.3f} "
        f"acc={row['acc']:.3f} miou={row['miou']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on curated sky segmentation dataset")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-ch", type=int, default=32, help="U-Net base channel width")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--disk-margin",
        type=int,
        default=DEFAULT_DISK_MARGIN,
        help="Inset from fisheye rim for training mask (default 3, same as edge_filter)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip training_curves.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    valid_np = train_valid_mask((64, 64), margin=args.disk_margin)
    valid_mask = torch.from_numpy(valid_np).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"Training mask: inset fisheye disk (margin={args.disk_margin}px), "
        f"{int(valid_np.sum())}/{valid_np.size} pixels"
    )

    stems = list_paired_stems(images_dir, masks_dir)
    train_stems, val_stems, test_stems = train_val_test_split(
        stems,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    save_split_manifests(
        checkpoint_dir, train_stems, val_stems, test_stems, seed=args.seed, dataset_dir=dataset_dir
    )
    print(
        f"Dataset: {len(stems)} pairs "
        f"({len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test held out)"
    )
    print(f"  Test split saved to {checkpoint_dir / 'split_test.txt'} (not used during training)")

    train_ds = SegPairDataset(images_dir, masks_dir, train_stems)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_stems:
        val_ds = SegPairDataset(images_dir, masks_dir, val_stems)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_mask = valid_mask.to(device)
    print(f"Device: {device}")

    model = UNet64(in_channels=3, num_classes=NUM_CLASSES, base=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_val = float("inf")
    history: list[dict] = []
    t0 = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        tr = run_epoch(model, train_loader, valid_mask, device, optimizer, desc="train")
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_miou": tr["miou"],
            "train_dice": tr["dice"],
            "train_recall": tr["recall"],
            "train_specificity": tr["specificity"],
        }

        if val_loader is not None:
            va = run_epoch(model, val_loader, valid_mask, device, None, desc="val")
            row.update(
                {
                    "val_loss": va["loss"],
                    "val_acc": va["acc"],
                    "val_miou": va["miou"],
                    "val_dice": va["dice"],
                    "val_recall": va["recall"],
                    "val_specificity": va["specificity"],
                }
            )
            scheduler.step(va["loss"])
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va["loss"],
                        "val_dice": va["dice"],
                        "num_classes": NUM_CLASSES,
                        "base_ch": args.base_ch,
                        "disk_margin": args.disk_margin,
                        "seed": args.seed,
                    },
                    checkpoint_dir / "unet_best.pt",
                )
        else:
            scheduler.step(tr["loss"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": tr["loss"],
                    "num_classes": NUM_CLASSES,
                    "base_ch": args.base_ch,
                    "disk_margin": args.disk_margin,
                    "seed": args.seed,
                },
                checkpoint_dir / "unet_best.pt",
            )

        history.append(row)
        if val_loader is not None:
            epoch_bar.set_postfix(
                train_loss=f"{tr['loss']:.4f}",
                val_loss=f"{va['loss']:.4f}",
                val_dice=f"{va['dice']:.3f}",
            )
            tqdm.write(
                f"Epoch {epoch:3d}/{args.epochs}  {format_metrics('train', tr)}  |  {format_metrics('val', va)}"
            )
        else:
            epoch_bar.set_postfix(train_loss=f"{tr['loss']:.4f}", train_dice=f"{tr['dice']:.3f}")
            tqdm.write(f"Epoch {epoch:3d}/{args.epochs}  {format_metrics('train', tr)}")

    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "num_classes": NUM_CLASSES,
            "base_ch": args.base_ch,
            "disk_margin": args.disk_margin,
            "seed": args.seed,
        },
        checkpoint_dir / "unet_last.pt",
    )

    hist_path = checkpoint_dir / "train_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    curves_path = None
    if not args.no_plots:
        curves_path = save_training_plots(checkpoint_dir, history, has_val=val_loader is not None)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"  Best checkpoint: {checkpoint_dir / 'unet_best.pt'}")
    print(f"  Last checkpoint: {checkpoint_dir / 'unet_last.pt'}")
    print(f"  History: {hist_path}")
    print(f"  Splits: {checkpoint_dir / 'split_train.txt'}, split_val.txt, split_test.txt")
    if curves_path is not None:
        print(f"  Training curves: {curves_path}")


if __name__ == "__main__":
    main()
