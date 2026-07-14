"""
Run trained U-Net on new SKIPP'D images, then review predictions in the GUI.

Approved pairs are appended to --dataset-dir (default: output/dataset_round2).

  python predict_review.py --continue -n 1000
  python predict_review.py -n 1000 --seed 42 --start-index 450
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from disk_mask import mask_labels_outside_disk, zero_outside_fisheye
from lib.sun_position_identification import sun_position
from review_ui import ReviewItem, run_review_ui, setup_matplotlib_backend
from skippd_io import (
    DATA_DIR,
    datetime_from_filename,
    load_rgb_64,
    load_segmentation_progress,
    record_segmentation_batch,
    resolve_image_sample,
    sample_jpg_paths,
)
from unet_model import UNet64, predict_labels

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "output" / "checkpoints" / "unet_best.pt"
DEFAULT_DATASET_OUT = ROOT / "output" / "dataset_round2"
DEFAULT_TRAIN_DATASET = ROOT / "output" / "dataset"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNet64, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_ch = int(ckpt.get("base_ch", 32))
    num_classes = int(ckpt.get("num_classes", 5))
    disk_margin = int(ckpt.get("disk_margin", 3))
    model = UNet64(in_channels=3, num_classes=num_classes, base=base_ch)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, disk_margin


def collect_exclude_stems(exclude_dirs: list[Path]) -> set[str]:
    stems: set[str] = set()
    for d in exclude_dirs:
        images_dir = d / "images"
        if images_dir.is_dir():
            stems.update(p.stem for p in images_dir.glob("*.png"))
        elif d.is_dir():
            stems.update(p.stem for p in d.glob("*.png"))
    return stems


def sample_new_paths(
    data_dir: Path,
    n: int,
    seed: int,
    start_index: int,
    manifest: Path | None,
    no_manifest: bool,
    exclude_stems: set[str],
) -> list[Path]:
    if no_manifest:
        paths = sample_jpg_paths(data_dir, n, seed, start_index=start_index)
    else:
        paths = resolve_image_sample(
            data_dir, n, seed, manifest_path=manifest, start_index=start_index
        )

    if not exclude_stems:
        return paths

    picked = [p for p in paths if p.stem not in exclude_stems]
    if len(picked) < n:
        overlap = n - len(picked)
        print(
            f"Warning: {overlap} sampled image(s) overlap excluded stems; "
            f"using {len(picked)} non-overlapping from this slice."
        )
    return picked


@torch.no_grad()
def predict_one(
    model: UNet64,
    img_rgb: np.ndarray,
    device: torch.device,
    disk_margin: int,
) -> np.ndarray:
    img_rgb = zero_outside_fisheye(img_rgb)
    x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = x.to(device)
    labels = predict_labels(model, x).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask_labels_outside_disk(labels, margin=disk_margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net inference + interactive dataset review")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("-n", type=int, default=1000, help="Number of images to predict and review")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (same as round 1)")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="0-based index in seed-shuffled list (default: from progress file with --continue)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_batch",
        action="store_true",
        help="Use next_start_index from output/segmentation_progress.json",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument(
        "--use-manifest",
        action="store_true",
        help="Use cloud-demo manifest instead of shuffle+slice",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_OUT),
        help="Where approved pairs are saved",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=str,
        action="append",
        default=None,
        help="Skip stems already in these dataset folders (default: dataset + dataset_round2)",
    )
    args = parser.parse_args()

    if args.start_index is None:
        progress = load_segmentation_progress(args.seed)
        args.start_index = int(progress["next_start_index"])
        if args.continue_batch:
            print(f"Continuing from saved progress: start_index={args.start_index}")
        else:
            print(f"Using start_index={args.start_index} (from progress file or default)")

    setup_matplotlib_backend(interactive=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    exclude_dirs = [Path(p) for p in (args.exclude_dataset or [str(DEFAULT_TRAIN_DATASET), str(DEFAULT_DATASET_OUT)])]
    exclude_stems = collect_exclude_stems(exclude_dirs)
    if exclude_stems:
        print(f"Excluding {len(exclude_stems)} stems from prior datasets (safety check)")

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else None
    no_manifest = not args.use_manifest
    sample = sample_new_paths(
        data_dir,
        args.n,
        args.seed,
        args.start_index,
        manifest,
        no_manifest,
        exclude_stems,
    )
    first_idx = args.start_index + 1
    last_idx = args.start_index + len(sample)
    print(
        f"Predicting {len(sample)} images "
        f"(seed={args.seed}, shuffle indices {first_idx}..{last_idx})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, disk_margin = load_model(checkpoint_path, device)

    items: list[ReviewItem] = []
    for path in tqdm(sample, desc="Predicting", unit="img"):
        img = zero_outside_fisheye(load_rgb_64(path))
        t = datetime_from_filename(path.name)
        sx, sy, _ = sun_position(t)
        sun_row, sun_col = int(sx), int(sy)
        labels = predict_one(model, img, device, disk_margin)
        items.append(
            ReviewItem(
                path=path,
                timestamp=t,
                img=img,
                labels=labels,
                sun_row=sun_row,
                sun_col=sun_col,
                status_text="U-Net prediction",
            )
        )

    dataset_dir = Path(args.dataset_dir)
    run_review_ui(
        items,
        dataset_dir,
        title="U-Net review — Correct / Fix mask / Skip",
        append=True,
    )

    next_index = record_segmentation_batch(
        args.seed,
        args.start_index,
        len(sample),
        dataset_dir=str(dataset_dir),
    )
    print(f"Next batch should use --start-index {next_index} (saved to segmentation_progress.json)")


if __name__ == "__main__":
    main()
