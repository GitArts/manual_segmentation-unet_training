"""
Second-pass audit of saved dataset pairs (images/ + masks/).

Leave = keep the pair. Delete = remove both PNG files from disk.

  python audit_dataset.py
  python audit_dataset.py --dataset-dir output/dataset --dataset-dir output/dataset_round2
  python audit_dataset.py --seed 42 --shuffle
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from audit_ui import AuditItem, run_audit_ui
from lib.sun_position_identification import sun_position
from review_ui import setup_matplotlib_backend
from seg_dataset import list_paired_stems
from skippd_io import datetime_from_filename

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIRS = [
    ROOT / "output" / "dataset",
    ROOT / "output" / "dataset_round2",
]


def collect_pairs(dataset_dirs: list[Path]) -> list[tuple[Path, Path, Path]]:
    """Return (image_path, mask_path, dataset_dir) for every paired PNG."""
    pairs: list[tuple[Path, Path, Path]] = []
    for dataset_dir in dataset_dirs:
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            print(f"Skipping (missing images/ or masks/): {dataset_dir}")
            continue
        try:
            stems = list_paired_stems(images_dir, masks_dir)
        except FileNotFoundError:
            print(f"Skipping (no pairs): {dataset_dir}")
            continue
        for stem in stems:
            pairs.append((images_dir / f"{stem}.png", masks_dir / f"{stem}.png", dataset_dir))
    return pairs


def load_audit_items(
    pairs: list[tuple[Path, Path, Path]],
) -> list[AuditItem]:
    items: list[AuditItem] = []
    for image_path, mask_path, dataset_dir in pairs:
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        labels = np.array(Image.open(mask_path), dtype=np.uint8)
        t = datetime_from_filename(image_path.name)
        sx, sy, _ = sun_position(t)
        items.append(
            AuditItem(
                image_path=image_path,
                mask_path=mask_path,
                dataset_dir=dataset_dir,
                timestamp=t,
                img=img,
                labels=labels,
                sun_row=int(sx),
                sun_col=int(sy),
            )
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved dataset pairs — leave (keep) or delete from disk"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        action="append",
        dest="dataset_dirs",
        help="Dataset root with images/ and masks/ (repeatable; default: dataset + dataset_round2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed when --shuffle is set")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize order (default: sorted by filename)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Where to write audit_kept_manifest.txt and audit_deleted_manifest.txt",
    )
    args = parser.parse_args()

    dataset_dirs = [Path(p) for p in args.dataset_dirs] if args.dataset_dirs else DEFAULT_DATASET_DIRS
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "output"

    pairs = collect_pairs(dataset_dirs)
    if not pairs:
        raise SystemExit("No paired images/masks found in the given dataset directories.")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    print(f"Loading {len(pairs)} paired samples for audit...")
    for d in dataset_dirs:
        n = sum(1 for _, _, dd in pairs if dd == d)
        if n:
            print(f"  {d}: {n}")

    setup_matplotlib_backend(interactive=True)
    items = load_audit_items(pairs)
    run_audit_ui(
        items,
        title="Dataset audit — Leave / Delete",
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
