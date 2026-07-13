"""
Demo Laplacian edge filter on random sky images.

  python demo_edges.py -n 6
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edge_filter import (
    DEFAULT_DISK_MARGIN,
    LAPLACIAN_3X3,
    edge_laplacian,
    edge_overlay,
)
from skippd_io import DATA_DIR, collect_jpg_paths, datetime_from_filename, load_rgb_64

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "edge_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Laplacian edge detection demo")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_DISK_MARGIN,
        help="Pixels inset from fisheye rim to ignore (default 3)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = collect_jpg_paths(data_dir)
    random.seed(args.seed)
    sample = random.sample(paths, min(args.n, len(paths)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Laplacian kernel:\n{LAPLACIAN_3X3.astype(int)}")
    print(f"Sky disk margin (ignore lens rim): {args.margin} px\n")

    saved = []
    for i, path in enumerate(sample, 1):
        img = load_rgb_64(path)
        t = datetime_from_filename(path.name)
        edges_raw = edge_laplacian(img, mask_disk=False)
        edges, edge_mask = edge_laplacian(
            img,
            mask_disk=True,
            disk_margin=args.margin,
            return_mask=True,
        )
        overlay = edge_overlay(img, edges, edge_mask=edge_mask)

        fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
        axes[0].imshow(img)
        axes[0].set_title(t.strftime("%Y-%m-%d %H:%M"), fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(edges_raw, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("edges (no mask)", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(edge_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("edge mask (≥150)", fontsize=9)
        axes[2].axis("off")

        axes[3].imshow(edges, cmap="gray", vmin=0, vmax=255)
        axes[3].set_title("edges × mask", fontsize=9)
        axes[3].axis("off")

        axes[4].imshow(overlay)
        axes[4].set_title("overlay", fontsize=9)
        axes[4].axis("off")

        fig.suptitle(path.name, fontsize=8)
        fig.tight_layout()
        out = OUTPUT_DIR / f"{i:02d}_{path.stem}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"  [{i}/{len(sample)}] {path.name}")

    imgs = [plt.imread(p) for p in saved]
    cols = min(2, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    for ax in axes[len(imgs) :]:
        ax.axis("off")
    fig.suptitle("Laplacian edges — lens rim removed by inset fisheye mask", fontsize=10)
    fig.tight_layout()
    grid = OUTPUT_DIR / "_summary.png"
    fig.savefig(grid, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {grid}")


if __name__ == "__main__":
    main()
