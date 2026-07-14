"""
Demo NRBR segmentation on random SKIPP'D images.

  python demo_segment.py -n 8
  python demo_segment.py --compare  # include Nie et al. cloud_detection mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.cloud_detection import cloud_detection
from lib.sun_position_identification import sun_position
from segment_nrbr import colorize_labels, load_sun_lum_threshold, overlay_labels, segment
from skippd_io import (
    CLOUD_DEMO_MANIFEST,
    DATA_DIR,
    datetime_from_filename,
    load_rgb_64,
    resolve_image_sample,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "segment_demo"


def main() -> None:
    parser = argparse.ArgumentParser(description="NRBR segmentation demo")
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help=f"Use exact paths from file (default: {CLOUD_DEMO_MANIFEST.name} if present)",
    )
    parser.add_argument("--compare", action="store_true", help="Add cloud_detection.py panel")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else None
    sample = resolve_image_sample(data_dir, args.n, args.seed, manifest_path=manifest)
    if manifest is None and CLOUD_DEMO_MANIFEST.is_file():
        print(f"Using cloud-demo manifest: {CLOUD_DEMO_MANIFEST}")
    else:
        print(f"Using deterministic sample (seed={args.seed}, n={len(sample)})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lum_thr = load_sun_lum_threshold()
    print(f"Sun luminance threshold (clean vs clouded): {lum_thr:.1f}")
    print(f"Segmenting {len(sample)} images…")

    rows = []
    for i, path in enumerate(sample, 1):
        img = load_rgb_64(path)
        t = datetime_from_filename(path.name)
        sx, sy, _ = sun_position(t)
        result = segment(img, int(sx), int(sy))
        overlay = overlay_labels(img, result.labels)

        ncols = 4 if args.compare else 3
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.5))
        if ncols == 1:
            axes = [axes]

        axes[0].imshow(img)
        axes[0].set_title(t.strftime("%Y-%m-%d %H:%M"), fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(colorize_labels(result.labels))
        axes[1].set_title("NRBR seg", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("overlay", fontsize=9)
        axes[2].axis("off")

        if args.compare:
            _, cloud_mask, sun_mask = cloud_detection(t, img)
            cloud = cloud_mask[:, :, 1] > 0
            sun = sun_mask[:, :, 0] > 0
            blend = img.copy()
            blend[cloud] = (0.5 * blend[cloud] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            blend[sun] = [255, 0, 0]
            axes[3].imshow(blend)
            axes[3].set_title("Nie NRBR+CSL", fontsize=9)
            axes[3].axis("off")

        fig.suptitle(path.name, fontsize=8)
        fig.tight_layout()
        out_one = OUTPUT_DIR / f"{i:02d}_{path.stem}.png"
        fig.savefig(out_one, dpi=120, bbox_inches="tight")
        plt.close(fig)
        rows.append(out_one)
        print(f"  [{i}/{len(sample)}] {path.name}")

    # summary grid
    imgs = [plt.imread(p) for p in rows]
    cols = min(2, len(imgs))
    rows_n = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.5 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, im in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    for ax in axes[len(imgs) :]:
        ax.axis("off")
    legend = "Labels: blue=sky  grey=cloud  yellow=sun  orange=sun_clouded"
    fig.suptitle(f"NRBR segmentation — {legend}", fontsize=10)
    fig.tight_layout()
    grid = OUTPUT_DIR / "_summary.png"
    fig.savefig(grid, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {grid}")


if __name__ == "__main__":
    main()
