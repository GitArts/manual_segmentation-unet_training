"""
Load random sky images from the local SKIPP'D dataset.

Dataset layout (same as Cloud-dection-in-sky-images/codes/run_skippd_cloud_demo.py):
  SkyGPT/Codes/CloudPred-PV/Data/
    2017_03_images_raw/03/01/20170301060000.jpg
    2017_05_images_raw/05/20/20170520120000.jpg
    ...

Each filename is YYYYMMDDHHMMSS (1-minute cadence during daylight).
Images are fisheye sky photos from Stanford; we load them as 64x64 RGB uint8
(the resolution used throughout the LACISE cloud-detection pipeline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skippd_io import (
    CLOUD_DEMO_MANIFEST,
    DATA_DIR,
    datetime_from_filename,
    load_rgb_64,
    resolve_image_sample,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "sample_images"
CLOUD_CODES = Path(__file__).resolve().parent.parent / "Cloud-dection-in-sky-images" / "codes"
sys.path.insert(0, str(CLOUD_CODES))
from sun_position_identification import sun_position  # noqa: E402


def overlay_sun(image: np.ndarray, sun_mask: np.ndarray) -> np.ndarray:
    """Mark sun pixels in red (same as run_skippd_cloud_demo / cloud_detection)."""
    out = image.copy()
    out[sun_mask[:, :, 0] > 0] = [255, 0, 0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Load random SKIPP'D sky images")
    parser.add_argument("-n", type=int, default=8, help="Number of random images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help=f"Use exact paths from file (default: {CLOUD_DEMO_MANIFEST.name} if present)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Data root: {data_dir}")
    manifest = Path(args.manifest) if args.manifest else None
    sample = resolve_image_sample(data_dir, args.n, args.seed, manifest_path=manifest)
    if manifest is None and CLOUD_DEMO_MANIFEST.is_file():
        print(f"Using cloud-demo manifest ({len(sample)} images): {CLOUD_DEMO_MANIFEST}")
    else:
        print(f"Using deterministic sample (seed={args.seed}, n={len(sample)})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images: list[np.ndarray] = []
    overlays: list[np.ndarray] = []
    sun_coords: list[tuple[int, int]] = []
    print(f"\nLoading {len(sample)} random images + sun position:")
    for i, path in enumerate(sample, 1):
        t = datetime_from_filename(path.name)
        img = load_rgb_64(path)
        sun_x, sun_y, sun_mask = sun_position(t)
        images.append(img)
        overlays.append(overlay_sun(img, sun_mask))
        sun_coords.append((sun_x, sun_y))
        rel = path.relative_to(data_dir)
        print(
            f"  [{i}] {rel}\n"
            f"      time={t:%Y-%m-%d %H:%M:%S}  sun=({sun_x}, {sun_y})  "
            f"shape={img.shape}  min/max={img.min()}/{img.max()}"
        )

    # Two-row grid: originals on top, sun-marked (red) on bottom.
    n = len(images)
    cols = min(4, n)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    axes = np.atleast_2d(axes)

    for col in range(cols):
        t = datetime_from_filename(sample[col].name)
        sun_x, sun_y = sun_coords[col]
        axes[0, col].imshow(images[col])
        axes[0, col].set_title(t.strftime("%Y-%m-%d %H:%M"), fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(overlays[col])
        axes[1, col].set_title(f"sun ({sun_x}, {sun_y})", fontsize=9)
        axes[1, col].axis("off")
    for col in range(n, cols):
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    fig.suptitle("SKIPP'D images with sun position (red, sun_position_identification)", fontsize=11)
    fig.tight_layout()
    grid_path = OUTPUT_DIR / "random_sample_with_sun.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved grid -> {grid_path}")


if __name__ == "__main__":
    main()
