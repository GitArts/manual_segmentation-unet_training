"""

Combined NRBR + sun palette + edge refinement demo.



Batch visualization (no GUI):

  python demo_combined.py -n 20 --seed 42



Interactive dataset curation (images + masks in separate folders):

  python demo_combined.py --review -n 150 --seed 42 --start-index 0 --no-manifest
  # round 2 predict: predict_review.py -n 150 --seed 42 --start-index 150

"""



from __future__ import annotations



import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from edge_filter import edge_overlay
from lib.sun_position_identification import sun_position
from review_ui import ReviewItem, run_review_ui, setup_matplotlib_backend

from segment_nrbr import colorize_labels, load_sun_lum_threshold, overlay_labels, segment

from segment_refine import segment_and_refine

from skippd_io import (

    CLOUD_DEMO_MANIFEST,

    DATA_DIR,

    datetime_from_filename,

    load_rgb_64,

    resolve_image_sample,

    sample_jpg_paths,

)



OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "combined_demo"

DATASET_DIR = Path(__file__).resolve().parent / "output" / "dataset"




def flag_text(result) -> str:

    parts = []

    if result.cond1_clear_all:

        parts.append("C1: clear (few edges)")

    if result.cond2_sun_edges_removed:

        parts.append("C2: sun edges cleared")

    if result.cond3_clear_all:

        parts.append("C3: clear (sun glare)")

    if not parts:

        parts.append("no rule fired")

    occ = "clean sun" if result.sun_not_occluded else "clouded sun"

    parts.append(f"{occ}, edges={result.global_edge_fraction:.1%}")

    return " | ".join(parts)





def resolve_sample(
    data_dir: Path,
    n: int,
    seed: int,
    manifest: Path | None,
    no_manifest: bool,
    start_index: int = 0,
) -> list[Path]:
    if no_manifest:
        return sample_jpg_paths(data_dir, n, seed, start_index=start_index)
    return resolve_image_sample(
        data_dir, n, seed, manifest_path=manifest, start_index=start_index
    )





def prepare_items(sample: list[Path]) -> list[ReviewItem]:

    items: list[ReviewItem] = []

    for i, path in enumerate(sample, 1):

        img = load_rgb_64(path)

        t = datetime_from_filename(path.name)

        sx, sy, _ = sun_position(t)

        sun_row, sun_col = int(sx), int(sy)

        result = segment_and_refine(img, sun_row, sun_col)

        items.append(

            ReviewItem(

                path=path,

                timestamp=t,

                img=img,

                labels=result.labels,

                sun_row=sun_row,

                sun_col=sun_col,

                status_text=flag_text(result),

            )

        )

        print(f"  [{i}/{len(sample)}] segmented {path.name} — {flag_text(result)}")

    return items





def run_batch_demo(sample: list[Path], output_dir: Path) -> None:

    import matplotlib.pyplot as plt



    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for i, path in enumerate(sample, 1):

        img = load_rgb_64(path)

        t = datetime_from_filename(path.name)

        sx, sy, _ = sun_position(t)

        sun_row, sun_col = int(sx), int(sy)



        seg = segment(img, sun_row, sun_col)

        result = segment_and_refine(img, sun_row, sun_col)



        fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))



        axes[0].imshow(img)

        axes[0].plot(sun_col, sun_row, "r+", ms=8, mew=1.5)

        axes[0].set_title(t.strftime("%Y-%m-%d %H:%M"), fontsize=8)

        axes[0].axis("off")



        axes[1].imshow(colorize_labels(seg.labels))

        axes[1].set_title("NRBR (raw)", fontsize=8)

        axes[1].axis("off")



        axes[2].imshow(result.edge_mask, cmap="gray", vmin=0, vmax=1)

        axes[2].set_title("edge mask", fontsize=8)

        axes[2].axis("off")



        axes[3].imshow(colorize_labels(result.labels))

        axes[3].set_title("NRBR refined", fontsize=8)

        axes[3].axis("off")



        axes[4].imshow(overlay_labels(img, result.labels))

        axes[4].set_title("refined overlay", fontsize=8)

        axes[4].axis("off")



        axes[5].imshow(edge_overlay(img, result.edges, edge_mask=result.edge_mask))

        axes[5].set_title("edges overlay", fontsize=8)

        axes[5].axis("off")



        fig.suptitle(f"{path.name}\n{flag_text(result)}", fontsize=7)

        fig.tight_layout()

        out = output_dir / f"{i:02d}_{path.stem}.png"

        fig.savefig(out, dpi=120, bbox_inches="tight")

        plt.close(fig)

        saved.append(out)

        print(f"  [{i}/{len(sample)}] {path.name} — {flag_text(result)}")



    imgs = [plt.imread(p) for p in saved]

    cols = min(2, len(imgs))

    rows = (len(imgs) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.5 * rows))

    axes = np.atleast_1d(axes).ravel()

    for ax, im in zip(axes, imgs):

        ax.imshow(im)

        ax.axis("off")

    for ax in axes[len(imgs) :]:

        ax.axis("off")

    fig.suptitle("NRBR → edge refine (C1/C2/C3 rules)", fontsize=10)

    fig.tight_layout()

    grid = output_dir / "_summary.png"

    fig.savefig(grid, dpi=100, bbox_inches="tight")

    plt.close(fig)

    print(f"\nSaved -> {grid}")





def main() -> None:

    parser = argparse.ArgumentParser(description="NRBR + edge refinement demo")

    parser.add_argument("-n", type=int, default=8, help="Number of images (use 150 for dataset review)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index in seed-shuffled list (0=image 1, 150=image 151)",
    )

    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))

    parser.add_argument(

        "--manifest",

        type=str,

        default="",

        help=f"Use exact paths from file (default: {CLOUD_DEMO_MANIFEST.name} if present)",

    )

    parser.add_argument(

        "--no-manifest",

        action="store_true",

        help="Ignore cloud-demo manifest; sample -n images with --seed",

    )

    parser.add_argument(

        "--review",

        action="store_true",

        help="Interactive UI: approve correct segmentations into dataset/images and dataset/masks",

    )

    parser.add_argument(

        "--dataset-dir",

        type=str,

        default=str(DATASET_DIR),

        help="Output root for approved images/ and masks/ (review mode)",

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default=str(OUTPUT_DIR),

        help="Output folder for batch demo PNGs",

    )

    args = parser.parse_args()



    setup_matplotlib_backend(interactive=args.review)



    data_dir = Path(args.data_dir)

    manifest = Path(args.manifest) if args.manifest else None

    sample = resolve_sample(
        data_dir, args.n, args.seed, manifest, args.no_manifest, args.start_index
    )

    if args.no_manifest:
        first = args.start_index + 1
        last = args.start_index + len(sample)
        print(
            f"Using shuffle sample (seed={args.seed}, "
            f"indices {first}..{last}, n={len(sample)})"
        )

    elif manifest is None and CLOUD_DEMO_MANIFEST.is_file():

        print(f"Using cloud-demo manifest: {CLOUD_DEMO_MANIFEST} ({len(sample)} images)")

    else:

        print(f"Using deterministic sample (seed={args.seed}, n={len(sample)})")



    lum_thr = load_sun_lum_threshold()

    print(f"Sun luminance threshold: {lum_thr:.1f}")



    if args.review:

        print(f"Segmenting {len(sample)} images before review…\n")

        items = prepare_items(sample)

        run_review_ui(items, Path(args.dataset_dir), title="NRBR review — Correct / Incorrect")

    else:

        print(f"Refining {len(sample)} images…\n")

        run_batch_demo(sample, Path(args.output_dir))





if __name__ == "__main__":

    main()


