"""
Statistics and plots for clouded vs clean sun palette patches.

  python analyze_sun_palette.py
  python analyze_sun_palette.py --dir output/sun_palette
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE_DIR = Path(__file__).resolve().parent / "output" / "sun_palette"


def nrbr(rgb: np.ndarray) -> np.ndarray:
    """Normalized red-blue ratio on RGB uint8, shape (..., 3) -> (...)"""
    r = rgb[..., 0].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    denom = r + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, (b - r) / denom, 0.0)
    return out


def luminance(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def patch_means(patches: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) -> (N, 3) mean RGB per patch."""
    return patches.reshape(patches.shape[0], -1, 3).mean(axis=1)


def detailed_stats(patches: np.ndarray, name: str) -> dict:
    """Full statistics for one label group."""
    flat = patches.reshape(-1, 3).astype(np.float64)
    per_patch_rgb = patch_means(patches)
    per_patch_lum = luminance(patches).reshape(patches.shape[0], -1).mean(axis=1)
    per_patch_nrbr = nrbr(patches).reshape(patches.shape[0], -1).mean(axis=1)
    nrbr_flat = nrbr(patches).ravel()

    def _summ(arr: np.ndarray) -> dict:
        return {
            "mean": round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
        }

    ch_names = ["R", "G", "B"]
    pixel_by_channel = {ch: _summ(flat[:, i]) for i, ch in enumerate(ch_names)}
    patch_mean_by_channel = {ch: _summ(per_patch_rgb[:, i]) for i, ch in enumerate(ch_names)}

    return {
        "label": name,
        "num_patches": int(patches.shape[0]),
        "num_pixels": int(flat.shape[0]),
        "pixel_rgb": pixel_by_channel,
        "per_patch_mean_rgb": patch_mean_by_channel,
        "pixel_luminance": _summ(luminance(patches).ravel()),
        "per_patch_mean_luminance": _summ(per_patch_lum),
        "pixel_nrbr": _summ(nrbr_flat),
        "per_patch_mean_nrbr": _summ(per_patch_nrbr),
    }


def compare_groups(clouded: np.ndarray, clean: np.ndarray) -> dict:
    """Differences clean minus clouded on key scalars."""
    sc = detailed_stats(clouded, "clouded")
    sk = detailed_stats(clean, "clean")

    def delta(path: str) -> dict:
        parts = path.split(".")
        a, b = sc, sk
        for p in parts[:-1]:
            a, b = a[p], b[p]
        key = parts[-1]
        return {
            "clouded": a[key],
            "clean": b[key],
            "delta_clean_minus_clouded": round(b[key] - a[key], 3),
        }

    return {
        "mean_R": delta("pixel_rgb.R.mean"),
        "mean_G": delta("pixel_rgb.G.mean"),
        "mean_B": delta("pixel_rgb.B.mean"),
        "mean_luminance": delta("pixel_luminance.mean"),
        "mean_nrbr": delta("pixel_nrbr.mean"),
        "per_patch_luminance_std": delta("per_patch_mean_luminance.std"),
    }


def load_palettes(palette_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    clouded_path = palette_dir / "sun_palette_clouded.npy"
    clean_path = palette_dir / "sun_palette_clean.npy"
    clouded = np.load(clouded_path) if clouded_path.exists() else None
    clean = np.load(clean_path) if clean_path.exists() else None
    if clouded is not None and clouded.size == 0:
        clouded = None
    if clean is not None and clean.size == 0:
        clean = None
    if clouded is None and clean is None:
        raise SystemExit(f"No palette .npy files in {palette_dir}")
    return clouded, clean


def plot_analysis(
    clouded: np.ndarray | None,
    clean: np.ndarray | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- 1. Mean RGB bar chart ---
    ax = axes[0, 0]
    labels = []
    r_vals, g_vals, b_vals, r_err, g_err, b_err = [], [], [], [], [], []
    colors_bar = []
    for name, patches, color in [
        ("clouded", clouded, "#c67b4a"),
        ("clean", clean, "#4caf50"),
    ]:
        if patches is None:
            continue
        flat = patches.reshape(-1, 3).astype(np.float64)
        labels.append(name)
        r_vals.append(flat[:, 0].mean())
        g_vals.append(flat[:, 1].mean())
        b_vals.append(flat[:, 2].mean())
        r_err.append(flat[:, 0].std())
        g_err.append(flat[:, 1].std())
        b_err.append(flat[:, 2].std())
        colors_bar.append(color)

    x = np.arange(len(labels))
    w = 0.25
    if labels:
        ax.bar(x - w, r_vals, w, yerr=r_err, label="R", color="#e74c3c", capsize=3)
        ax.bar(x, g_vals, w, yerr=g_err, label="G", color="#2ecc71", capsize=3)
        ax.bar(x + w, b_vals, w, yerr=b_err, label="B", color="#3498db", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.set_ylabel("pixel value (0–255)")
    ax.set_title("Mean RGB ± std (all patch pixels)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- 2. Per-patch mean luminance histogram ---
    ax = axes[0, 1]
    if clouded is not None:
        lum_c = luminance(clouded).reshape(clouded.shape[0], -1).mean(axis=1)
        ax.hist(lum_c, bins=20, alpha=0.6, label=f"clouded (n={len(lum_c)})", color="#c67b4a")
    if clean is not None:
        lum_k = luminance(clean).reshape(clean.shape[0], -1).mean(axis=1)
        ax.hist(lum_k, bins=20, alpha=0.6, label=f"clean (n={len(lum_k)})", color="#4caf50")
    ax.set_xlabel("mean luminance per patch")
    ax.set_title("Per-patch brightness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 3. NRBR pixel histogram ---
    ax = axes[1, 0]
    if clouded is not None:
        ax.hist(
            nrbr(clouded).ravel(),
            bins=40,
            range=(-1, 1),
            alpha=0.6,
            label="clouded",
            color="#c67b4a",
            density=True,
        )
    if clean is not None:
        ax.hist(
            nrbr(clean).ravel(),
            bins=40,
            range=(-1, 1),
            alpha=0.6,
            label="clean",
            color="#4caf50",
            density=True,
        )
    ax.set_xlabel("NRBR = (B−R)/(B+R)")
    ax.set_title("Pixel NRBR distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 4. Box plot: per-patch mean R,G,B ---
    ax = axes[1, 1]
    box_data, box_labels = [], []
    for name, patches in [("clouded", clouded), ("clean", clean)]:
        if patches is None:
            continue
        pm = patch_means(patches)
        for i, ch in enumerate("RGB"):
            box_data.append(pm[:, i])
            box_labels.append(f"{name}\n{ch}")
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels, showfliers=True)
    ax.set_ylabel("mean per patch")
    ax.set_title("Per-patch mean RGB")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sun palette statistics: clouded vs clean (5×5 at predicted sun)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def print_summary(report: dict) -> None:
    print("\n=== Sun palette statistics ===\n")
    for key in ("clouded", "clean"):
        if key not in report:
            continue
        s = report[key]
        print(f"{key.upper()}  ({s['num_patches']} patches, {s['num_pixels']} pixels)")
        px = s["pixel_rgb"]
        print(
            f"  Pixel RGB mean:  R={px['R']['mean']}  G={px['G']['mean']}  B={px['B']['mean']}"
        )
        print(
            f"  Pixel RGB std:   R={px['R']['std']}  G={px['G']['std']}  B={px['B']['std']}"
        )
        print(f"  Luminance mean:  {s['pixel_luminance']['mean']}")
        print(f"  NRBR mean:       {s['pixel_nrbr']['mean']}")
        print()

    if "comparison" in report:
        c = report["comparison"]
        print("CLEAN minus CLOUDED (delta):")
        print(f"  ΔR={c['mean_R']['delta_clean_minus_clouded']}  "
              f"ΔG={c['mean_G']['delta_clean_minus_clouded']}  "
              f"ΔB={c['mean_B']['delta_clean_minus_clouded']}")
        print(f"  Δluminance={c['mean_luminance']['delta_clean_minus_clouded']}")
        print(f"  ΔNRBR={c['mean_nrbr']['delta_clean_minus_clouded']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze clouded vs clean sun palettes")
    parser.add_argument("--dir", type=str, default=str(PALETTE_DIR), help="Palette output folder")
    args = parser.parse_args()
    palette_dir = Path(args.dir)

    clouded, clean = load_palettes(palette_dir)
    report: dict = {}
    if clouded is not None:
        report["clouded"] = detailed_stats(clouded, "clouded")
    if clean is not None:
        report["clean"] = detailed_stats(clean, "clean")
    if clouded is not None and clean is not None:
        report["comparison"] = compare_groups(clouded, clean)

    out_json = palette_dir / "sun_palette_analysis.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_png = palette_dir / "sun_palette_analysis.png"
    plot_analysis(clouded, clean, out_png)

    print_summary(report)
    print(f"\nSaved {out_json.name}")
    print(f"Saved {out_png.name}")


if __name__ == "__main__":
    main()
