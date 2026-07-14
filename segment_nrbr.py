"""
NRBR-based sky image segmentation.

Classes (match SkyGPT convention where possible):
  0 void   — outside fisheye disk / black border
  1 sky    — clear blue sky (high NRBR)
  2 cloud  — low NRBR, not sun disk
  3 sun    — predicted sun disk, clean (high luminance)
  4 sun_clouded — sun disk, cloud-attenuated (from palette stats)

NRBR = (B - R) / (B + R)   using RGB channels only (no green).
Sun position from lib/sun_position_identification (timestamp in filename).
Luminance threshold for clean vs clouded sun from sun_palette_analysis.json if present.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Fisheye disk in 64×64 frame (row, col) — same as cloud_detection.py
DISK_CENTER_ROW = 29
DISK_CENTER_COL = 30
DISK_RADIUS = 29
SUN_DISK_RADIUS = 3  # ~5×5 patch at predicted center

# NRBR = (B-R)/(B+R): clear sky is blue → B>R → positive
SKY_NRBR_MIN = 0.10
CLOUD_NRBR_MAX = 0.10  # below this (and not sun) → cloud

DEFAULT_SUN_LUM_THRESHOLD = 194.0  # midpoint from your palette stats

LABEL_VOID = 0
LABEL_SKY = 1
LABEL_CLOUD = 2
LABEL_SUN = 3
LABEL_SUN_CLOUDED = 4

LABEL_NAMES = {
    0: "void",
    1: "sky",
    2: "cloud",
    3: "sun",
    4: "sun_clouded",
}

# RGB overlay colors (0–255)
LABEL_RGB = np.array(
    [
        [0, 0, 0],
        [68, 153, 255],
        [220, 220, 230],
        [255, 200, 0],
        [255, 120, 0],
    ],
    dtype=np.uint8,
)

PALETTE_STATS = Path(__file__).resolve().parent / "output" / "sun_palette" / "sun_palette_analysis.json"


class SegmentResult(NamedTuple):
    labels: np.ndarray  # (64, 64) int
    nrbr: np.ndarray  # (64, 64) float
    luminance: np.ndarray  # (64, 64) float


def nrbr_map(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    denom = r + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, (b - r) / denom, 0.0)
    return out


def luminance_map(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def fisheye_mask(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (rows - DISK_CENTER_ROW) ** 2 + (cols - DISK_CENTER_COL) ** 2 <= DISK_RADIUS ** 2


def load_sun_lum_threshold(stats_path: Path | None = None) -> float:
    path = stats_path or PALETTE_STATS
    if not path.is_file():
        return DEFAULT_SUN_LUM_THRESHOLD
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "clouded" not in data or "clean" not in data:
        return DEFAULT_SUN_LUM_THRESHOLD
    lc = data["clouded"]["pixel_luminance"]["mean"]
    lk = data["clean"]["pixel_luminance"]["mean"]
    return (lc + lk) / 2.0


def segment(
    rgb: np.ndarray,
    sun_row: int,
    sun_col: int,
    *,
    sky_nrbr_min: float = SKY_NRBR_MIN,
    cloud_nrbr_max: float = CLOUD_NRBR_MAX,
    sun_lum_threshold: float | None = None,
) -> SegmentResult:
    """
    Segment 64×64 RGB image.

    sun_row, sun_col: predicted sun center (sun_position_identification convention).
    """
    if sun_lum_threshold is None:
        sun_lum_threshold = load_sun_lum_threshold()

    nrbr = nrbr_map(rgb)
    lum = luminance_map(rgb)
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)

    disk = fisheye_mask()
    rows = np.arange(64)[:, None]
    cols = np.arange(64)[None, :]
    sun_disk = (rows - sun_row) ** 2 + (cols - sun_col) ** 2 <= SUN_DISK_RADIUS ** 2

    labels[~disk] = LABEL_VOID

    # Sun disk (overrides NRBR)
    on_sun = disk & sun_disk
    labels[on_sun & (lum >= sun_lum_threshold)] = LABEL_SUN
    labels[on_sun & (lum < sun_lum_threshold)] = LABEL_SUN_CLOUDED

    rest = disk & ~sun_disk
    labels[rest & (nrbr >= sky_nrbr_min)] = LABEL_SKY
    labels[rest & (nrbr < cloud_nrbr_max)] = LABEL_CLOUD
    # gap between sky_nrbr_min and cloud_nrbr_max if equal: cloud wins for nrbr < threshold

    return SegmentResult(labels=labels, nrbr=nrbr, luminance=lum)


def overlay_labels(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float64).copy()
    for lid in range(1, len(LABEL_RGB)):
        m = labels == lid
        if not m.any():
            continue
        color = LABEL_RGB[lid].astype(np.float64)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    return LABEL_RGB[labels]
