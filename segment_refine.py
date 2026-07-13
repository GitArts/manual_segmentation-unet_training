"""
Refine NRBR segmentation using sun palette (occlusion) and Laplacian edge mask.

Pipeline:
  1. NRBR segment (first step)
  2. Laplacian edge mask
  3. Apply edge/sun rules to labels and edge mask

Sun not occluded = clean sun (LABEL_SUN) from palette luminance threshold.

Rules:
  1. Sun not occluded + edge pixels < 8% of sky disk → clear all in-disk labels (sky only)
  2. Sun not occluded + edges on sun disk → remove edge mask within 3 px of sun center
  3. Sun not occluded + sun near disk center + many edges on sun → clear all in-disk labels
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from edge_filter import edge_laplacian
from segment_nrbr import (
    DISK_CENTER_COL,
    DISK_CENTER_ROW,
    LABEL_SKY,
    LABEL_SUN,
    SUN_DISK_RADIUS,
    SegmentResult,
    fisheye_mask,
    segment,
)

EDGE_FRACTION_CLEAR = 0.06
SUN_EDGE_CLEAR_RADIUS = 3
SUN_CENTER_MAX_DIST = 10.0
SUN_EDGE_FRACTION_LOT = 0.30

class RefineResult(NamedTuple):
    labels: np.ndarray
    edge_mask: np.ndarray
    edges: np.ndarray
    segment: SegmentResult
    sun_not_occluded: bool
    global_edge_fraction: float
    sun_edge_fraction: float
    cond1_clear_all: bool
    cond2_sun_edges_removed: bool
    cond3_clear_all: bool

def sun_disk_mask(
    shape: tuple[int, int],
    sun_row: int,
    sun_col: int,
    radius: int = SUN_DISK_RADIUS,
) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (rows - sun_row) ** 2 + (cols - sun_col) ** 2 <= radius**2


def sun_not_occluded(labels: np.ndarray, sun_row: int, sun_col: int) -> bool:
    """Clean sun (palette / luminance) — not cloud-attenuated."""
    disk = sun_disk_mask(labels.shape, sun_row, sun_col)
    on_sun = labels[disk]
    if on_sun.size == 0:
        return False
    return bool(np.any(on_sun == LABEL_SUN))


def sun_in_middle(sun_row: int, sun_col: int) -> bool:
    dist = float(np.hypot(sun_row - DISK_CENTER_ROW, sun_col - DISK_CENTER_COL))
    return dist <= SUN_CENTER_MAX_DIST


def edge_fraction(edge_mask: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return 0.0
    return float(edge_mask[region].sum()) / float(region.sum())


def clear_all_labels(labels: np.ndarray) -> np.ndarray:
    """Remove cloud/sun segmentation — keep void, set disk to clear sky."""
    out = labels.copy()
    disk = fisheye_mask()
    out[disk] = LABEL_SKY
    return out


def remove_edges_near_sun(
    edge_mask: np.ndarray,
    sun_row: int,
    sun_col: int,
    radius: int = SUN_EDGE_CLEAR_RADIUS,
) -> np.ndarray:
    out = edge_mask.copy()
    near = sun_disk_mask(edge_mask.shape, sun_row, sun_col, radius=radius)
    out[near] = False
    return out


def refine_segmentation(
    labels: np.ndarray,
    edge_mask: np.ndarray,
    sun_row: int,
    sun_col: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
    """
    Apply the three edge/sun rules to NRBR labels and edge mask.

    Returns refined labels, refined edge_mask, and flags for which rules fired.
    """
    labels = labels.copy()
    edge_mask = edge_mask.copy()
    flags = {
        "cond1_clear_all": False,
        "cond2_sun_edges_removed": False,
        "cond3_clear_all": False,
    }

    disk = fisheye_mask()
    sun_disk = sun_disk_mask(labels.shape, sun_row, sun_col)
    not_occluded = sun_not_occluded(labels, sun_row, sun_col)

    global_edge_frac = edge_fraction(edge_mask, disk)
    sun_edge_frac = edge_fraction(edge_mask, sun_disk)
    edges_on_sun = sun_edge_frac > 0.0

    if not_occluded and sun_in_middle(sun_row, sun_col) and sun_edge_frac >= SUN_EDGE_FRACTION_LOT:
        labels = clear_all_labels(labels)
        flags["cond3_clear_all"] = True
    elif not_occluded and global_edge_frac < EDGE_FRACTION_CLEAR:
        labels = clear_all_labels(labels)
        flags["cond1_clear_all"] = True

    if not_occluded and edges_on_sun:
        edge_mask = remove_edges_near_sun(edge_mask, sun_row, sun_col)
        flags["cond2_sun_edges_removed"] = True

    return labels, edge_mask, flags


def segment_and_refine(
    rgb: np.ndarray,
    sun_row: int,
    sun_col: int,
    *,
    disk_margin: int = 3,
) -> RefineResult:
    """NRBR segment → edges → refine labels and edge mask."""
    seg = segment(rgb, sun_row, sun_col)
    edges, edge_mask = edge_laplacian(rgb, mask_disk=True, disk_margin=disk_margin, return_mask=True)

    disk = fisheye_mask()
    sun_disk = sun_disk_mask(rgb.shape[:2], sun_row, sun_col)
    not_occluded = sun_not_occluded(seg.labels, sun_row, sun_col)
    global_frac = edge_fraction(edge_mask, disk)
    sun_frac = edge_fraction(edge_mask, sun_disk)

    refined_labels, refined_edge_mask, flags = refine_segmentation(
        seg.labels, edge_mask, sun_row, sun_col
    )
    refined_edges = np.where(refined_edge_mask, edges, 0.0)

    return RefineResult(
        labels=refined_labels,
        edge_mask=refined_edge_mask,
        edges=refined_edges,
        segment=seg,
        sun_not_occluded=not_occluded,
        global_edge_fraction=global_frac,
        sun_edge_fraction=sun_frac,
        cond1_clear_all=flags["cond1_clear_all"],
        cond2_sun_edges_removed=flags["cond2_sun_edges_removed"],
        cond3_clear_all=flags["cond3_clear_all"],
    )
