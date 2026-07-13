"""Single 3×3 convolution filter for edge detection on sky images."""

from __future__ import annotations

import numpy as np

from segment_nrbr import DISK_CENTER_COL, DISK_CENTER_ROW, DISK_RADIUS

LAPLACIAN_3X3 = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

DEFAULT_DISK_MARGIN = 3
NORMALIZE_PERCENTILE = 99.5
EDGE_MASK_THRESHOLD = 100.0  # after 0–255 scaling, drop pixels below this


def sky_disk_mask(shape: tuple[int, int] = (64, 64), margin: int = DEFAULT_DISK_MARGIN) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    inner_r = max(1, DISK_RADIUS - margin)
    return (rows - DISK_CENTER_ROW) ** 2 + (cols - DISK_CENTER_COL) ** 2 <= inner_r**2


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = padded[i : i + kh, j : j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def build_edge_mask(
    edges: np.ndarray,
    disk: np.ndarray,
    *,
    threshold: float = EDGE_MASK_THRESHOLD,
) -> np.ndarray:
    """Binarize scaled edge map: mask pixel iff inside disk and edge value >= threshold."""
    return disk & (edges >= threshold)


def edge_laplacian(
    rgb: np.ndarray,
    *,
    normalize: bool = True,
    mask_disk: bool = True,
    disk_margin: int = DEFAULT_DISK_MARGIN,
    norm_percentile: float = NORMALIZE_PERCENTILE,
    mask_threshold: float = EDGE_MASK_THRESHOLD,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Laplacian edge map with boolean edge mask (fixed threshold after scaling)."""
    gray = to_grayscale(rgb)
    response = convolve2d(gray, LAPLACIAN_3X3)
    edges = np.abs(response)
    disk = sky_disk_mask(edges.shape, disk_margin)

    if normalize:
        ref = edges[disk] if disk.any() else edges.ravel()
        peak = float(np.percentile(ref, norm_percentile)) if ref.size else float(edges.max())
        if peak <= 0:
            peak = 1.0
        edges = np.clip(edges / peak * 255.0, 0, 255.0)

    edges = edges.copy()
    if mask_disk:
        edges[~disk] = 0

    edge_mask = build_edge_mask(edges, disk, threshold=mask_threshold)
    edges = np.where(edge_mask, edges, 0.0)

    if return_mask:
        return edges, edge_mask
    return edges


def edge_overlay(
    rgb: np.ndarray,
    edges: np.ndarray,
    *,
    edge_mask: np.ndarray | None = None,
    alpha: float = 0.65,
) -> np.ndarray:
    """Draw edges using the boolean edge mask."""
    out = rgb.astype(np.float64).copy()
    if edge_mask is None:
        edge_mask = edges > 0
    out[edge_mask] = (1 - alpha) * out[edge_mask] + alpha * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
