"""Shared SKIPP'D image loading (no matplotlib — safe for interactive tools)."""

from __future__ import annotations

import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from paths import CACHE_DIR, DATA_DIR, DEFAULT_MANIFEST, IMAGE_INDEX_PICKLE, ROOT

MANUAL_SEG_ROOT = ROOT
PATH_CACHE_DIR = CACHE_DIR

# Optional fixed image list (e.g. write_cloud_manifest.py -> data/image_manifest.txt)
CLOUD_DEMO_MANIFEST = DEFAULT_MANIFEST

# Clear-sky reference days from Nie et al. / cloud-detection README
CLEAR_DAYS = [(5, 20), (8, 15), (9, 23), (10, 22)]

PATCH_SIZE = 5
PATCH_HALF = PATCH_SIZE // 2


def datetime_from_filename(name: str) -> datetime:
    return datetime.strptime(Path(name).stem, "%Y%m%d%H%M%S")


def load_rgb_64(path: Path) -> np.ndarray:
    """Load JPG as 64×64 RGB uint8 (Pillow — avoids OpenCV Qt conflicts in WSL)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def _path_sort_key(path: Path) -> str:
    """Fast stable sort key — avoids pathlib.resolve() on 131k WSL paths."""
    return str(path).replace("\\", "/").lower()


def _path_cache_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def _sorted_manifest_paths() -> tuple[Path, Path]:
    PATH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = PATH_CACHE_DIR / "sorted_jpg_paths.manifest.txt"
    meta = PATH_CACHE_DIR / "sorted_jpg_paths.meta.json"
    return manifest, meta


def _shuffled_manifest_paths(seed: int) -> tuple[Path, Path]:
    PATH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = PATH_CACHE_DIR / f"shuffled_seed_{seed}.manifest.txt"
    meta = PATH_CACHE_DIR / f"shuffled_seed_{seed}.meta.json"
    return manifest, meta


def _read_path_meta(meta_path: Path) -> dict | None:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_path_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_path_manifest(paths: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p).replace("\\", "/") for p in paths]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_jpg_paths_from_index() -> list[Path] | None:
    """Load JPG paths from cache/image_index.pkl if available (unsorted)."""
    if not IMAGE_INDEX_PICKLE.is_file():
        return None
    with open(IMAGE_INDEX_PICKLE, "rb") as f:
        pairs = pickle.load(f)
    return [Path(p) for p, _ in pairs]


def collect_jpg_paths(data_dir: Path) -> list[Path]:
    cached = collect_jpg_paths_from_index()
    if cached is not None:
        return cached
    paths: list[Path] = []
    for month_dir in sorted(data_dir.glob("*_images_raw")):
        paths.extend(month_dir.rglob("*.jpg"))
    return paths


def _load_path_manifest_fast(manifest_path: Path) -> list[Path]:
    """Load manifest lines as Paths without per-path existence checks."""
    paths: list[Path] = []
    text = manifest_path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def _load_manifest_slice(manifest_path: Path, start_index: int, n: int) -> list[Path]:
    """Read only ``n`` lines starting at ``start_index`` (resolves paths for WSL)."""
    if n <= 0:
        return []
    paths: list[Path] = []
    with manifest_path.open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            if len(paths) >= n:
                break
            line = line.strip()
            if line:
                paths.append(_normalize_manifest_path(line))
    return paths


def _sorted_cache_valid(*, rebuild: bool) -> tuple[bool, dict | None]:
    if rebuild:
        return False, None
    manifest_path, meta_path = _sorted_manifest_paths()
    meta = _read_path_meta(meta_path)
    source_mtime = _path_cache_mtime(IMAGE_INDEX_PICKLE)
    if (
        manifest_path.is_file()
        and meta
        and meta.get("source_mtime") == source_mtime
        and meta.get("count", 0) > 0
    ):
        return True, meta
    return False, None


def collect_jpg_paths_sorted(data_dir: Path, *, rebuild: bool = False) -> list[Path]:
    """Stable sorted JPG list; cached to disk (no per-path resolve on WSL)."""
    manifest_path, meta_path = _sorted_manifest_paths()
    ok, meta = _sorted_cache_valid(rebuild=rebuild)
    if ok and meta:
        print(f"Loaded sorted path index from cache ({meta['count']} images)")
        return _load_path_manifest_fast(manifest_path)

    print("Building sorted path index (one-time; caching for next run)...")
    paths = collect_jpg_paths(data_dir)
    paths.sort(key=_path_sort_key)
    _save_path_manifest(paths, manifest_path)
    source_mtime = _path_cache_mtime(IMAGE_INDEX_PICKLE)
    _write_path_meta(
        meta_path,
        {
            "source": "image_index.pkl" if IMAGE_INDEX_PICKLE.is_file() else "filesystem_scan",
            "source_mtime": source_mtime,
            "count": len(paths),
        },
    )
    print(f"Cached sorted index -> {manifest_path}")
    return paths


def _shuffled_cache_valid(seed: int, *, rebuild: bool) -> tuple[bool, dict | None]:
    if rebuild:
        return False, None
    sorted_manifest, _ = _sorted_manifest_paths()
    shuffled_manifest, shuffled_meta_path = _shuffled_manifest_paths(seed)
    sh_meta = _read_path_meta(shuffled_meta_path)
    if (
        shuffled_manifest.is_file()
        and sh_meta
        and sh_meta.get("sorted_mtime") == _path_cache_mtime(sorted_manifest)
        and sh_meta.get("seed") == seed
        and sh_meta.get("count", 0) > 0
    ):
        return True, sh_meta
    return False, None


def _build_shuffled_cache(data_dir: Path, seed: int) -> dict:
    sorted_manifest, _ = _sorted_manifest_paths()
    shuffled_manifest, shuffled_meta_path = _shuffled_manifest_paths(seed)
    paths = collect_jpg_paths_sorted(data_dir)
    rng = random.Random(seed)
    rng.shuffle(paths)
    _save_path_manifest(paths, shuffled_manifest)
    meta = {
        "seed": seed,
        "sorted_mtime": _path_cache_mtime(sorted_manifest),
        "count": len(paths),
    }
    _write_path_meta(shuffled_meta_path, meta)
    print(f"Cached shuffled index (seed={seed}) -> {shuffled_manifest}")
    return meta


def shuffled_jpg_paths(data_dir: Path, seed: int = 42, *, rebuild: bool = False) -> list[Path]:
    """Seed-shuffled JPG list; cached per seed (loads full list — prefer sample_jpg_paths)."""
    ok, sh_meta = _shuffled_cache_valid(seed, rebuild=rebuild)
    shuffled_manifest, _ = _shuffled_manifest_paths(seed)
    if not ok:
        sh_meta = _build_shuffled_cache(data_dir, seed)
    print(f"Loaded shuffled path index from cache (seed={seed}, {sh_meta['count']} images)")
    return _load_path_manifest_fast(shuffled_manifest)


def sample_jpg_paths(
    data_dir: Path,
    n: int,
    seed: int = 42,
    start_index: int = 0,
    *,
    rebuild: bool = False,
) -> list[Path]:
    """
    Take ``n`` paths from the seed-shuffled list.

    ``start_index`` is 0-based (0 = 1st image, 150 = 151st image).
    Round 1: start_index=0, n=150. Round 2: start_index=150, n=150.
    """
    if n <= 0:
        return []
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")

    ok, sh_meta = _shuffled_cache_valid(seed, rebuild=rebuild)
    shuffled_manifest, _ = _shuffled_manifest_paths(seed)
    if not ok:
        sh_meta = _build_shuffled_cache(data_dir, seed)

    total = int(sh_meta["count"])
    end = start_index + n
    if end > total:
        raise ValueError(
            f"Only {total} images under {data_dir}, "
            f"requested indices {start_index}..{end - 1} (n={n})"
        )

    print(
        f"Loaded shuffled slice from cache (seed={seed}, "
        f"indices {start_index + 1}..{end}, n={n})"
    )
    return _load_manifest_slice(shuffled_manifest, start_index, n)


def save_image_manifest(paths: list[Path], manifest_path: Path) -> None:
    _save_path_manifest(paths, manifest_path)


def _normalize_manifest_path(line: str) -> Path:
    line = line.strip().lstrip("\ufeff")
    p = Path(line)
    if p.exists():
        return p
    # Windows path in manifest while running under WSL
    if len(line) >= 2 and line[1] == ":":
        drive = line[0].lower()
        rest = line[2:].replace("\\", "/").lstrip("/")
        wsl = Path(f"/mnt/{drive}/{rest}")
        if wsl.exists():
            return wsl
    return p


def load_image_manifest(manifest_path: Path) -> list[Path]:
    text = manifest_path.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [_normalize_manifest_path(ln) for ln in lines]


def resolve_image_sample(
    data_dir: Path,
    n: int,
    seed: int,
    manifest_path: Path | None = None,
    start_index: int = 0,
) -> list[Path]:
    """
    Load the exact image list for a comparison run.

    Priority:
      1. Explicit --manifest file
      2. Default cloud-demo manifest if it exists
      3. Deterministic shuffle + slice (sorted paths, seed, start_index)
    """
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        paths = load_image_manifest(manifest_path)
        return paths[:n] if n > 0 else paths

    if CLOUD_DEMO_MANIFEST.is_file():
        paths = load_image_manifest(CLOUD_DEMO_MANIFEST)
        if n > 0:
            return paths[:n]
        return paths

    return sample_jpg_paths(data_dir, n, seed, start_index=start_index)


def collect_clear_day_paths(
    data_dir: Path,
    stride: int = 30,
    max_per_day: int | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for month, day in CLEAR_DAYS:
        day_dir = data_dir / f"2017_{month:02d}_images_raw" / f"{month:02d}" / f"{day:02d}"
        if not day_dir.is_dir():
            continue
        jpgs = sorted(day_dir.glob("*.jpg"))[::stride]
        if max_per_day is not None:
            jpgs = jpgs[:max_per_day]
        paths.extend(jpgs)
    return paths


def extract_patch(image: np.ndarray, col: int, row: int, size: int = PATCH_SIZE) -> np.ndarray | None:
    """Extract size×size RGB patch centered on (col, row). Returns None if out of bounds."""
    half = size // 2
    r0, r1 = row - half, row - half + size
    c0, c1 = col - half, col - half + size
    if r0 < 0 or c0 < 0 or r1 > image.shape[0] or c1 > image.shape[1]:
        return None
    return image[r0:r1, c0:c1].copy()


SEGMENTATION_PROGRESS = MANUAL_SEG_ROOT / "output" / "segmentation_progress.json"
# Round 1 (150) + round 2 (150) + round 3 (150) → next slice starts at index 450.
DEFAULT_NEXT_START_INDEX = 450


def load_segmentation_progress(seed: int = 42) -> dict:
    if SEGMENTATION_PROGRESS.is_file():
        try:
            data = json.loads(SEGMENTATION_PROGRESS.read_text(encoding="utf-8"))
            if data.get("seed") == seed:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "seed": seed,
        "next_start_index": DEFAULT_NEXT_START_INDEX,
        "batches": [],
    }


def record_segmentation_batch(
    seed: int,
    start_index: int,
    n: int,
    *,
    dataset_dir: str = "",
) -> int:
    """Advance progress after a predict/review batch. Returns new next_start_index."""
    data = load_segmentation_progress(seed)
    next_index = start_index + n
    data["next_start_index"] = next_index
    data.setdefault("batches", []).append(
        {
            "start_index": start_index,
            "n": n,
            "end_index": next_index - 1,
            "dataset_dir": dataset_dir,
        }
    )
    SEGMENTATION_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_PROGRESS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return next_index
