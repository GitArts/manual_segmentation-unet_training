"""Project root paths. Override the sky-image dataset with SKIPPD_DATA_DIR."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB_DIR = ROOT / "lib"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

# Sky-image JPEG dataset (not shipped with this repo).
DATA_DIR = Path(os.environ.get("SKIPPD_DATA_DIR", ROOT / "data")).resolve()

# Optional fixed image list for demos / comparisons.
DEFAULT_MANIFEST = DATA_DIR / "image_manifest.txt"

# Optional path index for large datasets (built locally, see skippd_io).
IMAGE_INDEX_PICKLE = CACHE_DIR / "image_index.pkl"

# Optional Nie et al. clear-sky library (only for demo_segment.py --compare).
CLEAR_SKY_LIBRARY_DIR = DATA_DIR / "clear_sky_library"
