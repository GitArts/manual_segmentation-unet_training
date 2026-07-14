"""
Review sky images and build two 5×5 sun RGB palettes: clouded vs clean.

  python extract_sun_palette.py

Opens http://127.0.0.1:8765 — two buttons, both extract 5×5 at predicted sun:
  Clouded sun  — sun blocked / clouds nearby
  Clean sun    — clear sun disk

Outputs -> output/sun_palette/
  sun_picks.json, sun_palette_clouded.npy, sun_palette_clean.npy, stats, previews
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib.sun_position_identification import sun_position
from skippd_io import (
    CLEAR_DAYS,
    DATA_DIR,
    PATCH_SIZE,
    collect_clear_day_paths,
    collect_jpg_paths,
    datetime_from_filename,
    extract_patch,
    load_rgb_64,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "sun_palette"
HTML_PATH = Path(__file__).resolve().parent / "sun_palette_picker.html"
DEFAULT_PORT = 8765


def load_sample_paths(
    data_dir: Path,
    n: int,
    seed: int,
    source: str,
    stride: int,
) -> list[Path]:
    if source == "csl":
        pool = collect_clear_day_paths(data_dir, stride=stride)
        if not pool:
            raise SystemExit(f"No images on CSL clear days under {data_dir}")
    else:
        pool = collect_jpg_paths(data_dir)
    random.seed(seed)
    if len(pool) < n:
        print(f"Warning: only {len(pool)} candidates, using all.")
        return pool
    return random.sample(pool, n)


def rgb_to_b64(img: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_stats(patches: np.ndarray) -> dict:
    flat = patches.reshape(-1, 3).astype(np.float64)
    return {
        "num_patches": int(patches.shape[0]),
        "num_pixels": int(flat.shape[0]),
        "patch_size": PATCH_SIZE,
        "mean_rgb": flat.mean(axis=0).round(2).tolist(),
        "median_rgb": np.median(flat, axis=0).round(2).tolist(),
        "std_rgb": flat.std(axis=0).round(2).tolist(),
        "min_rgb": flat.min(axis=0).astype(int).tolist(),
        "max_rgb": flat.max(axis=0).astype(int).tolist(),
    }


def save_palettes(
    picks: list[dict],
    clouded: np.ndarray | None,
    clean: np.ndarray | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "sun_picks.json").open("w", encoding="utf-8") as f:
        json.dump(picks, f, indent=2)

    stats: dict = {"patch_size": PATCH_SIZE}
    if clouded is not None and clouded.size:
        np.save(out_dir / "sun_palette_clouded.npy", clouded)
        stats["clouded"] = compute_stats(clouded)
        _save_preview(clouded, out_dir / "sun_palette_preview_clouded.png", "Clouded sun")
    if clean is not None and clean.size:
        np.save(out_dir / "sun_palette_clean.npy", clean)
        stats["clean"] = compute_stats(clean)
        _save_preview(clean, out_dir / "sun_palette_preview_clean.png", "Clean sun")

    with (out_dir / "sun_palette_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    nc = 0 if clouded is None else clouded.shape[0]
    nk = 0 if clean is None else clean.shape[0]
    print(f"\nSaved -> {out_dir}")
    print(f"  clouded: {nc} patches")
    print(f"  clean:   {nk} patches")
    if "clouded" in stats:
        print(f"  clouded mean RGB = {stats['clouded']['mean_rgb']}")
    if "clean" in stats:
        print(f"  clean   mean RGB = {stats['clean']['mean_rgb']}")


def _save_preview(patches: np.ndarray, path: Path, title: str, zoom: int = 12) -> None:
    n = patches.shape[0]
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        up = np.repeat(np.repeat(patches[i], zoom, axis=0), zoom, axis=1)
        ax.imshow(up, interpolation="nearest")
        ax.set_title(f"#{i + 1}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{title} — {PATCH_SIZE}×{PATCH_SIZE} RGB", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


class PickerState:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.idx = 0
        self.picks: list[dict] = []
        self.clouded_patches: list[np.ndarray] = []
        self.clean_patches: list[np.ndarray] = []
        self.done = threading.Event()
        self._lock = threading.Lock()

    def _current(self) -> tuple[Path, np.ndarray, datetime, int, int] | None:
        if self.idx >= len(self.paths):
            return None
        path = self.paths[self.idx]
        t = datetime_from_filename(path.name)
        img = load_rgb_64(path)
        sx, sy, _ = sun_position(t)
        return path, img, t, int(sy), int(sx)  # col, row

    def get_json(self) -> dict:
        with self._lock:
            cur = self._current()
            if cur is None:
                return {
                    "done": True,
                    "count_clouded": len(self.clouded_patches),
                    "count_clean": len(self.clean_patches),
                    "total": len(self.paths),
                    "index": self.idx,
                }
            path, img, t, col, row = cur
            return {
                "done": False,
                "index": self.idx,
                "total": len(self.paths),
                "count_clouded": len(self.clouded_patches),
                "count_clean": len(self.clean_patches),
                "filename": path.name,
                "time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "sun_col": col,
                "sun_row": row,
                "patch_size": PATCH_SIZE,
                "image_b64": rgb_to_b64(img),
            }

    def _label(self, kind: str) -> None:
        cur = self._current()
        if cur is None:
            self.done.set()
            return
        path, img, t, col, row = cur
        patch = extract_patch(img, col, row)
        if patch is None:
            print(f"  [{self.idx + 1}/{len(self.paths)}] sun out of frame, skipped")
        else:
            self.picks.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "time": t.isoformat(),
                    "label": kind,
                    "col": col,
                    "row": row,
                    "predicted_sun_row": row,
                    "predicted_sun_col": col,
                }
            )
            if kind == "clouded":
                self.clouded_patches.append(patch)
            else:
                self.clean_patches.append(patch)
            print(
                f"  [{self.idx + 1}/{len(self.paths)}] {kind}  "
                f"mean RGB {patch.mean(axis=(0, 1)).round(1).tolist()}"
            )
        self.idx += 1
        if self.idx >= len(self.paths):
            self.done.set()

    def clouded(self) -> None:
        with self._lock:
            self._label("clouded")

    def clean(self) -> None:
        with self._lock:
            self._label("clean")


def make_handler(state: PickerState):
    html = HTML_PATH.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(state.get_json())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path == "/api/clouded":
                state.clouded()
                self._json(state.get_json())
            elif self.path == "/api/clean":
                state.clean()
                self._json(state.get_json())
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def web_pick(paths: list[Path], port: int = DEFAULT_PORT) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
    state = PickerState(paths)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    print(f"\nOpen in browser: {url}")
    print("  Clouded sun — extract 5×5 patch, next image")
    print("  Clean sun   — extract 5×5 patch, next image\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    state.done.wait()
    server.shutdown()

    if not state.clouded_patches and not state.clean_patches:
        raise SystemExit("No patches collected.")

    clouded = np.stack(state.clouded_patches, axis=0) if state.clouded_patches else None
    clean = np.stack(state.clean_patches, axis=0) if state.clean_patches else None
    return state.picks, clouded, clean


def load_from_picks(picks_path: Path) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
    with picks_path.open(encoding="utf-8") as f:
        picks = json.load(f)
    clouded_list: list[np.ndarray] = []
    clean_list: list[np.ndarray] = []
    kept: list[dict] = []
    for pick in picks:
        img = load_rgb_64(Path(pick["path"]))
        patch = extract_patch(img, pick["col"], pick["row"])
        if patch is None:
            continue
        kept.append(pick)
        label = pick.get("label", "clean")
        if label == "clouded":
            clouded_list.append(patch)
        else:
            clean_list.append(patch)
    if not kept:
        raise SystemExit("No valid patches from picks file.")
    clouded = np.stack(clouded_list, axis=0) if clouded_list else None
    clean = np.stack(clean_list, axis=0) if clean_list else None
    return kept, clouded, clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 5×5 sun RGB palette")
    parser.add_argument("-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--source",
        choices=("csl", "random"),
        default="random",
        help="random = mixed days (better for clouded+clean); csl = clear-sky days only",
    )
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--from-picks", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.from_picks:
        picks, clouded, clean = load_from_picks(Path(args.from_picks))
        save_palettes(picks, clouded, clean, out_dir)
        return

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    paths = load_sample_paths(data_dir, args.n, args.seed, args.source, args.stride)
    print(f"Loaded {len(paths)} images from {args.source}")

    picks, clouded, clean = web_pick(paths, port=args.port)
    save_palettes(picks, clouded, clean, out_dir)


if __name__ == "__main__":
    main()
