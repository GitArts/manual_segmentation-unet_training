"""Interactive Correct / Incorrect review UI for segmentation datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from mask_editor import edit_mask_in_figure, focus_figure_window, release_mouse_grab
from segment_nrbr import colorize_labels, overlay_labels


@dataclass
class ReviewItem:
    path: Path
    timestamp: datetime
    img: np.ndarray
    labels: np.ndarray
    sun_row: int
    sun_col: int
    status_text: str = ""


def save_dataset_pair(item: ReviewItem, images_dir: Path, masks_dir: Path) -> tuple[Path, Path]:
    stem = item.path.stem
    image_out = images_dir / f"{stem}.png"
    mask_out = masks_dir / f"{stem}.png"
    Image.fromarray(item.img).save(image_out)
    Image.fromarray(item.labels.astype(np.uint8), mode="L").save(mask_out)
    return image_out, mask_out


def run_review_ui(
    items: list[ReviewItem],
    dataset_dir: Path,
    *,
    title: str = "Segmentation review — Correct / Fix / Skip",
    append: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    approved: list[Path] = []
    rejected: list[Path] = []

    fig = plt.figure(figsize=(13, 5.5))
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

    ax_orig = fig.add_axes([0.03, 0.18, 0.28, 0.72])
    ax_labels = fig.add_axes([0.34, 0.18, 0.28, 0.72])
    ax_overlay = fig.add_axes([0.65, 0.18, 0.32, 0.72])
    ax_status = fig.add_axes([0.03, 0.02, 0.94, 0.1])
    ax_status.axis("off")

    ax_correct = fig.add_axes([0.08, 0.92, 0.16, 0.06])
    ax_fix = fig.add_axes([0.36, 0.92, 0.16, 0.06])
    ax_incorrect = fig.add_axes([0.64, 0.92, 0.16, 0.06])
    btn_correct = Button(ax_correct, "Correct (C)")
    btn_fix = Button(ax_fix, "Fix mask (D)")
    btn_incorrect = Button(ax_incorrect, "Skip (I)")

    state = {"idx": 0, "busy": False, "suppress_keys_until": 0.0}

    def update_display() -> None:
        item = items[state["idx"]]
        ax_orig.clear()
        ax_labels.clear()
        ax_overlay.clear()

        ax_orig.imshow(item.img)
        ax_orig.plot(item.sun_col, item.sun_row, "r+", ms=8, mew=1.5)
        ax_orig.set_title(item.timestamp.strftime("%Y-%m-%d %H:%M"), fontsize=9)
        ax_orig.axis("off")

        ax_labels.imshow(colorize_labels(item.labels))
        ax_labels.set_title("Labels", fontsize=9)
        ax_labels.axis("off")

        ax_overlay.imshow(overlay_labels(item.img, item.labels))
        ax_overlay.set_title("Overlay", fontsize=9)
        ax_overlay.axis("off")

        ax_status.clear()
        ax_status.axis("off")
        ax_status.text(
            0.5,
            0.65,
            f"{state['idx'] + 1} / {len(items)}  |  approved: {len(approved)}  |  rejected: {len(rejected)}",
            ha="center",
            fontsize=11,
            weight="bold",
        )
        status = item.status_text or item.path.name
        ax_status.text(0.5, 0.15, f"{item.path.name}\n{status}", ha="center", fontsize=8)
        fig.canvas.draw_idle()
        focus_figure_window(fig)

    def write_manifests() -> None:
        manifest_approved = dataset_dir / "approved_manifest.txt"
        manifest_rejected = dataset_dir / "rejected_manifest.txt"
        mode = "a" if append else "w"
        with manifest_approved.open(mode, encoding="utf-8") as f:
            for p in approved:
                f.write(f"{p}\n")
        with manifest_rejected.open(mode, encoding="utf-8") as f:
            for p in rejected:
                f.write(f"{p}\n")

    def finish() -> None:
        write_manifests()
        print("\nReview complete.")
        print(f"  Approved this session: {len(approved)} -> {images_dir} and {masks_dir}")
        print(f"  Rejected this session: {len(rejected)}")
        plt.close(fig)

    def advance() -> None:
        if state["idx"] + 1 >= len(items):
            finish()
            return
        state["idx"] += 1
        update_display()

    def on_correct(_event=None) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        item = items[state["idx"]]
        image_out, mask_out = save_dataset_pair(item, images_dir, masks_dir)
        approved.append(item.path)
        print(f"  [OK] {item.path.name} -> {image_out.name}, {mask_out.name}")
        state["busy"] = False
        advance()

    def on_fix(_event=None) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        item = items[state["idx"]]
        release_mouse_grab(fig.canvas)
        focus_figure_window(fig)

        hide_axes = [ax_orig, ax_labels, ax_overlay, ax_status, ax_correct, ax_fix, ax_incorrect]
        edited = edit_mask_in_figure(
            fig,
            hide_axes,
            item.img,
            item.labels,
            item.sun_row,
            item.sun_col,
            title=f"Fix mask — {item.path.name}",
        )

        state["busy"] = False
        state["suppress_keys_until"] = time.time() + 0.5

        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        if edited is None:
            print(f"  [edit cancelled] {item.path.name}")
            update_display()
            focus_figure_window(fig)
            return

        item.labels = edited
        image_out, mask_out = save_dataset_pair(item, images_dir, masks_dir)
        approved.append(item.path)
        print(f"  [fixed] {item.path.name} -> {image_out.name}, {mask_out.name}")
        advance()

    def on_incorrect(_event=None) -> None:
        if state["busy"]:
            return
        item = items[state["idx"]]
        rejected.append(item.path)
        print(f"  [skip] {item.path.name}")
        advance()

    def on_key(event) -> None:
        if state["busy"]:
            return
        if time.time() < state["suppress_keys_until"]:
            return
        key = (event.key or "").lower()
        if key in ("c", "y"):
            on_correct()
        elif key == "d":
            on_fix()
        elif key in ("i", "n", "backspace", "delete"):
            on_incorrect()

    def on_fix_click(_event=None) -> None:
        focus_figure_window(fig)
        on_fix(_event)

    btn_correct.on_clicked(on_correct)
    btn_fix.on_clicked(on_fix_click)
    btn_incorrect.on_clicked(on_incorrect)
    fig.canvas.mpl_connect("key_press_event", on_key)

    focus_figure_window(fig)

    print(
        "\nReview UI open. Click Correct / Fix mask / Skip or use keys:\n"
        "  Correct (save as-is): C or Y  (click Correct button)\n"
        "  Fix mask: D — same window switches to editor\n"
        "    0-4 / v/s/l/u/o class  |  [ ] brush  |  A = all sky  |  C = save  |  Esc cancel\n"
        "  Skip (do not save): I or N or Backspace\n"
    )
    update_display()
    plt.show()


def setup_matplotlib_backend(interactive: bool) -> None:
    import matplotlib

    if interactive:
        for backend in ("TkAgg", "Qt5Agg", "WXAgg"):
            try:
                matplotlib.use(backend)
                break
            except ImportError:
                continue
    else:
        matplotlib.use("Agg")
