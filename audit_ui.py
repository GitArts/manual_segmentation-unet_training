"""Second-pass audit UI: keep (leave) or delete saved dataset pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from mask_editor import focus_figure_window, release_mouse_grab
from segment_nrbr import colorize_labels, overlay_labels


@dataclass
class AuditItem:
    image_path: Path
    mask_path: Path
    dataset_dir: Path
    timestamp: datetime
    img: np.ndarray
    labels: np.ndarray
    sun_row: int
    sun_col: int


def run_audit_ui(
    items: list[AuditItem],
    *,
    title: str = "Dataset audit — Leave / Delete",
    log_dir: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    log_dir = log_dir or (items[0].dataset_dir if items else Path("."))
    log_dir.mkdir(parents=True, exist_ok=True)
    kept_log = log_dir / "audit_kept_manifest.txt"
    deleted_log = log_dir / "audit_deleted_manifest.txt"

    left_count = 0
    deleted_count = 0
    session_left: list[str] = []
    session_deleted: list[str] = []

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

    ax_leave = fig.add_axes([0.22, 0.92, 0.22, 0.06])
    ax_delete = fig.add_axes([0.56, 0.92, 0.22, 0.06])
    btn_leave = Button(ax_leave, "Leave (L)")
    btn_delete = Button(ax_delete, "Delete (D)")

    state = {"idx": 0, "busy": False}

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
            f"{state['idx'] + 1} / {len(items)}  |  left: {left_count}  |  deleted: {deleted_count}",
            ha="center",
            fontsize=11,
            weight="bold",
        )
        ax_status.text(
            0.5,
            0.15,
            f"{item.image_path.name}\n{item.dataset_dir.name}",
            ha="center",
            fontsize=8,
        )
        fig.canvas.draw_idle()
        focus_figure_window(fig)

    def write_logs() -> None:
        with kept_log.open("a", encoding="utf-8") as f:
            for line in session_left:
                f.write(f"{line}\n")
        with deleted_log.open("a", encoding="utf-8") as f:
            for line in session_deleted:
                f.write(f"{line}\n")

    def finish() -> None:
        write_logs()
        print("\nAudit complete.")
        print(f"  Left (kept): {left_count}")
        print(f"  Deleted: {deleted_count}")
        print(f"  Logs: {kept_log.name}, {deleted_log.name} in {log_dir}")
        plt.close(fig)

    def advance() -> None:
        if state["idx"] + 1 >= len(items):
            finish()
            return
        state["idx"] += 1
        update_display()

    def on_leave(_event=None) -> None:
        nonlocal left_count
        if state["busy"]:
            return
        state["busy"] = True
        item = items[state["idx"]]
        left_count += 1
        session_left.append(f"{item.image_path}\t{item.mask_path}")
        print(f"  [leave] {item.image_path.name}")
        state["busy"] = False
        advance()

    def on_delete(_event=None) -> None:
        nonlocal deleted_count
        if state["busy"]:
            return
        state["busy"] = True
        item = items[state["idx"]]
        release_mouse_grab(fig.canvas)
        for path in (item.image_path, item.mask_path):
            if path.is_file():
                path.unlink()
        deleted_count += 1
        session_deleted.append(f"{item.image_path}\t{item.mask_path}")
        print(f"  [delete] {item.image_path.name}, {item.mask_path.name}")
        state["busy"] = False
        advance()

    def on_key(event) -> None:
        if state["busy"]:
            return
        key = (event.key or "").lower()
        if key in ("l", "y", "enter"):
            on_leave()
        elif key in ("d", "delete", "backspace"):
            on_delete()

    def on_leave_click(_event=None) -> None:
        focus_figure_window(fig)
        on_leave(_event)

    def on_delete_click(_event=None) -> None:
        focus_figure_window(fig)
        on_delete(_event)

    btn_leave.on_clicked(on_leave_click)
    btn_delete.on_clicked(on_delete_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    focus_figure_window(fig)

    print(
        "\nDataset audit open. Second verification pass:\n"
        "  Leave (keep image + mask): L or Y  (click Leave)\n"
        "  Delete (remove image + mask): D or Backspace  (click Delete)\n"
    )
    update_display()
    plt.show()
