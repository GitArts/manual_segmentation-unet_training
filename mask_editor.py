"""Mask brush editor — runs inside the review figure (one window, WSL-safe)."""

from __future__ import annotations

import numpy as np

from disk_mask import train_valid_mask
from segment_nrbr import LABEL_NAMES, LABEL_SKY, LABEL_SUN, SUN_DISK_RADIUS, overlay_labels


def fill_all_sky(
    labels: np.ndarray,
    valid: np.ndarray,
    sun_row: int,
    sun_col: int,
) -> None:
    """Set valid disk to clear sky; keep sun patch at predicted position."""
    labels[valid] = LABEL_SKY
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])[None, :]
    sun_disk = (rows - sun_row) ** 2 + (cols - sun_col) ** 2 <= SUN_DISK_RADIUS**2
    labels[valid & sun_disk] = LABEL_SUN


def _paint_disk(labels: np.ndarray, row: int, col: int, cls: int, radius: int, valid: np.ndarray) -> None:
    r0 = max(0, row - radius)
    r1 = min(labels.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(labels.shape[1], col + radius + 1)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if (rr - row) ** 2 + (cc - col) ** 2 <= radius * radius and valid[rr, cc]:
                labels[rr, cc] = cls


def release_mouse_grab(canvas) -> None:
    grabber = getattr(canvas, "mouse_grabber", None)
    if grabber is not None:
        try:
            canvas.release_mouse(grabber)
        except Exception:
            pass


def focus_figure_window(fig) -> None:
    """Give the figure window + canvas keyboard focus (needed on WSL/Tk)."""
    try:
        win = fig.canvas.manager.window
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()
        win.after(150, lambda: win.attributes("-topmost", False))
    except Exception:
        pass
    try:
        widget = fig.canvas.get_tk_widget()
        widget.focus_set()
        widget.focus_force()
    except Exception:
        pass
    try:
        fig.canvas.setFocus()
    except Exception:
        pass
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def edit_mask_in_figure(
    fig,
    hide_axes: list,
    img: np.ndarray,
    labels: np.ndarray,
    sun_row: int,
    sun_col: int,
    *,
    title: str = "Edit mask",
) -> np.ndarray | None:
    """
    Paint UI inside an existing figure. Hides ``hide_axes`` until save/cancel.

    Keys: 0-4 / v s l u o = class   [ ] = brush   A = all sky   C = save   Esc = cancel
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    release_mouse_grab(fig.canvas)

    labels = labels.copy()
    valid = train_valid_mask(labels.shape)
    paint_class = 2
    brush_radius = 2
    painting = {"active": False}
    done = {"finished": False, "labels": None}
    class_keys = {"v": 0, "s": 1, "l": 2, "u": 3, "o": 4}
    class_legend = "  |  ".join(f"{k}={LABEL_NAMES[k]}" for k in range(5))

    for ax in hide_axes:
        ax.set_visible(False)

    ax_shortcuts = fig.add_axes([0.04, 0.86, 0.92, 0.11])
    ax_shortcuts.axis("off")
    ax = fig.add_axes([0.06, 0.24, 0.88, 0.58])
    ax_help = fig.add_axes([0.06, 0.11, 0.88, 0.10])
    ax_help.axis("off")
    ax_btn = fig.add_axes([0.06, 0.02, 0.28, 0.07])
    edit_axes = [ax_shortcuts, ax, ax_help, ax_btn]
    widget_axes = {ax_btn}

    shortcut_text = (
        "PAINT: left-drag   |   CLASS: 0-4 or v/s/l/u/o   |   BRUSH: [ ]   |   "
        "A or button = all sky   |   SAVE: C   |   CANCEL: Esc"
    )
    ax_shortcuts.text(
        0.5,
        0.5,
        shortcut_text,
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#888888"),
    )

    try:
        fig.canvas.manager.set_window_title(f"{title}  |  C=save  Esc=cancel")
    except Exception:
        pass

    print("\n--- Mask editor (same window) ---")
    print("  A or [All sky] = fill disk with clear sky (removes clouds)")
    print("  C = save and next image")
    print("  Esc = cancel")
    print("---------------------------------\n")

    btn_all_sky = Button(ax_btn, "All sky (clear clouds)")

    im_overlay = ax.imshow(overlay_labels(img, labels))
    ax.plot(sun_col, sun_row, "r+", ms=8, mew=1.5)
    ax.axis("off")

    def refresh() -> None:
        im_overlay.set_data(overlay_labels(img, labels))
        cls_name = LABEL_NAMES[paint_class]
        ax_help.clear()
        ax_help.axis("off")
        ax_help.text(
            0.5,
            0.78,
            f"ACTIVE: class {paint_class} ({cls_name})  —  brush {brush_radius}px",
            ha="center",
            fontsize=11,
            weight="bold",
            color="#1a5276",
        )
        ax_help.text(0.5, 0.42, class_legend, ha="center", fontsize=9)
        ax_help.text(
            0.5,
            0.1,
            "A or [All sky] = clear whole disk to sky  •  C = save  •  Esc = cancel",
            ha="center",
            fontsize=9,
            weight="bold",
            color="#117a3d",
        )
        fig.canvas.draw()
        focus_figure_window(fig)

    def apply_all_sky(_event=None) -> None:
        release_mouse_grab(fig.canvas)
        fill_all_sky(labels, valid, sun_row, sun_col)
        refresh()

    btn_all_sky.on_clicked(apply_all_sky)

    def cleanup() -> None:
        for cid in conn_ids:
            fig.canvas.mpl_disconnect(cid)
        for eax in edit_axes:
            eax.remove()
        for hax in hide_axes:
            hax.set_visible(True)
        release_mouse_grab(fig.canvas)

    def on_save() -> None:
        done["labels"] = labels.copy()
        done["finished"] = True
        cleanup()

    def on_cancel() -> None:
        done["labels"] = None
        done["finished"] = True
        cleanup()

    def on_mouse_press(event) -> None:
        if event.inaxes in widget_axes or event.inaxes is not ax or event.button != 1:
            return
        painting["active"] = True
        if event.xdata is not None and event.ydata is not None:
            _paint_disk(
                labels,
                int(round(event.ydata)),
                int(round(event.xdata)),
                paint_class,
                brush_radius,
                valid,
            )
            refresh()

    def on_mouse_release(_event) -> None:
        painting["active"] = False
        release_mouse_grab(fig.canvas)

    def on_mouse_move(event) -> None:
        if not painting["active"] or event.inaxes is not ax:
            return
        if event.xdata is not None and event.ydata is not None:
            _paint_disk(
                labels,
                int(round(event.ydata)),
                int(round(event.xdata)),
                paint_class,
                brush_radius,
                valid,
            )
            refresh()

    def on_key(event) -> None:
        nonlocal paint_class, brush_radius
        key = (event.key or "").lower()
        if key == "c":
            on_save()
            return
        if key == "escape":
            on_cancel()
            return
        if key == "a":
            apply_all_sky()
            return
        if key in class_keys:
            paint_class = class_keys[key]
            refresh()
            return
        if key in "01234" and len(key) == 1:
            paint_class = int(key)
            refresh()
            return
        if key == "[":
            brush_radius = int(np.clip(brush_radius - 1, 1, 8))
            refresh()
            return
        if key == "]":
            brush_radius = int(np.clip(brush_radius + 1, 1, 8))
            refresh()
            return

    conn_ids = [
        fig.canvas.mpl_connect("button_press_event", on_mouse_press),
        fig.canvas.mpl_connect("button_release_event", on_mouse_release),
        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]

    refresh()
    focus_figure_window(fig)
    try:
        fig.canvas.manager.window.after(50, lambda: focus_figure_window(fig))
        fig.canvas.manager.window.after(150, lambda: focus_figure_window(fig))
    except Exception:
        pass

    while not done["finished"]:
        plt.pause(0.05)

    if done["labels"] is None:
        return None
    return np.asarray(done["labels"], dtype=np.uint8)


# Backward-compatible alias
def edit_mask_interactive(
    img: np.ndarray,
    labels: np.ndarray,
    sun_row: int,
    sun_col: int,
    *,
    title: str = "Edit mask",
    parent_fig=None,
    hide_axes=None,
) -> np.ndarray | None:
    if parent_fig is not None and hide_axes is not None:
        return edit_mask_in_figure(
            parent_fig, hide_axes, img, labels, sun_row, sun_col, title=title
        )
    raise RuntimeError("edit_mask_interactive requires parent_fig and hide_axes")
