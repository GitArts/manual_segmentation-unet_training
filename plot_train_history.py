"""Plot training curves from train_history.json without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_unet import save_training_plots

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = ROOT / "output" / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot U-Net training curves from history JSON")
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    hist_path = checkpoint_dir / "train_history.json"
    if not hist_path.is_file():
        raise FileNotFoundError(f"No history file: {hist_path}")

    history = json.loads(hist_path.read_text(encoding="utf-8"))
    has_val = any("val_loss" in row for row in history)
    curves_path = save_training_plots(checkpoint_dir, history, has_val=has_val)
    if curves_path is None:
        raise SystemExit("Plotting failed (see message above).")
    print(f"Saved: {curves_path}")


if __name__ == "__main__":
    main()
