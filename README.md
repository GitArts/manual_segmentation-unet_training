# manual_segmentation — U-Net training

Iterative sky-image segmentation for **SKIPP'D** fisheye photos: rule-based auto-labeling, interactive manual curation, and U-Net training on approved masks.

## Pipeline

1. **Auto segmentation** — NRBR (normalized blue–red ratio) with timestamp-based sun position and edge refinement → initial labels (void / sky / cloud; sun classes remapped for training).
2. **Manual curation** — GUI to approve, fix masks inline, or skip; saves paired `images/` + `masks/` PNGs.
3. **U-Net** — train on curated data (CE + Dice loss, 80/10/10 split), evaluate on held-out test set, predict + review on new images.

## Main scripts

| Script | Purpose |
|--------|---------|
| `demo_combined.py --review` | NRBR auto-seg + review GUI (round 1) |
| `predict_review.py` | U-Net inference + review (round 2+) |
| `train_unet.py` | Train U-Net on `output/dataset/` |
| `evaluate_unet.py` | Test-split metrics + prediction figures |
| `audit_dataset.py` | Second-pass leave/delete audit |
| `unet_infer.py` | Load checkpoint and predict masks on RGB |

## Setup

```bash
conda activate LACISE
cd manual_segmentation
```

Requires SKIPP'D data paths configured in `skippd_io.py` and sun-position code from the LACISE cloud-detection repo.

## Train

```bash
python train_unet.py --dataset-dir output/dataset_round2 --epochs 120
python evaluate_unet.py --dataset-dir output/dataset_round2
```

## Classes (training)

| ID | Class |
|----|--------|
| 0 | void (outside fisheye) |
| 1 | sky |
| 2 | cloud |

Stored masks may use 5 classes; sun / sun_clouded are remapped to sky/cloud at train time.
