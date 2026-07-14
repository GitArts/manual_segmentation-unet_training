# manual_segmentation — U-Net training

Iterative sky-image segmentation for **SKIPP'D** fisheye photos: rule-based auto-labeling, interactive manual curation, and U-Net training on approved masks.

## Pipeline

1. **Auto segmentation** — NRBR (normalized blue–red ratio) with timestamp-based sun position and edge refinement → initial labels (void / sky / cloud; sun classes remapped for training).
2. **Manual curation** — GUI to approve, fix masks inline, or skip; saves paired `images/` + `masks/` PNGs.
3. **U-Net** — train on curated data (CE + Dice loss, 80/10/10 split), evaluate on held-out test set, predict + review on new images.

## Setup

```bash
conda activate LACISE
cd manual_segmentation
```

Python dependencies: `numpy`, `torch`, `Pillow`, `matplotlib`, `tqdm`.

Sun-position and optional Nie et al. cloud-detection code are vendored under `lib/` (no external LACISE repos required).

## Sky-image dataset (not included in this repo)

Place your fisheye sky-camera JPEGs under `data/` (or set `SKIPPD_DATA_DIR` to another folder). Scripts also accept `--data-dir`.

Example if images stay in CloudPred-PV:

```bash
export SKIPPD_DATA_DIR=/path/to/SkyGPT/Codes/CloudPred-PV/Data
```

### What each image must provide

| Requirement | Why |
|-------------|-----|
| **RGB JPEG** | NRBR segmentation uses red and blue channels: `(B−R)/(B+R)`. |
| **Filename = capture timestamp** `YYYYMMDDHHMMSS.jpg` | Parsed as local time to compute **solar azimuth and zenith** (`lib/sun_position_identification.py`), then map the sun onto the 64×64 fisheye frame. |
| **Fisheye geometry (SKIPP'D / Nie et al.)** | Assumes a 64×64 disc: center row 29, col 30, radius 29 px; image north offset **δ = 14.036°** relative to geological north. |
| **Camera site (default: Stanford SKIPP'D)** | Solar angles use lat **37.424107°**, lon **−122.174199°**, PST center lon **−120°** (with DST correction). For another site, edit defaults in `lib/sun_position_identification.py` (`solar_angle()` / `sun_position()`). |

Images are loaded at any resolution and resized to **64×64 RGB** internally.

### Expected folder layout

```
data/
  2017_03_images_raw/03/01/20170301060000.jpg
  2017_05_images_raw/05/20/20170520120000.jpg
  ...
```

Month folders: `{year}_{month:02d}_images_raw/{month:02d}/{day:02d}/*.jpg`.

Alternatives:

- `--manifest path/to/image_manifest.txt` — one absolute path per line (fixed subset).
- `data/image_manifest.txt` — used automatically when present (`write_cloud_manifest.py` can create it).

Optional speed-up for large datasets: put a path index at `cache/image_index.pkl` as a list of `(path_str, pv_value)` tuples (only paths are used).

### Optional: Nie et al. comparison (`demo_segment.py --compare`)

Requires clear-sky library NumPy files under `data/clear_sky_library/`:

- `csl_times.npy`, `csl_images.npy`, `csl_sun_center.npy`

(from the [Cloud-dection-in-sky-images](https://github.com/GitArts/Cloud-dection-in-sky-images) project). Not needed for NRBR segmentation, training, or U-Net inference.

## Main scripts

| Script | Purpose |
|--------|---------|
| `demo_combined.py --review` | NRBR auto-seg + review GUI (round 1) |
| `predict_review.py` | U-Net inference + review (round 2+) |
| `train_unet.py` | Train U-Net on `output/dataset/` |
| `evaluate_unet.py` | Test-split metrics + prediction figures |
| `audit_dataset.py` | Second-pass leave/delete audit |
| `unet_infer.py` | Load checkpoint and predict masks on RGB |

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

## Project layout

```
manual_segmentation/
  lib/                  # vendored sun_position_identification, cloud_detection
  paths.py              # DATA_DIR, cache, output paths (SKIPPD_DATA_DIR env)
  data/                 # your sky JPEGs (gitignored)
  output/               # datasets, checkpoints, demos (gitignored)
  cache/                # path indexes (gitignored)
```
