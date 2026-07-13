"""Write image_manifest.txt for cloud-demo / manual_segmentation comparison."""
from skippd_io import CLOUD_DEMO_MANIFEST, DATA_DIR, sample_jpg_paths, save_image_manifest

if __name__ == "__main__":
    sample = sample_jpg_paths(DATA_DIR, 20, 42)
    save_image_manifest(sample, CLOUD_DEMO_MANIFEST)
    print(f"Saved {len(sample)} paths -> {CLOUD_DEMO_MANIFEST}")
    for i, p in enumerate(sample, 1):
        print(f"  {i:02d} {p.name}")
