"""
╔═══════════════════════════════════════════════════════════╗
║  DATASET PREPARATION  –  7-Day Structure                  ║
║  Organises raw images into day_1 … day_7 folders          ║
╚═══════════════════════════════════════════════════════════╝

Your raw images should be in:
  raw_images/
    day_1/   ← 200 images from Day 1 (fresh)
    day_2/   ← 200 images
    ...
    day_7/   ← 200 images (most dried)

Run: python prepare_dataset.py
Output: dataset/day_1/ … dataset/day_7/
"""

import os, shutil, random
from pathlib import Path

def prepare(
    raw_dir    = 'raw_images',
    out_dir    = 'dataset',
    val_split  = 0.20,
    seed       = 42
):
    random.seed(seed)
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    days      = [f'day_{i}' for i in range(1, 8)]

    print("=" * 50)
    print(" DATASET PREPARATION / Dataset Prepare")
    print("=" * 50)

    total = 0
    for day in days:
        src = os.path.join(raw_dir, day)
        if not os.path.exists(src):
            print(f"  ⚠️   Missing: {src}")
            continue

        imgs = [f for f in os.listdir(src)
                if Path(f).suffix.lower() in valid_ext]
        random.shuffle(imgs)

        n_val = int(len(imgs) * val_split)
        splits = {
            'train': imgs[n_val:],
            'val':   imgs[:n_val],
        }

        for split, files in splits.items():
            dst = os.path.join(out_dir, split, day)
            os.makedirs(dst, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

        total += len(imgs)
        print(f"  ✅  {day}: {len(imgs)} images → "
              f"train={len(splits['train'])}, val={len(splits['val'])}")

    print(f"\n  Total: {total} images prepared in '{out_dir}/'")
    print("  Ready to run: python train_cnn_7day.py")

if __name__ == '__main__':
    prepare()