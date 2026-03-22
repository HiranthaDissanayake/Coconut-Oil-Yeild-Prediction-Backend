import os, json
import numpy as np
import cv2
from glob import glob

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

DATASET_PATH = "dataset/train"   # <-- your dataset folder
OUTPUT_PATH = "models/copra_features.json"

os.makedirs("models", exist_ok=True)

print("\n" + "═"*70)
print("  🔥 ANALYZING COPRA DATASET")
print("  Extracting feature statistics...")
print("═"*70)


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (SAME AS API)
# ══════════════════════════════════════════════════════════════

def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        'h_mean': np.mean(hsv[:,:,0]) / 180.0,
        's_mean': np.mean(hsv[:,:,1]) / 255.0,
        'v_mean': np.mean(hsv[:,:,2]) / 255.0,

        'brown_ratio': np.sum(
            cv2.inRange(hsv, np.array([10,20,50]), np.array([30,200,255])) > 0
        ) / img.size * 3,

        'white_ratio': np.sum(
            cv2.inRange(hsv, np.array([0,0,180]), np.array([180,50,255])) > 0
        ) / img.size * 3,

        'edge_density': np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size,

        'texture_var': np.var(gray) / 10000.0,

        'brightness_mean': np.mean(gray) / 255.0,
        'brightness_std': np.std(gray) / 255.0,
    }


# ══════════════════════════════════════════════════════════════
# LOAD IMAGES
# ══════════════════════════════════════════════════════════════

image_paths = glob(os.path.join(DATASET_PATH, "*", "*.*"))

if not image_paths:
    print("❌ No images found!")
    exit()

print(f"\n  Found {len(image_paths)} images")

features_list = []

# ══════════════════════════════════════════════════════════════
# PROCESS IMAGES
# ══════════════════════════════════════════════════════════════

for i, path in enumerate(image_paths):
    try:
        img = cv2.imread(path)

        if img is None:
            continue

        features = extract_features(img)
        features_list.append(features)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_paths)}")

    except Exception as e:
        print(f"  Error: {path} → {e}")

print(f"\n  ✅ Processed {len(features_list)} valid images")


# ══════════════════════════════════════════════════════════════
# CALCULATE STATISTICS
# ══════════════════════════════════════════════════════════════

all_keys = features_list[0].keys()

copra_profile = {}

print("\n  Calculating feature statistics...")

for key in all_keys:
    values = np.array([f[key] for f in features_list])

    copra_profile[key] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

    print(f"  {key:<18} mean={np.mean(values):.4f}  std={np.std(values):.4f}")


# ══════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════

copra_profile["_metadata"] = {
    "total_images_analyzed": len(features_list),
    "image_size": [300, 300],
    "description": "Statistical feature profile of copra dataset for validation"
}


# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════

with open(OUTPUT_PATH, "w") as f:
    json.dump(copra_profile, f, indent=2)

print("\n" + "="*70)
print(f"  ✅ Copra profile saved → {OUTPUT_PATH}")
print("  Now restart Flask API")
print("="*70)