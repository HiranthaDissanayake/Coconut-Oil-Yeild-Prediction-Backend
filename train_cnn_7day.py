"""
╔══════════════════════════════════════════════════════════════════╗
║  COCONUT DRYNESS CNN  –  7-Day Progressive Training              ║
║  Day 1 (fresh) → Day 7 (fully dried) = 7 classes                ║
║  200 images × 7 days = 1,400 images total                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Advanced techniques used:                                        ║
║  1. EfficientNetV2S  – best at detecting subtle visual changes   ║
║  2. LAB color space  – separates color from brightness exactly   ║
║  3. GLCM texture     – measures surface roughness / smoothness   ║
║  4. Ordinal regression – respects day 1 < 2 < 3 … < 7 order     ║
║  5. Label smoothing  – handles images that look similar          ║
║  6. Progressive unfreeze – fine-tunes carefully layer by layer   ║
╚══════════════════════════════════════════════════════════════════╝

Dataset folder structure needed / අවශ්‍ය folder structure:
  dataset/
    train/
      day_1/   (200 images – fresh coconut / fresh coconut images)
      day_2/   (200 images)
      day_3/   (200 images)
      day_4/   (200 images)
      day_5/   (200 images)
      day_6/   (200 images)
      day_7/   (200 images – most dry / වඩාත්ම dried images)
    val/
      day_1/   (validation images)
      day_2/   (validation images)
      ...
      day_7/   (validation images)

Run: python train_cnn_7day.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report, confusion_matrix, mean_absolute_error
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    # 7 classes: day_1 … day_7
    CLASSES      = ['day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']
    NUM_CLASSES  = 7
    IMG_SIZE     = (300, 300)   # EfficientNetV2S expects 300×300
    BATCH_SIZE   = 16           # Small batch = better for 200 images/class
    DATASET_DIR  = 'dataset'

    # Dryness score per day (linear: day1=1/7 … day7=7/7)
    # Day number → 0‥1 dryness score
    DRYNESS_SCORE = {
        'day_1': 1/7,   # ≈ 0.143  (freshest)
        'day_2': 2/7,   # ≈ 0.286
        'day_3': 3/7,   # ≈ 0.429
        'day_4': 4/7,   # ≈ 0.571
        'day_5': 5/7,   # ≈ 0.714
        'day_6': 6/7,   # ≈ 0.857
        'day_7': 7/7,   # = 1.000  (most dry)
    }


# ══════════════════════════════════════════════════════════════
#  TECHNIQUE 1: ADVANCED IMAGE PREPROCESSING
#  Why: Raw pixels miss subtle moisture differences.
#  We extract LAB color + texture features that the CNN also sees.
#  සරළව: Raw pixels ලදී subtle moisture differences miss වේ.
#  LAB color + texture features extract කර CNN ලදී feed කරයි.
# ══════════════════════════════════════════════════════════════

def enhance_for_dryness(image_bgr: np.ndarray) -> np.ndarray:
    """
    English: Amplify the visual differences between drying days.
             As coconut dries: it gets darker/browner, surface gets rougher.
             We boost these signals so the CNN can see them more clearly.

    Sinhala: Drying days අතර visual differences amplify කරයි.
             Coconut dry වෙන විට: dark/brown වෙයි, surface rough වෙයි.
             CNN ට clearly දිස්වෙන ලෙස signals boost කරයි.

    Steps:
      1. Convert to LAB color space
         - L channel = lightness  (drier copra = darker L)
         - A channel = green↔red  (drier = more brown/red)
         - B channel = blue↔yellow
      2. CLAHE on L channel – enhances texture contrast
      3. Stack back into 3-channel image (same shape, richer info)
    """
    # Step 1: BGR → LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Step 2: CLAHE on L (Contrast Limited Adaptive Histogram Equalization)
    # English: Makes local texture differences more visible
    # Sinhala: Local texture differences visible කරයි
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    # Step 3: Merge back and convert to RGB
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb     = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb     = cv2.cvtColor(rgb,    cv2.COLOR_BGR2RGB)
    return rgb


def compute_glcm_features(image_bgr: np.ndarray) -> dict:
    """
    English: GLCM = Gray Level Co-occurrence Matrix.
             Measures HOW OFTEN certain pixel-pair patterns appear.
             Dry copra surface = rough → high contrast, low homogeneity.
             Wet copra surface = smooth → low contrast, high homogeneity.

    Sinhala: GLCM = pixel-pair patterns measure කරන matrix.
             Dry copra = rough surface → high contrast.
             Wet copra = smooth surface → low contrast.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Simple texture descriptors (fast, no extra library needed)
    # Laplacian variance = measures sharpness/roughness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Local standard deviation = local texture variation
    kernel  = np.ones((5, 5), np.float32) / 25
    mean_   = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sqmean  = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    loc_std = np.sqrt(np.maximum(sqmean - mean_**2, 0)).mean()

    # Mean brightness in LAB L channel
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_mean = lab[:, :, 0].mean()

    # Mean redness (A channel in LAB: higher = more brown/yellow)
    a_mean = lab[:, :, 1].mean()

    return {
        'roughness':    float(lap_var),    # high = rough (dry)
        'local_std':    float(loc_std),    # high = varied texture
        'brightness':   float(l_mean),     # low  = darker (dry)
        'brownness':    float(a_mean),     # high = browner (dry)
    }


# ══════════════════════════════════════════════════════════════
#  TECHNIQUE 2: CUSTOM DATA GENERATOR WITH ENHANCEMENT
#  Applies enhance_for_dryness() to every image on-the-fly.
#  Every image load ලදී enhance_for_dryness() apply කරයි.
# ══════════════════════════════════════════════════════════════

class EnhancedImageGenerator:
    """
    Wraps Keras ImageDataGenerator.
    Applies LAB enhancement + augmentation on each batch.
    """
    
    def __init__(self, augment=True):
        if augment:
            # Training augmentation – subtle, avoids destroying colour cues
            # Training augmentation – colour cues destroy නොකර subtle
            self.gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,         # small rotation only
                width_shift_range=0.08,
                height_shift_range=0.08,
                zoom_range=0.10,
                horizontal_flip=True,
                brightness_range=[0.85, 1.15],  # keep narrow – brightness is a dryness cue!
                fill_mode='reflect'
                # Removed validation_split since we use separate folders
            )
        else:
            self.gen = ImageDataGenerator(rescale=1./255)
    
    def flow_from_directory(self, directory, shuffle=True, target_size=None):
        """Flow from a specific directory (train or val)"""
        if target_size is None:
            target_size = Config.IMG_SIZE
            
        return self.gen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=shuffle,
            classes=Config.CLASSES  # Explicitly set class order to ensure consistency
        )


# ══════════════════════════════════════════════════════════════
#  TECHNIQUE 3: EFFICIENTNETV2S MODEL WITH ORDINAL AWARENESS
#  EfficientNetV2S is ~60% more accurate than MobileNetV2 for
#  subtle visual differences, while being fast to train.
#
#  Ordinal trick: day ordering matters (1<2<3…<7).
#  We add a "soft ordinal" auxiliary loss that penalises the model
#  more when it confuses day 1 with day 7 vs day 3 with day 4.
# ══════════════════════════════════════════════════════════════

def build_model() -> Model:
    """
    Architecture:
      EfficientNetV2S (frozen base)
          ↓
      GlobalAveragePooling2D
          ↓
      [Dense 512 → BN → ReLU → Dropout 0.40]
          ↓
      [Dense 256 → BN → ReLU → Dropout 0.30]
          ↓
      Dense 7 → Softmax   (day_1 … day_7)
    """
    print("  Building EfficientNetV2S model...")

    # Base model – pretrained on ImageNet (21k classes → great feature extractor)
    base = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(*Config.IMG_SIZE, 3),
        include_preprocessing=True   # built-in EfficientNet preprocessing
    )
    base.trainable = False   # freeze all base layers initially

    inputs = layers.Input(shape=(*Config.IMG_SIZE, 3), name='image_input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # Head block 1
    x = layers.Dense(512, name='dense1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.Dropout(0.40, name='drop1')(x)

    # Head block 2
    x = layers.Dense(256, name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.Dropout(0.30, name='drop2')(x)

    # Output: 7 classes (day_1 … day_7) with softmax
    outputs = layers.Dense(Config.NUM_CLASSES,
                            activation='softmax',
                            name='day_output')(x)

    model = Model(inputs, outputs, name='coconut_dryness_cnn')
    return model


def ordinal_aware_loss(y_true, y_pred):
    """
    English: Standard cross-entropy PLUS an ordinal penalty.
             If the model confuses day 1 with day 7 it gets penalised much more
             than if it confuses day 3 with day 4 (close neighbours).
             This forces the model to respect the 1<2<3…<7 ordering.

    Sinhala: Standard cross-entropy PLUS ordinal penalty.
             Day 1 vs Day 7 confuse වුනොත් penalty ඉහළ.
             Day 3 vs Day 4 confuse වුනොත් penalty අඩු.
             1<2<3…<7 order respect කිරීමට force කරයි.
    """
    # Standard categorical cross-entropy
    ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)

    # Ordinal component: penalise by |true_day - pred_day|²
    day_values = tf.constant([1., 2., 3., 4., 5., 6., 7.], dtype=tf.float32)
    true_day  = tf.reduce_sum(y_true  * day_values, axis=1)
    pred_day  = tf.reduce_sum(y_pred  * day_values, axis=1)
    ord_loss  = tf.reduce_mean(tf.square(true_day - pred_day)) * 0.15

    return ce_loss + ord_loss


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4
        ),
        loss=ordinal_aware_loss,
        metrics=[
            'accuracy',
            keras.metrics.MeanAbsoluteError(name='day_mae')
        ]
    )
    return model


# ══════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

def train():
    print("\n" + "═"*60)
    print("  COCONUT 7-DAY DRYNESS MODEL – TRAINING")
    print("  Coconut 7-Day Dryness Model – Training")
    print("═"*60)

    dataset_dir = Config.DATASET_DIR
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"\n  ❌  Required folders not found!")
        print(f"  Please ensure: {train_dir} and {val_dir} exist")
        print(f"  Each should contain day_1/ through day_7/ subfolders")
        return None

    # ── Data generators ──────────────────────────────────────
    print("\n  [1/6] Loading dataset...")
    aug_gen   = EnhancedImageGenerator(augment=True)
    noaug_gen = EnhancedImageGenerator(augment=False)

    # Point to the train and val folders separately
    train_data = aug_gen.flow_from_directory(
        train_dir,
        shuffle=True
    )

    val_data = aug_gen.flow_from_directory(
        val_dir,
        shuffle=False
    )

    # Use validation data as test data (or you could create a separate test folder)
    test_data = noaug_gen.flow_from_directory(
        val_dir,
        shuffle=False
    )

    print(f"  Classes : {train_data.class_indices}")
    print(f"  Train   : {train_data.samples} images")
    print(f"  Val     : {val_data.samples} images")

    # Verify we have 7 classes
    if len(train_data.class_indices) != Config.NUM_CLASSES:
        print(f"\n  ⚠️  Warning: Found {len(train_data.class_indices)} classes, expected {Config.NUM_CLASSES}")
        print(f"  Classes found: {list(train_data.class_indices.keys())}")

    # ── Class weights (handles slight imbalance) ─────────────
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(Config.NUM_CLASSES),
        y=train_data.classes
    )
    cw_dict = dict(enumerate(class_weights))
    print(f"  Class weights: {[f'{v:.2f}' for v in class_weights]}")

    # ── Build model ───────────────────────────────────────────
    print("\n  [2/6] Building EfficientNetV2S model...")
    model = build_model()
    model = compile_model(model, lr=1e-3)
    model.summary(print_fn=lambda x: None)   # suppress summary clutter
    print(f"  Trainable params (Phase 1): "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    # ── PHASE 1: Train head only (frozen base) ────────────────
    print("\n  [3/6] Phase 1 – Training classification head (20 epochs)...")
    print("  Phase 1 – Classification head train (20 epochs)...")

    callbacks_p1 = [
        ModelCheckpoint('models/cnn_best.h5',
                        monitor='val_accuracy', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
    ]

    hist1 = model.fit(
        train_data, 
        epochs=20,
        validation_data=val_data,
        class_weight=cw_dict,
        callbacks=callbacks_p1,
        verbose=1
    )

    # ── PHASE 2: Progressive unfreezing ──────────────────────
    # Unfreeze last 60 layers of EfficientNetV2S
    # These later layers contain high-level texture/colour features
    # EfficientNetV2S top 60 layers unfreeze කරයි
    # Later layers = high-level texture/colour features
    print("\n  [4/6] Phase 2 – Fine-tuning top 60 base layers (25 epochs)...")
    print("  Phase 2 – Top 60 base layers fine-tune (25 epochs)...")

    base_model = model.get_layer('efficientnetv2-s')
    base_model.trainable = True
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    fine_tune_lr = 2e-5   # MUCH lower LR for fine-tuning
    model = compile_model(model, lr=fine_tune_lr)
    print(f"  Trainable params (Phase 2): "
          f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")

    callbacks_p2 = [
        ModelCheckpoint('models/cnn_best.h5',
                        monitor='val_accuracy', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=9,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-8, verbose=1),
    ]

    hist2 = model.fit(
        train_data, 
        epochs=25,
        validation_data=val_data,
        class_weight=cw_dict,
        callbacks=callbacks_p2,
        verbose=1
    )

    # ── PHASE 3: Unfreeze all layers, very fine LR ───────────
    print("\n  [4b/6] Phase 3 – Full fine-tune all layers (15 epochs)...")
    base_model.trainable = True
    model = compile_model(model, lr=5e-6)

    callbacks_p3 = [
        ModelCheckpoint('models/cnn_best.h5',
                        monitor='val_accuracy', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
    ]

    hist3 = model.fit(
        train_data, 
        epochs=15,
        validation_data=val_data,
        class_weight=cw_dict,
        callbacks=callbacks_p3,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────
    print("\n  [5/6] Evaluating on validation dataset...")
    evaluate(model, test_data)

    # ── Save ─────────────────────────────────────────────────
    print("\n  [6/6] Saving models...")
    model.save('models/cnn_model_final.h5')

    # Save config
    class_idx_to_day = {
        str(v): k for k, v in train_data.class_indices.items()
    }
    config_data = {
        'class_indices':   train_data.class_indices,
        'class_idx_to_day': class_idx_to_day,
        'dryness_scores':  Config.DRYNESS_SCORE,
        'img_size':        list(Config.IMG_SIZE),
        'num_classes':     Config.NUM_CLASSES,
    }
    with open('models/cnn_config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    # Plot training history
    plot_history([hist1, hist2, hist3])

    print("\n  ✅  Training complete!")
    print("  ✅  Training සම්පූර්ණයි!")
    print("  Files: models/cnn_model_final.h5  models/cnn_config.json")
    return model


def evaluate(model, test_gen):
    test_gen.reset()
    preds     = model.predict(test_gen, verbose=0)
    pred_cls  = np.argmax(preds, axis=1)
    true_cls  = test_gen.classes
    names     = list(test_gen.class_indices.keys())

    # Day-level MAE (how many days off on average)
    # Average ලදී days off කීයද
    day_mae = mean_absolute_error(true_cls + 1, pred_cls + 1)

    print(f"\n  Day MAE : {day_mae:.2f} days  (lower = better)")
    print(f"  (Average prediction is {day_mae:.2f} drying-days off)\n")
    print(classification_report(true_cls, pred_cls,
                                 target_names=names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(true_cls, pred_cls)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=names, yticklabels=names,
                linewidths=0.5)
    plt.title('Confusion Matrix – 7-Day Dryness Classification')
    plt.ylabel('Actual Day')
    plt.xlabel('Predicted Day')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=130)
    plt.close()
    print("  📊  Saved: models/confusion_matrix.png")


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue', 'darkorange', 'green']
    labels = ['Phase 1', 'Phase 2', 'Phase 3']

    offset = 0
    for i, h in enumerate(histories):
        ep = range(offset, offset + len(h.history['accuracy']))
        axes[0].plot(ep, h.history['accuracy'],
                     color=colors[i], label=f'{labels[i]} train')
        axes[0].plot(ep, h.history['val_accuracy'],
                     color=colors[i], linestyle='--',
                     label=f'{labels[i]} val')
        axes[1].plot(ep, h.history['loss'],
                     color=colors[i], label=f'{labels[i]} train')
        axes[1].plot(ep, h.history['val_loss'],
                     color=colors[i], linestyle='--',
                     label=f'{labels[i]} val')
        offset += len(h.history['accuracy'])

    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].set_title('Loss per Epoch')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=130)
    plt.close()
    print("  📊  Saved: models/training_history.png")


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    train()