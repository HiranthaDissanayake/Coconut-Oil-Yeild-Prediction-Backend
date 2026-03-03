import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

# ── Dryness score mapping (same as CNN config) ─────────────
# Day number → dryness score (0‥1)
# Higher score = more dried = more oil expected
DRYNESS_SCORE = {
    'day_1': 1/7,   # ≈ 0.143  fresh
    'day_2': 2/7,
    'day_3': 3/7,
    'day_4': 4/7,
    'day_5': 5/7,
    'day_6': 6/7,
    'day_7': 7/7,   # = 1.000  fully dried
}

print("\n" + "═"*55)
print("  OIL YIELD REGRESSION MODEL – TRAINING")
print("  Oil Yield Regression Model – Training")
print("═"*55)

# ══════════════════════════════════════════════════════════
#  STEP 1: LOAD YOUR REAL BATCH DATA
#  batch_data.csv ලදී ඔබේ real data load කරයි
# ══════════════════════════════════════════════════════════
print("\n  [1/4] Loading batch data from batch_data.csv...")

df = pd.read_csv('batch_data.csv')
print(f"\n  ✅  {len(df)} batches loaded")
print(df.to_string(index=False))

# Convert weight to grams
df['weight_g']       = df['whole_batch_weight_kg'] * 1000
df['oil_per_gram']   = df['oil_extracted_ml'] / df['weight_g']

# Map drying_days to dryness score
# All your batches are 7-day dried → dryness_score = 1.0
df['dryness_score'] = df['drying_days'].apply(
    lambda d: min(d, 7) / 7.0
)

print(f"\n  Oil per gram stats:")
print(f"    Min : {df['oil_per_gram'].min():.4f} mL/g")
print(f"    Max : {df['oil_per_gram'].max():.4f} mL/g")
print(f"    Mean: {df['oil_per_gram'].mean():.4f} mL/g  "
      f"(≈ {df['oil_per_gram'].mean()*1000:.0f} mL per kg)")

# ══════════════════════════════════════════════════════════
#  STEP 2: TRAIN REGRESSION MODELS
#  Weight (g) + Dryness Score → Oil (mL)
# ══════════════════════════════════════════════════════════
print("\n  [2/4] Training regression models...")

"""
English: We train TWO regression models and keep the better one.

  Model A – Linear Regression (simple):
    oil_mL = a × weight_g + b × dryness_score + c
    Fast, interpretable, good when data is limited (5 batches).

  Model B – Gradient Boosting (smarter):
    Learns non-linear relationship between weight/dryness and oil.
    Better when dryness has a non-linear effect on yield.

  We pick whichever has higher R² on the training data.

Sinhala: Regression models 2ක් train කර හොඳ model keep කරයි.

  Model A – Linear (සරළ):
    oil_mL = a × weight_g + b × dryness_score + c

  Model B – Gradient Boosting (smart):
    Weight/dryness සහ oil අතර non-linear relationship ඉගෙනගනී.

  R² ඉහළ model select කරයි.
"""

# Features: [weight_g,  dryness_score]
X = df[['weight_g', 'dryness_score']].values
y = df['oil_extracted_ml'].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# Model A: Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_sc, y)
lin_r2  = r2_score(y, lin_model.predict(X_sc))
lin_mae = mean_absolute_error(y, lin_model.predict(X_sc))

# Model B: Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_sc, y)
gb_r2  = r2_score(y, gb_model.predict(X_sc))
gb_mae = mean_absolute_error(y, gb_model.predict(X_sc))

print(f"\n  Model A (Linear Regression):")
print(f"    R²  = {lin_r2:.4f}   MAE = {lin_mae:.1f} mL")
print(f"  Model B (Gradient Boosting):")
print(f"    R²  = {gb_r2:.4f}   MAE = {gb_mae:.1f} mL")

# Pick better model
if gb_r2 >= lin_r2:
    best_model      = gb_model
    best_model_name = 'gradient_boosting'
    print(f"\n  ✅  Selected: Gradient Boosting (R²={gb_r2:.4f})")
else:
    best_model      = lin_model
    best_model_name = 'linear_regression'
    print(f"\n  ✅  Selected: Linear Regression (R²={lin_r2:.4f})")

# ══════════════════════════════════════════════════════════
#  STEP 3: SHOW PREDICTIONS ON YOUR 5 BATCHES
#  ඔබේ Batches 5 ලදී predictions show කරයි
# ══════════════════════════════════════════════════════════
print("\n  [3/4] Checking predictions on your 5 batches...")
print(f"\n  {'Batch':<12} {'Weight(kg)':<12} {'Actual(mL)':<14}"
      f"{'Predicted(mL)':<16} {'Error':<10}")
print("  " + "-"*62)

for _, row in df.iterrows():
    X_t = scaler.transform([[row['weight_g'], row['dryness_score']]])
    pred = best_model.predict(X_t)[0]
    err  = pred - row['oil_extracted_ml']
    print(f"  {row['batch_name']:<12} "
          f"{row['whole_batch_weight_kg']:<12.1f} "
          f"{row['oil_extracted_ml']:<14.0f} "
          f"{pred:<16.0f} "
          f"{err:+.0f} mL")

# ══════════════════════════════════════════════════════════
#  STEP 4: SAVE EVERYTHING
# ══════════════════════════════════════════════════════════
print("\n  [4/4] Saving models & config...")

joblib.dump(best_model, 'models/oil_model.pkl')
joblib.dump(scaler,     'models/oil_scaler.pkl')

# Full config used by flask_api.py
config = {
    'oil_model_type':    best_model_name,
    'oil_model_r2':      float(max(lin_r2, gb_r2)),
    'dryness_scores':    DRYNESS_SCORE,
    'avg_ml_per_g':      float(df['oil_per_gram'].mean()),
    'min_ml_per_g':      float(df['oil_per_gram'].min()),
    'max_ml_per_g':      float(df['oil_per_gram'].max()),
    'num_batches':       len(df),
    'features':          ['weight_g', 'dryness_score'],
}

with open('models/oil_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("  💾  models/oil_model.pkl")
print("  💾  models/oil_scaler.pkl")
print("  💾  models/oil_config.json")

# Quick test
print("\n  Quick test predictions:")
tests = [
    (3000, 'day_5', 5/7),
    (5000, 'day_7', 1.0),
    (7000, 'day_6', 6/7),
]
for w, day, ds in tests:
    Xt  = scaler.transform([[w, ds]])
    ml  = best_model.predict(Xt)[0]
    print(f"    {w}g copra dried {day} → {ml:.0f} mL oil")

print("\n" + "═"*55)
print("  ✅  Oil model training complete!")
print("  ✅  Oil model training සම්පූර්ණයි!")
print("  Next: python flask_api.py")
print("═"*55 + "\n")