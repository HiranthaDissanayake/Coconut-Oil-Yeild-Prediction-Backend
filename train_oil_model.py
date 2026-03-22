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

DRYNESS_SCORE = {
    'day_1': 1/7, 'day_2': 2/7, 'day_3': 3/7, 'day_4': 4/7,
    'day_5': 5/7, 'day_6': 6/7, 'day_7': 7/7,
}

print("\n" + "═"*70)
print("  🔥 IMPROVED OIL YIELD MODEL – TRAINING")
print("  Fixed: Now respects BOTH weight AND dryness!")
print("═"*70)

# ══════════════════════════════════════════════════════════
#  STEP 1: LOAD AND VALIDATE DATA
# ══════════════════════════════════════════════════════════
print("\n  [1/5] Loading batch data from batch_data.csv...")

df = pd.read_csv('batch_data.csv')
print(f"\n  ✅  {len(df)} batches loaded")

# Convert to grams
df['weight_g'] = df['whole_batch_weight_kg'] * 1000
df['oil_per_gram'] = df['oil_extracted_ml'] / df['weight_g']
df['dryness_score'] = df['drying_days'].apply(lambda d: min(d, 7) / 7.0)

print("\n  Data preview:")
print(df[['batch_name', 'whole_batch_weight_kg', 'drying_days', 'oil_extracted_ml']].to_string(index=False))

# ══════════════════════════════════════════════════════════
#  🔥 CRITICAL VALIDATION: Check for weight variation
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  CRITICAL VALIDATION - Checking data quality...")
print("="*70)

unique_weights = df['whole_batch_weight_kg'].nunique()
weight_std = df['whole_batch_weight_kg'].std()

print(f"\n  Unique weights in data: {unique_weights}")
print(f"  Weight std deviation: {weight_std:.2f} kg")

if unique_weights < 3:
    print("  → Use batch_data_IMPROVED.csv instead")
    print("  → Or collect data with varied weights (1kg, 2kg, 3kg, etc.)")
    print("\n  Continuing with FORMULA-BASED fallback model...")
    use_formula = True
else:
    print(f"\n  ✅  Good variation! {unique_weights} different weights.")
    use_formula = False

# ══════════════════════════════════════════════════════════
#  STEP 2: TRAIN MODELS
# ══════════════════════════════════════════════════════════
print("\n  [2/5] Training regression models...")

X = df[['weight_g', 'dryness_score']].values
y = df['oil_extracted_ml'].values

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Model A: Linear
lin_model = LinearRegression()
lin_model.fit(X_sc, y)
lin_r2 = r2_score(y, lin_model.predict(X_sc))
lin_mae = mean_absolute_error(y, lin_model.predict(X_sc))

# Model B: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb_model.fit(X_sc, y)
gb_r2 = r2_score(y, gb_model.predict(X_sc))
gb_mae = mean_absolute_error(y, gb_model.predict(X_sc))

print(f"\n  Model A (Linear): R²={lin_r2:.4f}  MAE={lin_mae:.1f}mL")
print(f"  Model B (GradBoost): R²={gb_r2:.4f}  MAE={gb_mae:.1f}mL")

if gb_r2 >= lin_r2:
    best_model = gb_model
    best_model_name = 'gradient_boosting'
    print(f"\n  ✅  Selected: Gradient Boosting")
else:
    best_model = lin_model
    best_model_name = 'linear_regression'
    print(f"\n  ✅  Selected: Linear Regression")

# ══════════════════════════════════════════════════════════
#  🔥 STEP 3: VALIDATE WEIGHT DEPENDENCY
# ══════════════════════════════════════════════════════════
print("\n  [3/5] Validating weight dependency...")

test_weights = [1000, 2000, 3000, 4000]
test_dryness = 1.0

print(f"\n  Test: Same dryness, different weights")
print(f"  {'Weight':<10} {'Predicted':<12} {'Should be':<15}")
print("  " + "-"*40)

predictions = []
for i, w in enumerate(test_weights):
    X_test = scaler.transform([[w, test_dryness]])
    pred = best_model.predict(X_test)[0]
    predictions.append(pred)
    
    if i == 0:
        base = pred
        expected = "base"
    else:
        ratio = w / test_weights[0]
        expected = f"{ratio}x base"
    
    print(f"  {w}g{'':<5} {pred:<12.1f} {expected:<15}")

# Validate increasing
increasing = all(predictions[i] < predictions[i+1] 
                for i in range(len(predictions)-1))

if not increasing:
    print("\n  ❌ FAILED! Oil doesn't increase with weight!")
    print("  → Using formula-based model")
    use_formula = True
else:
    print("\n  ✅ PASSED! Oil increases with weight")

# ══════════════════════════════════════════════════════════
#  STEP 4: SAVE
# ══════════════════════════════════════════════════════════
print("\n  [4/5] Saving models...")

if use_formula:
    avg_ml_per_kg = df['oil_per_gram'].mean() * 1000
    formula_model = {
        'type': 'formula',
        'avg_ml_per_kg': float(avg_ml_per_kg),
    }
    joblib.dump(formula_model, 'models/oil_model.pkl')
    best_model_name = 'formula_based'
else:
    joblib.dump(best_model, 'models/oil_model.pkl')

joblib.dump(scaler, 'models/oil_scaler.pkl')

config = {
    'oil_model_type': best_model_name,
    'dryness_scores': DRYNESS_SCORE,
    'avg_ml_per_kg': float(df['oil_per_gram'].mean() * 1000),
    'use_formula': use_formula,
    'num_batches': len(df),
}

with open('models/oil_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("  💾  models/oil_model.pkl")
print("  💾  models/oil_scaler.pkl")
print("  💾  models/oil_config.json")

# ══════════════════════════════════════════════════════════
#  STEP 5: FINAL TEST
# ══════════════════════════════════════════════════════════
print("\n  [5/5] Final test - Different weights:")

test_cases = [
    (400, 1.0, "400g day 7"),
    (800, 1.0, "800g day 7"),
    (1200, 1.0, "1200g day 7"),
    (1000, 0.43, "1000g day 3"),
    (1000, 0.71, "1000g day 5"),
    (1000, 1.0, "1000g day 7"),
]

print(f"\n  {'Test':<20} {'Predicted':<12}")
print("  " + "-"*35)

for w, d, desc in test_cases:
    if use_formula:
        pred = (w / 1000) * config['avg_ml_per_kg'] * d
    else:
        X_test = scaler.transform([[w, d]])
        pred = best_model.predict(X_test)[0]
    print(f"  {desc:<20} {pred:<12.1f}mL")

print("\n" + "="*70)
print("  ✅  Training complete!")
if use_formula:
    print("  Using FORMULA (guarantees correct scaling)")
else:
    print("  Using TRAINED MODEL (validated)")
print("  Next: python flask_api.py")
print("="*70 + "\n")