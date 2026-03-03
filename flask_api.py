"""
╔══════════════════════════════════════════════════════════════════╗
║  FLASK API with COPRA IMAGE VALIDATION                          ║
║  Validates images are actually copra before prediction          ║
║  Copra images ද නොද validate කරයි prediction පෙර               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, base64, logging
import numpy as np
import joblib
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ── Global model holders ───────────────────────────────────
cnn_model  = None
oil_model  = None
oil_scaler = None
cnn_cfg    = {}
oil_cfg    = {}


def load_models():
    global cnn_model, oil_model, oil_scaler, cnn_cfg, oil_cfg

    # CNN model (7-class day classifier)
    cnn_path = 'models/cnn_model_final.h5'
    cnn_cfg_path = 'models/cnn_config.json'
    if os.path.exists(cnn_path) and os.path.exists(cnn_cfg_path):
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(
            cnn_path,
            custom_objects={'ordinal_aware_loss': lambda y,p: p}
        )
        with open(cnn_cfg_path) as f:
            cnn_cfg = json.load(f)
        log.info("✅  CNN 7-day dryness model loaded")
    else:
        log.warning("⚠️   CNN model not found")

    # Oil regression model
    oil_path = 'models/oil_model.pkl'
    scl_path = 'models/oil_scaler.pkl'
    cfg_path = 'models/oil_config.json'
    if os.path.exists(oil_path):
        oil_model  = joblib.load(oil_path)
        oil_scaler = joblib.load(scl_path)
        with open(cfg_path) as f:
            oil_cfg = json.load(f)
        log.info("✅  Oil regression model loaded")
    else:
        log.error("❌  oil_model.pkl not found")


# ══════════════════════════════════════════════════════════════════
#  COPRA IMAGE VALIDATION FUNCTIONS
#  Copra Image Validate කරන Functions
# ══════════════════════════════════════════════════════════════════

def decode_image(b64_str: str) -> np.ndarray:
    """Base64 string → BGR numpy array (OpenCV format)."""
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def validate_copra_image(image_bgr: np.ndarray) -> dict:
    """
    English: Checks if image looks like coconut copra.
             Returns validation result with confidence score.
    
    Sinhala: Image coconut copra ලෙස දිස්වේද check කරයි.
             Confidence score සමඟ validation result return කරයි.
    
    Validation checks / Validate කරන දේවල්:
    1. Color range (should be brownish/white, not bright colors)
    2. Not too dark (not a black image)
    3. Not too bright (not overexposed)
    4. Has texture (not a solid color)
    5. Shape characteristics (organic, not geometric)
    """
    
    if image_bgr is None or image_bgr.size == 0:
        return {
            'is_copra': False,
            'confidence': 0.0,
            'reason': 'Invalid image / වලංගු නොවන image',
            'details': 'Image could not be decoded'
        }
    
    h, w = image_bgr.shape[:2]
    if h < 50 or w < 50:
        return {
            'is_copra': False,
            'confidence': 0.0,
            'reason': 'Image too small / Image ඉතා කුඩා',
            'details': f'Size: {w}×{h} (minimum 50×50 required)'
        }
    
    # ── CHECK 1: Color Analysis ────────────────────────────
    # Copra should be in brown/tan/cream/white range
    # NOT bright colors like red, blue, green faces, etc.
    
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Brown/tan/cream range in HSV:
    # Hue: 10-30 (orange-brown), Saturation: 20-180, Value: 80-255
    lower_copra1 = np.array([10, 20, 80])
    upper_copra1 = np.array([30, 180, 255])
    mask1 = cv2.inRange(hsv, lower_copra1, upper_copra1)
    
    # White/light cream (low saturation):
    # Hue: any, Saturation: 0-40, Value: 180-255
    lower_copra2 = np.array([0, 0, 180])
    upper_copra2 = np.array([180, 40, 255])
    mask2 = cv2.inRange(hsv, lower_copra2, upper_copra2)
    
    # Darker brown (for dried copra):
    # Hue: 10-30, Saturation: 30-200, Value: 50-150
    lower_copra3 = np.array([10, 30, 50])
    upper_copra3 = np.array([30, 200, 150])
    mask3 = cv2.inRange(hsv, lower_copra3, upper_copra3)
    
    copra_mask = cv2.bitwise_or(mask1, mask2)
    copra_mask = cv2.bitwise_or(copra_mask, mask3)
    
    copra_pixels = np.sum(copra_mask > 0)
    total_pixels = h * w
    copra_ratio = copra_pixels / total_pixels
    
    # ── CHECK 2: Brightness ────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    
    # Too dark (< 30) = probably not copra (unless very dark photo)
    # Too bright (> 240) = overexposed or white background
    if mean_brightness < 30:
        return {
            'is_copra': False,
            'confidence': 0.1,
            'reason': 'Image too dark / Image ඉතා අඳුරු',
            'details': f'Average brightness: {mean_brightness:.0f}/255'
        }
    
    if mean_brightness > 240:
        return {
            'is_copra': False,
            'confidence': 0.2,
            'reason': 'Image too bright / Image ඉතා දීප්ත',
            'details': f'Average brightness: {mean_brightness:.0f}/255 (overexposed)'
        }
    
    # ── CHECK 3: Texture / Not Solid Color ─────────────────
    # Calculate standard deviation - copra has texture variation
    std_dev = gray.std()
    
    if std_dev < 10:
        return {
            'is_copra': False,
            'confidence': 0.15,
            'reason': 'Image has no texture / Image texture නැත',
            'details': 'Looks like solid color or blank image'
        }
    
    # ── CHECK 4: Color Variety (Not too colorful) ──────────
    # Copra is not rainbow-colored or bright red/blue/green
    b_mean = image_bgr[:,:,0].mean()
    g_mean = image_bgr[:,:,1].mean()
    r_mean = image_bgr[:,:,2].mean()
    
    # Check for unnatural color dominance
    color_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
    
    # Faces/bright objects have high color channel differences
    if color_diff > 80 and copra_ratio < 0.3:
        return {
            'is_copra': False,
            'confidence': 0.25,
            'reason': 'Colors do not match copra / වර්ණ copra සමඟ නොගැලපේ',
            'details': 'Image has unnatural color distribution (faces, bright objects, etc.)'
        }
    
    # ── CHECK 5: Edge Density (Organic shapes) ─────────────
    # Copra has organic, irregular edges
    # Faces/geometric objects have different edge patterns
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels
    
    # Too many edges = busy scene (not isolated copra)
    # Too few edges = blank/smooth surface (not copra texture)
    if edge_density < 0.02:
        return {
            'is_copra': False,
            'confidence': 0.2,
            'reason': 'No texture detected / Texture detect නොවේ',
            'details': 'Image too smooth (not copra-like texture)'
        }
    
    # ── FINAL CONFIDENCE SCORE ─────────────────────────────
    # Combine all checks into confidence score
    
    confidence = 0.0
    
    # Color match (40% weight)
    if copra_ratio > 0.5:
        confidence += 0.40
    elif copra_ratio > 0.3:
        confidence += 0.25
    elif copra_ratio > 0.15:
        confidence += 0.15
    
    # Brightness range (20% weight)
    if 60 < mean_brightness < 200:
        confidence += 0.20
    elif 40 < mean_brightness < 220:
        confidence += 0.10
    
    # Texture (20% weight)
    if std_dev > 30:
        confidence += 0.20
    elif std_dev > 20:
        confidence += 0.15
    elif std_dev > 10:
        confidence += 0.10
    
    # Edge density (10% weight)
    if 0.05 < edge_density < 0.30:
        confidence += 0.10
    elif 0.03 < edge_density < 0.40:
        confidence += 0.05
    
    # Color naturalness (10% weight)
    if color_diff < 50:
        confidence += 0.10
    elif color_diff < 70:
        confidence += 0.05
    
    # ── DECISION ───────────────────────────────────────────
    is_copra = confidence >= 0.50  # Threshold: 50% confidence
    
    if is_copra:
        reason = 'Image looks like copra / Image copra ලෙස දිස්වේ'
    else:
        reason = 'Image does not look like copra / Image copra ලෙස නොපෙනේ'
    
    return {
        'is_copra': is_copra,
        'confidence': round(confidence, 2),
        'reason': reason,
        'details': {
            'copra_color_ratio': round(copra_ratio, 2),
            'brightness': round(mean_brightness, 1),
            'texture_std': round(std_dev, 1),
            'edge_density': round(edge_density, 3),
            'color_variation': round(color_diff, 1),
        }
    }


# ══════════════════════════════════════════════════════════════════
#  ENHANCED IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════

def enhance_for_dryness(image_bgr: np.ndarray) -> np.ndarray:
    """LAB color space + CLAHE enhancement"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


def image_to_dryness(b64_str: str) -> dict:
    """Full pipeline: base64 → enhance → CNN predict → dryness score"""
    if cnn_model is None:
        return {
            'day_label': 'day_7', 'day_number': 7,
            'dryness_score': 1.0, 'confidence_pct': 100.0,
            'cnn_used': False,
        }
    
    img_bgr = decode_image(b64_str)
    img_size = tuple(cnn_cfg.get('img_size', [300, 300]))
    img_rgb = enhance_for_dryness(img_bgr)
    img_res = cv2.resize(img_rgb, img_size)
    img_in = img_res.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, 0)
    
    probs = cnn_model.predict(img_in, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100.0
    
    idx_to_day = cnn_cfg.get('class_idx_to_day', {})
    day_label = idx_to_day.get(str(pred_idx), f'day_{pred_idx+1}')
    day_number = int(day_label.split('_')[1])
    
    dryness_scores = cnn_cfg.get('dryness_scores', {f'day_{i}': i/7 for i in range(1,8)})
    soft_dryness = sum(
        float(probs[idx]) * dryness_scores.get(
            idx_to_day.get(str(idx), f'day_{idx+1}'), idx/7
        )
        for idx in range(len(probs))
    )
    
    return {
        'day_label': day_label,
        'day_number': day_number,
        'dryness_score': round(float(soft_dryness), 4),
        'confidence_pct': round(confidence, 1),
        'cnn_used': True,
    }


def aggregate_dryness(image_analyses: list) -> float:
    """Average dryness score across all images (using median)"""
    scores = [a['dryness_score'] for a in image_analyses]
    return float(np.median(scores))


def predict_oil_ml(weight_g: float, dryness_score: float) -> dict:
    """Use regression model to predict oil in mL"""
    if oil_model is None:
        avg_ratio = oil_cfg.get('avg_ml_per_g', 0.54)
        predicted = weight_g * avg_ratio * dryness_score
        return {'predicted_oil_ml': round(predicted, 1), 'method': 'ratio_fallback'}
    
    X = np.array([[weight_g, dryness_score]])
    X_sc = oil_scaler.transform(X)
    oil_ml = float(oil_model.predict(X_sc)[0])
    oil_ml = max(0.0, oil_ml)
    
    return {
        'predicted_oil_ml': round(oil_ml, 1),
        'predicted_oil_litres': round(oil_ml / 1000, 3),
        'oil_per_kg_ml': round((oil_ml / weight_g) * 1000, 1),
        'low_estimate_ml': round(oil_ml * 0.92, 1),
        'high_estimate_ml': round(oil_ml * 1.08, 1),
        'method': 'trained_regression',
    }


def quality_grade(oil_per_kg: float) -> str:
    if oil_per_kg >= 550: return 'A'
    if oil_per_kg >= 460: return 'B'
    if oil_per_kg >= 380: return 'C'
    return 'D'


def recommendations(dryness_score: float, avg_day: float, oil_per_kg: float) -> list:
    recs = []
    if avg_day < 4:
        recs.append(f"Copra appears to be around day {avg_day:.0f} dryness. Consider drying for more days.")
        recs.append(f"Copra දළ වශයෙන් day {avg_day:.0f} dryness. Days more dry කරන්න.")
    elif avg_day < 6:
        recs.append("Good dryness level. More drying could maximise oil yield.")
        recs.append("හොඳ dryness level. Dry කිරීම ටිකක් increase කරන්න.")
    else:
        recs.append("Excellent! Copra is well-dried. Optimal for oil extraction.")
        recs.append("Excellent! Copra හොඳින් dried. Oil extraction ලදී optimal!")
    
    if oil_per_kg >= 550:
        recs.append("Grade A quality – premium oil yield expected.")
    elif oil_per_kg >= 460:
        recs.append("Grade B quality – good commercial yield.")
    else:
        recs.append("Grade C/D – consider improving drying process.")
    
    return recs


# ══════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'cnn_ready': cnn_model is not None,
        'oil_model_ready': oil_model is not None,
        'mode': '7-day progressive dryness with image validation',
    })


@app.route('/validate_images', methods=['POST'])
def validate_images():
    """
    NEW ENDPOINT: Validates if images are copra before full prediction.
    Request: {"images": ["base64...", "base64...", ...]}
    Response: {"all_valid": true/false, "validations": [...]}
    """
    try:
        data = request.get_json(force=True)
        b64_list = data.get('images', [])
        
        if not b64_list:
            return jsonify({'error': 'No images provided'}), 400
        
        validations = []
        for i, b64 in enumerate(b64_list):
            try:
                img = decode_image(b64)
                validation = validate_copra_image(img)
                validation['image_index'] = i + 1
                validations.append(validation)
                
                log.info(f"Image {i+1}: is_copra={validation['is_copra']}, "
                        f"confidence={validation['confidence']}")
            
            except Exception as e:
                validations.append({
                    'image_index': i + 1,
                    'is_copra': False,
                    'confidence': 0.0,
                    'reason': f'Error processing image: {str(e)}',
                })
        
        # Check if all images are valid
        all_valid = all(v['is_copra'] for v in validations)
        invalid_indices = [v['image_index'] for v in validations if not v['is_copra']]
        
        return jsonify({
            'all_valid': all_valid,
            'valid_count': sum(1 for v in validations if v['is_copra']),
            'total_count': len(validations),
            'invalid_indices': invalid_indices,
            'validations': validations,
        })
    
    except Exception as e:
        log.exception("Validation failed")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with BUILT-IN validation.
    Validates images first, then predicts if all valid.
    """
    if oil_model is None:
        return jsonify({'error': 'Oil model not loaded. Run train_oil_model.py'}), 503
    
    try:
        data = request.get_json(force=True)
        b64_list = data.get('images', [])
        weight_g = float(data.get('weight_g', 0))
        
        if weight_g <= 0:
            return jsonify({'error': 'weight_g must be > 0'}), 400
        if not b64_list:
            return jsonify({'error': 'No images provided'}), 400
        
        log.info(f"Predict: {len(b64_list)} images, {weight_g}g")
        
        # ── STEP 1: VALIDATE ALL IMAGES FIRST ───────────────
        log.info("Step 1: Validating images...")
        invalid_images = []
        
        for i, b64 in enumerate(b64_list):
            img = decode_image(b64)
            validation = validate_copra_image(img)
            
            if not validation['is_copra']:
                invalid_images.append({
                    'index': i + 1,
                    'reason': validation['reason'],
                    'confidence': validation['confidence'],
                })
                log.warning(f"Image {i+1} rejected: {validation['reason']}")
        
        # If ANY image is invalid, reject the whole request
        if invalid_images:
            return jsonify({
                'status': 'error',
                'error': 'invalid_images',
                'message': 'Some images do not look like coconut copra. Please use only copra photos.',
                'message_si': 'සමහර images coconut copra ලෙස නොපෙනේ. Copra photos පමණක් use කරන්න.',
                'invalid_images': invalid_images,
                'valid_count': len(b64_list) - len(invalid_images),
                'total_count': len(b64_list),
            }), 400
        
        # ── STEP 2: ALL VALID - Run CNN dryness analysis ────
        log.info("Step 2: All images valid. Analyzing dryness...")
        image_analyses = []
        
        for i, b64 in enumerate(b64_list):
            try:
                analysis = image_to_dryness(b64)
                analysis['image_index'] = i + 1
                image_analyses.append(analysis)
                log.info(f"  Image {i+1}: {analysis['day_label']} "
                        f"({analysis['confidence_pct']:.1f}%) "
                        f"dryness={analysis['dryness_score']:.3f}")
            except Exception as e:
                log.warning(f"  Image {i+1} analysis failed: {e}")
        
        if not image_analyses:
            return jsonify({'error': 'All image analyses failed'}), 500
        
        # ── STEP 3: Aggregate dryness ───────────────────────
        dryness = aggregate_dryness(image_analyses)
        avg_day = dryness * 7.0
        
        # ── STEP 4: Predict oil ─────────────────────────────
        oil_result = predict_oil_ml(weight_g, dryness)
        
        # ── STEP 5: Grade and recommendations ───────────────
        per_kg = oil_result.get('oil_per_kg_ml', 0)
        grade = quality_grade(per_kg)
        recs = recommendations(dryness, avg_day, per_kg)
        
        # ── Build response ──────────────────────────────────
        response = {
            'status': 'success',
            'weight_g': weight_g,
            'weight_kg': round(weight_g / 1000, 3),
            'images_analysed': len(image_analyses),
            
            'predicted_oil_ml': oil_result['predicted_oil_ml'],
            'predicted_oil_litres': oil_result.get('predicted_oil_litres', 0),
            'oil_per_kg_ml': per_kg,
            'low_estimate_ml': oil_result.get('low_estimate_ml', 0),
            'high_estimate_ml': oil_result.get('high_estimate_ml', 0),
            
            'dryness_score': round(dryness, 3),
            'avg_drying_day': round(avg_day, 1),
            'dryness_percent': round(dryness * 100, 1),
            
            'quality_grade': grade,
            'quality_score': dryness,  # For compatibility
            'recommendations': recs,
            
            'image_analyses': [
                {
                    'image_index': a['image_index'],
                    'day_label': a['day_label'],
                    'day_number': a['day_number'],
                    'confidence_pct': a['confidence_pct'],
                    'dryness_score': a['dryness_score'],
                }
                for a in image_analyses
            ],
        }
        
        return jsonify(response)
    
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    load_models()
    print("\n" + "═"*50)
    print("  Coconut Oil Flask API – 7-Day with Validation")
    print("  http://localhost:5000")
    print("  http://localhost:5000/health")
    print("  http://localhost:5000/validate_images  (NEW!)")
    print("═"*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)