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

app = Flask(__name__)
CORS(app)

cnn_model = None
oil_model = None
oil_scaler = None
cnn_cfg = {}
oil_cfg = {}


# ══════════════════════════════════════════════════════════════
#  🔥 IMPROVED PREPROCESSING (MATCHES TRAINING!)
#  CRITICAL: Must match training preprocessing exactly
# ══════════════════════════════════════════════════════════════

def enhance_for_dryness(image_bgr: np.ndarray) -> np.ndarray:
    """
    IMPROVED: Multi-step enhancement (SAME AS TRAINING!)
    වැඩිදියුණු: Multi-step enhancement (TRAINING එක ලෙස!)
    """
    # Step 1: LAB + CLAHE
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    # Step 2: Bilateral filter
    rgb_filtered = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 3: Sharpen
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    rgb_sharpened = cv2.filter2D(rgb_filtered, -1, kernel)
    
    # Step 4: Blend
    rgb_final = cv2.addWeighted(rgb_filtered, 0.5, rgb_sharpened, 0.5, 0)
    rgb_final = cv2.cvtColor(rgb_final, cv2.COLOR_BGR2RGB)
    
    return rgb_final


def load_models():
    global cnn_model, oil_model, oil_scaler, cnn_cfg, oil_cfg

    # CNN model
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
        
        preprocessing = cnn_cfg.get('preprocessing', 'standard')
        log.info(f"✅  CNN loaded ({preprocessing} preprocessing)")
    else:
        log.warning("⚠️   CNN model not found")

    # Oil model
    oil_path = 'models/oil_model.pkl'
    if os.path.exists(oil_path):
        oil_model = joblib.load(oil_path)
        oil_scaler = joblib.load('models/oil_scaler.pkl')
        with open('models/oil_config.json') as f:
            oil_cfg = json.load(f)
        log.info("✅  Oil model loaded")
    else:
        log.error("❌  oil_model.pkl not found")


# ══════════════════════════════════════════════════════════════
#  COPRA IMAGE VALIDATION
# ══════════════════════════════════════════════════════════════

def decode_image(b64_str: str) -> np.ndarray:
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def validate_copra_image(image_bgr: np.ndarray) -> dict:
    """Enhanced validation"""
    
    if image_bgr is None or image_bgr.size == 0:
        return {'is_copra': False, 'confidence': 0.0, 'reason': 'Invalid image'}
    
    h, w = image_bgr.shape[:2]
    if h < 50 or w < 50:
        return {'is_copra': False, 'confidence': 0.0, 'reason': 'Image too small'}
    
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Copra color ranges
    mask1 = cv2.inRange(hsv, np.array([10, 20, 80]), np.array([30, 180, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
    mask3 = cv2.inRange(hsv, np.array([10, 30, 50]), np.array([30, 200, 150]))
    
    copra_mask = cv2.bitwise_or(mask1, mask2)
    copra_mask = cv2.bitwise_or(copra_mask, mask3)
    
    copra_ratio = np.sum(copra_mask > 0) / (h * w)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    std_dev = gray.std()
    
    if mean_brightness < 30:
        return {'is_copra': False, 'confidence': 0.1, 'reason': 'Too dark'}
    if mean_brightness > 240:
        return {'is_copra': False, 'confidence': 0.2, 'reason': 'Too bright'}
    if std_dev < 10:
        return {'is_copra': False, 'confidence': 0.15, 'reason': 'No texture'}
    
    # Calculate confidence
    confidence = 0.0
    if copra_ratio > 0.5: confidence += 0.40
    elif copra_ratio > 0.3: confidence += 0.25
    
    if 60 < mean_brightness < 200: confidence += 0.20
    if std_dev > 30: confidence += 0.20
    elif std_dev > 20: confidence += 0.15
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    if 0.05 < edge_density < 0.30: confidence += 0.10
    
    is_copra = confidence >= 0.50
    reason = 'Looks like copra' if is_copra else 'Does not look like copra'
    
    return {
        'is_copra': is_copra,
        'confidence': round(confidence, 2),
        'reason': reason,
    }


# ══════════════════════════════════════════════════════════════
#  IMPROVED DRYNESS DETECTION
# ══════════════════════════════════════════════════════════════

def image_to_dryness(b64_str: str) -> dict:
    """Full pipeline with IMPROVED preprocessing"""
    if cnn_model is None:
        return {
            'day_label': 'day_7', 'day_number': 7,
            'dryness_score': 1.0, 'confidence_pct': 100.0,
            'cnn_used': False,
        }
    
    img_bgr = decode_image(b64_str)
    img_size = tuple(cnn_cfg.get('img_size', [300, 300]))
    
    # CRITICAL: Use IMPROVED preprocessing if model was trained with it
    preprocessing = cnn_cfg.get('preprocessing', 'standard')
    if preprocessing == 'improved_v2':
        img_rgb = enhance_for_dryness(img_bgr)
    else:
        # Fallback to standard
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        img_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
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
        'probabilities': {f'day_{i+1}': round(float(probs[i]) * 100, 1) for i in range(len(probs))}
    }


def aggregate_dryness(image_analyses: list) -> float:
    """Weighted median"""
    scores = [a['dryness_score'] for a in image_analyses]
    confidences = [a['confidence_pct'] for a in image_analyses]
    
    weighted_scores = [s * c for s, c in zip(scores, confidences)]
    total_weight = sum(confidences)
    
    if total_weight > 0:
        return sum(weighted_scores) / total_weight
    else:
        return float(np.median(scores))


def predict_oil_ml(weight_g: float, dryness_score: float) -> dict:
    """Predict oil using trained model OR formula fallback"""

    if oil_model is None:
        avg_ratio = oil_cfg.get('avg_ml_per_kg', 540)
        predicted = (weight_g / 1000) * avg_ratio * dryness_score
        return {
            'predicted_oil_ml': round(predicted, 1),
            'method': 'fallback_no_model'
        }

    # 🔥 CASE 1: Formula-based model (dict)
    if isinstance(oil_model, dict):
        avg_ml_per_kg = oil_model.get('avg_ml_per_kg', 540)

        predicted = (weight_g / 1000) * avg_ml_per_kg * dryness_score

        return {
            'predicted_oil_ml': round(predicted, 1),
            'predicted_oil_litres': round(predicted / 1000, 3),
            'oil_per_kg_ml': round(avg_ml_per_kg * dryness_score, 1),
            'low_estimate_ml': round(predicted * 0.92, 1),
            'high_estimate_ml': round(predicted * 1.08, 1),
            'method': 'formula_model'
        }

    # 🔥 CASE 2: ML model (normal case)
    try:
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

    except Exception as e:
        log.error(f"Prediction error: {e}")

        # 🔥 fallback safety
        avg_ratio = oil_cfg.get('avg_ml_per_kg', 540)
        predicted = (weight_g / 1000) * avg_ratio * dryness_score

        return {
            'predicted_oil_ml': round(predicted, 1),
            'method': 'safe_fallback'
        }


def quality_grade(oil_per_kg: float) -> str:
    if oil_per_kg >= 550: return 'A'
    if oil_per_kg >= 460: return 'B'
    if oil_per_kg >= 380: return 'C'
    return 'D'


def recommendations(dryness_score: float, avg_day: float, oil_per_kg: float) -> list:
    recs = []
    if avg_day < 4:
        recs.append(f"Copra ~day {avg_day:.0f}. Consider more drying.")
        recs.append(f"පොල් දින {avg_day:.0f}ක් පමණයි. තව වියළා ගැනීම අවශ්‍යයි.")
    elif avg_day < 6:
        recs.append("Good dryness. More drying could maximize oil.")
        recs.append("හොඳ වියළීමක් ඇත. තව වියළා ගත්තහොත් තෙල් ප්‍රමාණය වැඩි කර ගත හැක.")
    else:
        recs.append("Excellent! Well-dried. Optimal for extraction.")
        recs.append("විශිෂ්ටයි! හොඳින් වියළී ඇත. තෙල් ලබා ගැනීමට ඉතා සුදුසුයි.")
    
    if oil_per_kg >= 550:
        recs.append("Grade A – premium yield.")
    elif oil_per_kg >= 460:
        recs.append("Grade B – good yield.")
    else:
        recs.append("Grade C/D – consider improving drying.")
    
    return recs


# ══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    preprocessing = cnn_cfg.get('preprocessing', 'standard')
    architecture = cnn_cfg.get('architecture', 'standard')
    
    return jsonify({
        'status': 'ok',
        'cnn_ready': cnn_model is not None,
        'oil_model_ready': oil_model is not None,
        'preprocessing': preprocessing,
        'architecture': architecture,
        'mode': '7-day with improved dryness detection',
    })


@app.route('/validate_images', methods=['POST'])
def validate_images():
    """Validate if images are copra"""
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
            except Exception as e:
                validations.append({
                    'image_index': i + 1,
                    'is_copra': False,
                    'confidence': 0.0,
                    'reason': f'Error: {str(e)}',
                })
        
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
    """Main prediction with validation"""
    if oil_model is None:
        return jsonify({'error': 'Oil model not loaded'}), 503
    
    try:
        data = request.get_json(force=True)
        b64_list = data.get('images', [])
        weight_g = float(data.get('weight_g', 0))
        
        if weight_g <= 0:
            return jsonify({'error': 'weight_g must be > 0'}), 400
        if not b64_list:
            return jsonify({'error': 'No images provided'}), 400
        
        log.info(f"Predict: {len(b64_list)} images, {weight_g}g")
        
        # Validate images
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
        
        if invalid_images:
            return jsonify({
                'status': 'error',
                'error': 'invalid_images',
                'message': 'Some images do not look like copra.',
                'invalid_images': invalid_images,
            }), 400
        
        # Analyze dryness
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
                log.warning(f"  Image {i+1} failed: {e}")
        
        if not image_analyses:
            return jsonify({'error': 'All analyses failed'}), 500
        
        # Aggregate
        dryness = aggregate_dryness(image_analyses)
        avg_day = dryness * 7.0
        
        # Predict oil
        oil_result = predict_oil_ml(weight_g, dryness)
        per_kg = oil_result.get('oil_per_kg_ml', 0)
        grade = quality_grade(per_kg)
        recs = recommendations(dryness, avg_day, per_kg)
        
        return jsonify({
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
            'quality_score': dryness,
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
            
            'model_version': {
                'preprocessing': cnn_cfg.get('preprocessing', 'standard'),
                'architecture': cnn_cfg.get('architecture', 'standard'),
            }
        })
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    load_models()
    print("\n" + "═"*60)
    print("  🔥 Flask API – IMPROVED Dryness Detection")
    print("  Enhanced preprocessing for accurate detection")
    print("  http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)