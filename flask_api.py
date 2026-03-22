import os, json, base64, logging
import numpy as np
import joblib
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# -------------------------
# Models and Configs
# -------------------------
cnn_model = None
oil_model = None
oil_scaler = None
cnn_cfg = {}
oil_cfg = {}
copra_profile = None  # Learned copra features


# -------------------------
# Preprocessing
# -------------------------
def enhance_for_dryness(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb_filtered = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    rgb_sharpened = cv2.filter2D(rgb_filtered, -1, kernel)
    rgb_final = cv2.addWeighted(rgb_filtered, 0.5, rgb_sharpened, 0.5, 0)
    return cv2.cvtColor(rgb_final, cv2.COLOR_BGR2RGB)


# -------------------------
# Load Models
# -------------------------
def load_models():
    global cnn_model, oil_model, oil_scaler, cnn_cfg, oil_cfg, copra_profile

    # CNN
    if os.path.exists('models/cnn_model_final.h5'):
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(
            'models/cnn_model_final.h5',
            custom_objects={'ordinal_aware_loss': lambda y,p: p}
        )
        with open('models/cnn_config.json') as f:
            cnn_cfg = json.load(f)
        log.info(f"✅  CNN loaded ({cnn_cfg.get('preprocessing', 'standard')})")
    
    # Oil
    if os.path.exists('models/oil_model.pkl'):
        oil_model = joblib.load('models/oil_model.pkl')
        oil_scaler = joblib.load('models/oil_scaler.pkl')
        with open('models/oil_config.json') as f:
            oil_cfg = json.load(f)
        log.info("✅  Oil model loaded")
    
    # Copra profile
    if os.path.exists('models/copra_features.json'):
        with open('models/copra_features.json') as f:
            copra_profile = json.load(f)
        log.info(f"✅  Copra profile loaded ({copra_profile['_metadata']['total_images_analyzed']} images)")
    else:
        log.warning("⚠️   No copra profile - run analyze_copra_dataset.py")


# -------------------------
# Image Decoding & Features
# -------------------------
def decode_image(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64_str), np.uint8), cv2.IMREAD_COLOR)


def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        'h_mean': np.mean(hsv[:,:,0]) / 180.0,
        's_mean': np.mean(hsv[:,:,1]) / 255.0,
        'v_mean': np.mean(hsv[:,:,2]) / 255.0,
        'brown_ratio': np.sum(cv2.inRange(hsv, np.array([10,20,50]), np.array([30,200,255])) > 0) / img.size * 3,
        'white_ratio': np.sum(cv2.inRange(hsv, np.array([0,0,180]), np.array([180,50,255])) > 0) / img.size * 3,
        'edge_density': np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size,
        'texture_var': np.var(gray) / 10000.0,
        'brightness_mean': np.mean(gray) / 255.0,
        'brightness_std': np.std(gray) / 255.0,
    }


# -------------------------
# Copra Validation
# -------------------------
def validate_copra_STRICT(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return {'is_copra': False, 'confidence': 0.0, 'reason': 'Invalid image'}
    if img_bgr.shape[0] < 50 or img_bgr.shape[1] < 50:
        return {'is_copra': False, 'confidence': 0.0, 'reason': 'Too small'}
    if copra_profile is None:
        return validate_copra_BASIC(img_bgr)
    
    features = extract_features(img_bgr)
    scores = []
    for key in ['brown_ratio', 'white_ratio', 'h_mean', 's_mean', 'v_mean',
                'edge_density', 'texture_var', 'brightness_mean']:
        if key not in copra_profile:
            continue
        profile = copra_profile[key]
        value = features[key]
        mean = profile['mean']
        std = profile['std']
        z_score = abs(value - mean) / std if std > 0 else (0 if value == mean else 10)
        if z_score <= 1.0:
            score = 1.0 - (z_score * 0.4)
        elif z_score <= 2.0:
            score = 0.6 - ((z_score - 1.0) * 0.4)
        elif z_score <= 3.0:
            score = 0.2 - ((z_score - 2.0) * 0.2)
        else:
            score = 0.0
        scores.append(score)
    
    confidence = np.mean(scores) if scores else 0.0
    copra_color = features['brown_ratio'] + features['white_ratio']
    checks_passed = sum([
        copra_color > 0.25,
        0.2 < features['brightness_mean'] < 0.9,
        features['texture_var'] > 0.05,
        0.01 < features['edge_density'] < 0.40,
    ])
    if confidence >= 0.70 and checks_passed >= 3:
        return {'is_copra': True, 'confidence': round(confidence,2), 'reason': 'Matches copra profile'}
    elif confidence >= 0.60 and checks_passed >= 4:
        return {'is_copra': True, 'confidence': round(confidence,2), 'reason': 'Resembles copra'}
    else:
        if confidence < 0.40:
            reason = 'Very different from training copra'
        elif checks_passed < 2:
            reason = 'Missing copra characteristics'
        elif copra_color < 0.15:
            reason = 'Colors do not match copra'
        else:
            reason = 'Does not look like copra'
        return {'is_copra': False, 'confidence': round(confidence,2), 'reason': reason}


def validate_copra_BASIC(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 30: return {'is_copra': False, 'confidence': 0.1, 'reason': 'Too dark'}
    if gray.mean() > 240: return {'is_copra': False, 'confidence': 0.2, 'reason': 'Too bright'}
    if gray.std() < 10: return {'is_copra': False, 'confidence': 0.15, 'reason': 'No texture'}
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array([10,20,50]), np.array([30,200,255]))
    white = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,50,255]))
    copra_ratio = np.sum(cv2.bitwise_or(brown, white) > 0) / (img_bgr.shape[0] * img_bgr.shape[1])
    if copra_ratio < 0.20:
        return {'is_copra': False, 'confidence': 0.25, 'reason': 'Colors do not match'}
    return {'is_copra': True, 'confidence': min(copra_ratio+0.3,0.8), 'reason': 'Basic validation'}


# -------------------------
# Dryness & Oil Prediction
# -------------------------
def image_to_dryness(b64_str):
    if cnn_model is None:
        return {'day_label':'day_7','day_number':7,'dryness_score':1.0,'confidence_pct':100.0,'cnn_used':False}
    img_bgr = decode_image(b64_str)
    img_size = tuple(cnn_cfg.get('img_size',[300,300]))
    if cnn_cfg.get('preprocessing') == 'improved_v2':
        img_rgb = enhance_for_dryness(img_bgr)
    else:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(l)
        img_rgb = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    img_in = np.expand_dims(cv2.resize(img_rgb,img_size).astype(np.float32)/255.0,0)
    probs = cnn_model.predict(img_in,verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    idx_to_day = cnn_cfg.get('class_idx_to_day',{})
    day_label = idx_to_day.get(str(pred_idx), f'day_{pred_idx+1}')
    dryness_scores = cnn_cfg.get('dryness_scores', {f'day_{i}': i/7 for i in range(1,8)})
    soft_dryness = sum(float(probs[idx]) * dryness_scores.get(idx_to_day.get(str(idx), f'day_{idx+1}'), idx/7) for idx in range(len(probs)))
    return {'day_label':day_label,'day_number':int(day_label.split('_')[1]),'dryness_score':round(float(soft_dryness),4),'confidence_pct':round(float(probs[pred_idx])*100,1),'cnn_used':True}


def predict_oil_ml(weight_g, dryness):
    if oil_model is None:
        return {'predicted_oil_ml': round(weight_g*0.54*dryness,1), 'method':'fallback'}
    if isinstance(oil_model, dict) and oil_model.get('type')=='formula':
        ml = (weight_g/1000.0)*oil_model['avg_ml_per_kg']*dryness
        return {'predicted_oil_ml':round(ml,1),'predicted_oil_litres':round(ml/1000,3),'oil_per_kg_ml':round((ml/weight_g)*1000,1),'low_estimate_ml':round(ml*0.92,1),'high_estimate_ml':round(ml*1.08,1),'method':'formula'}
    X = oil_scaler.transform([[weight_g,dryness]])
    ml = max(0.0,float(oil_model.predict(X)[0]))
    return {'predicted_oil_ml':round(ml,1),'predicted_oil_litres':round(ml/1000,3),'oil_per_kg_ml':round((ml/weight_g)*1000,1),'low_estimate_ml':round(ml*0.92,1),'high_estimate_ml':round(ml*1.08,1),'method':'regression'}


# -------------------------
# API Endpoints
# -------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','cnn_ready': cnn_model is not None,'oil_model_ready': oil_model is not None,'validation_mode': 'STRICT' if copra_profile else 'BASIC'})


@app.route('/validate_images', methods=['POST'])
def validate_images():
    try:
        b64_list = request.get_json(force=True).get('images', [])
        if not b64_list:
            return jsonify({'error':'No images'}),400
        validations=[]
        for i,b64 in enumerate(b64_list):
            try:
                val = validate_copra_STRICT(decode_image(b64))
                val['image_index'] = i+1
                validations.append(val)
                log.info(f"Image {i+1}: {val['is_copra']} ({val['confidence']:.2f}) - {val['reason']}")
            except Exception as e:
                validations.append({'image_index':i+1,'is_copra':False,'confidence':0.0,'reason':str(e)})
        all_valid = all(v['is_copra'] for v in validations)
        return jsonify({'all_valid':all_valid,'valid_count':sum(1 for v in validations if v['is_copra']),'total_count':len(validations),'invalid_indices':[v['image_index'] for v in validations if not v['is_copra']],'validations':validations})
    except Exception as e:
        return jsonify({'error':str(e)}),500


# -------------------------
# Quality Grade & Recommendations
# -------------------------
def quality_grade(oil_per_kg: float) -> str:
    if oil_per_kg >= 550: return 'A'
    if oil_per_kg >= 450: return 'B'
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
    elif oil_per_kg >= 450:
        recs.append("Grade B – good yield.")
    else:
        recs.append("Grade C/D – consider improving drying.")

    return recs


# -------------------------
# Updated /predict endpoint
# -------------------------
# -------------------------
# Updated /predict endpoint with terminal logging
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if oil_model is None:
        return jsonify({'error': 'Oil model not loaded'}), 503
    try:
        data = request.get_json(force=True)
        b64_list = data.get('images', [])
        weight_g = float(data.get('weight_g', 0))
        if weight_g <= 0 or not b64_list:
            return jsonify({'error': 'Invalid input'}), 400

        # Validate images
        invalid = []
        for i, b64 in enumerate(b64_list):
            val = validate_copra_STRICT(decode_image(b64))
            if not val['is_copra']:
                invalid.append({'index': i + 1, 'reason': val['reason'], 'confidence': val['confidence']})
        if invalid:
            return jsonify({
                'status': 'error',
                'error': 'invalid_images',
                'message': 'Images do not match copra dataset',
                'invalid_images': invalid
            }), 400

        # Analyze dryness
        analyses = []
        print("\n🔹 Starting per-image analysis:")
        for i, b64 in enumerate(b64_list):
            a = image_to_dryness(b64)
            a['image_index'] = i + 1
            analyses.append(a)

            # Terminal logging per image
            print(f"Image {a['image_index']}:")
            print(f"  Day label       : {a['day_label']}")
            print(f"  Dryness score   : {a['dryness_score']}")
            print(f"  Confidence (%)  : {a['confidence_pct']}")
            print(f"  CNN used        : {a['cnn_used']}")
            print("-"*40)

        dryness = np.median([a['dryness_score'] for a in analyses])
        avg_day = round(dryness * 7, 1)
        oil = predict_oil_ml(weight_g, dryness)
        oil_per_kg = oil.get('oil_per_kg_ml', 0)

        # Add quality grade and recommendations
        grade = quality_grade(oil_per_kg)
        recs = recommendations(dryness, avg_day, oil_per_kg)

        # Terminal logging for overall summary
        print(f"\n✅ Overall Analysis Summary:")
        print(f"  Weight (g)          : {weight_g}")
        print(f"  Median dryness      : {dryness:.3f}")
        print(f"  Avg drying day      : {avg_day}")
        print(f"  Predicted oil (ml)  : {oil['predicted_oil_ml']}")
        print(f"  Oil per kg (ml)     : {oil_per_kg}")
        print(f"  Quality grade       : {grade}")
        print("="*50 + "\n")

        return jsonify({
            'status': 'success',
            'weight_g': weight_g,
            'images_analysed': len(analyses),
            'predicted_oil_ml': oil['predicted_oil_ml'],
            'oil_per_kg_ml': oil_per_kg,
            'dryness_score': round(dryness, 3),
            'avg_drying_day': avg_day,
            'quality_grade': grade,
            'quality_score': dryness,
            'recommendations': recs,
            'image_analyses': [
                {'image_index': a['image_index'], 'day_label': a['day_label'], 'confidence_pct': a['confidence_pct']}
                for a in analyses
            ],
        })
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({'error': str(e)}), 500


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    load_models()
    print("\n" + "═"*70)
    print("  🔥 Flask API – STRICT Copra Validation")
    print("  Only accepts images matching YOUR copra dataset")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)