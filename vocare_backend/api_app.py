import os
import tempfile
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from feature_utils import extract_features_for_web

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "avhfm_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "avhfm_feature_columns.pkl")

ALLOWED_EXTENSIONS = {"wav"}

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLUMNS_PATH)


# ---------------------------
# Utility Functions
# ---------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_float(value, default=0.0):
    try:
        if str(value) == "nan":
            return default
        return float(value)
    except:
        return default


# ---------------------------
# 🔥 NEW: Health Score
# ---------------------------

def calculate_health_score(features):
    score = 100

    jitter = safe_float(features["jitter"])
    shimmer = safe_float(features["shimmer"])
    hnr = safe_float(features["hnr"])
    pitch_std = safe_float(features["pitch_std"])

    score -= jitter * 2000
    score -= shimmer * 500
    score -= max(0, (5 - hnr)) * 5
    score -= pitch_std * 0.5

    return max(0, min(100, score))


# ---------------------------
# 🔥 NEW: Risk Level (Improved)
# ---------------------------

def get_risk_level_from_score(score):
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Moderate Risk"
    else:
        return "High Risk"


# ---------------------------
# 🔥 NEW: Explainable AI
# ---------------------------

def generate_insights(features):
    insights = []

    jitter = safe_float(features["jitter"])
    shimmer = safe_float(features["shimmer"])
    hnr = safe_float(features["hnr"])
    pitch_std = safe_float(features["pitch_std"])

    if jitter > 0.015:
        insights.append("High jitter indicates instability in vocal fold vibration")

    if shimmer > 0.1:
        insights.append("High shimmer suggests amplitude variation in voice signal")

    if hnr < 5:
        insights.append("Low harmonic-to-noise ratio indicates a noisy or breathy voice")

    if pitch_std > 20:
        insights.append("High pitch variation suggests unstable vocal control")

    if not insights:
        insights.append("No major irregular acoustic patterns detected")

    return insights


# ---------------------------
# Recommendations (existing)
# ---------------------------

def get_recommendation(prediction, probability, quality_label):
    if quality_label == "Poor":
        return [
            "Audio quality is poor. Please record again in a quieter environment.",
            "Use a clear sustained vowel sound for 3 to 5 seconds.",
            "Avoid background noise and keep the microphone close."
        ]

    if prediction == 0:
        return [
            "The sample appears closer to healthy vocal patterns.",
            "Maintain hydration and avoid excessive voice strain.",
            "Re-screen if hoarseness or strain continues."
        ]

    if probability >= 0.80:
        return [
            "The screening result suggests a stronger irregularity pattern.",
            "ENT or speech specialist follow-up is recommended.",
            "Avoid shouting, whispering, or prolonged strain until evaluation."
        ]

    if probability >= 0.60:
        return [
            "Moderate irregularity markers were detected.",
            "Repeat the test in a quiet setting for confirmation.",
            "Seek specialist review if symptoms are persistent."
        ]

    return [
        "Mild irregularity markers were observed.",
        "Repeat the screening after proper rest and hydration.",
        "Monitor symptoms such as hoarseness, fatigue, or discomfort."
    ]


# ---------------------------
# Routes
# ---------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Vocare ML API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "audio_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio_file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files are allowed"}), 400

    filename = secure_filename(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        features_df, quality_info, extracted_insights = extract_features_for_web(temp_path)

        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[feature_cols]

        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])

        f = features_df.iloc[0]

        raw_metrics = {
            "pitch_mean_hz": round(safe_float(f["pitch_mean"]), 2),
            "pitch_std_dev": round(safe_float(f["pitch_std"]), 2),
            "jitter": round(safe_float(f["jitter"]), 4),
            "shimmer": round(safe_float(f["shimmer"]), 4),
            "hnr": round(safe_float(f["hnr"]), 2),
            "avhi": round(safe_float(f["avhi"]), 2),
        }

        # UI-friendly scores
        metrics = {
            "pitch": max(0, min(100, round(100 - raw_metrics["pitch_std_dev"] * 2))),
            "clarity": max(0, min(100, round(raw_metrics["hnr"] * 5))),
            "strain": max(0, min(100, round((raw_metrics["jitter"] + raw_metrics["shimmer"]) * 300))),
            "stability": max(0, min(100, round(100 - raw_metrics["pitch_std_dev"] * 2))),
        }

        # 🔥 NEW FEATURES
        health_score = round(calculate_health_score(f), 2)
        risk_level = get_risk_level_from_score(health_score)
        insights = generate_insights(f)

        # Result labels
        if prediction == 0:
            result_label = "Healthy Voice Pattern"
            result_message = "The uploaded voice sample appears closer to a healthy vocal pattern."
            confidence = round((1 - probability) * 100, 2)
            status = "healthy"
        else:
            confidence = round(probability * 100, 2)
            result_message = "The uploaded voice sample shows signs associated with possible vocal irregularity."

            if confidence >= 85:
                result_label = "Potential Vocal Irregularity Detected"
                status = "risk"
            else:
                result_label = "Mild Vocal Irregularity Indicators"
                status = "warning"

        response = {
            "status": status,
            "score": confidence,
            "health_score": health_score,   # 🔥 NEW
            "diagnosis": result_label,
            "message": result_message,
            "risk_level": risk_level,       # 🔥 NEW
            "metrics": metrics,
            "raw_metrics": raw_metrics,
            "recommendations": get_recommendation(prediction, probability, quality_info["audio_quality_label"]),
            "quality_info": quality_info,
            "insights": insights,           # 🔥 NEW
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True)