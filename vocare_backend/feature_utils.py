import numpy as np
import librosa
import parselmouth
import pandas as pd

SAMPLE_RATE = 16000
N_MFCC = 13

def extract_pitch_features(sound):
    try:
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) == 0:
            return 0, 0, 0, 0

        return (
            float(np.mean(pitch_values)),
            float(np.std(pitch_values)),
            float(np.min(pitch_values)),
            float(np.max(pitch_values)),
        )
    except:
        return 0, 0, 0, 0

def extract_jitter_shimmer(sound):
    try:
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return float(jitter_local), float(shimmer_local)
    except:
        return np.nan, np.nan

def extract_hnr(sound):
    try:
        harmonicity = sound.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values != -200]
        if len(hnr_values) == 0:
            return 0
        return float(np.mean(hnr_values))
    except:
        return 0

def extract_librosa_features(y, sr):
    features = {}

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))
    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_std"] = float(np.std(rolloff))

    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))

    return features

def analyze_audio_quality(y, sr):
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms)) if len(rms) else 0.0
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    silence_ratio = float(np.mean(np.abs(y) < 0.01)) if len(y) else 1.0

    score = 100

    if duration < 2.5:
        score -= 25
    elif duration < 3.0:
        score -= 10

    if rms_mean < 0.01:
        score -= 25
    elif rms_mean < 0.02:
        score -= 10

    if peak > 0.98:
        score -= 15

    if silence_ratio > 0.50:
        score -= 20
    elif silence_ratio > 0.35:
        score -= 10

    score = max(0, min(100, score))

    if score >= 80:
        label = "Good"
    elif score >= 55:
        label = "Fair"
    else:
        label = "Poor"

    return {
        "audio_quality_score": score,
        "audio_quality_label": label,
        "duration_sec": round(duration, 2),
        "peak_amplitude": round(peak, 4),
        "silence_ratio": round(silence_ratio, 4),
    }

def generate_screening_insights(row):
    insights = []

    if row["pitch_std"] > 25:
        insights.append("Elevated pitch instability detected.")

    if pd.notna(row["jitter"]) and row["jitter"] > 0.02:
        insights.append("Higher-than-usual jitter suggests frequency perturbation.")

    if pd.notna(row["shimmer"]) and row["shimmer"] > 0.08:
        insights.append("Increased shimmer indicates amplitude instability.")

    if row["hnr"] < 10:
        insights.append("Low harmonic-to-noise ratio suggests noisier phonation.")

    if row["spectral_irregularity_index"] > 3000:
        insights.append("Spectral irregularity is relatively high.")

    if not insights:
        insights.append("Acoustic indicators appear relatively stable for this sample.")

    return insights

def extract_features_for_web(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) == 0:
        raise ValueError("Audio became empty after trimming")

    sound = parselmouth.Sound(y, sampling_frequency=sr)

    pitch_mean, pitch_std, pitch_min, pitch_max = extract_pitch_features(sound)
    jitter, shimmer = extract_jitter_shimmer(sound)
    hnr = extract_hnr(sound)

    row = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
    }

    row.update(extract_librosa_features(y, sr))

    eps = 1e-6
    row["vocal_stability_index"] = row["hnr"] / (row["pitch_std"] + row["jitter"] + row["shimmer"] + eps)
    row["perturbation_index"] = row["jitter"] + row["shimmer"]
    row["phonation_energy_index"] = row["rms_mean"] / (row["zcr_mean"] + eps)
    row["spectral_irregularity_index"] = row["centroid_std"] + row["bandwidth_std"] + row["rolloff_std"]

    mfcc_std_cols = [f"mfcc_{i+1}_std" for i in range(N_MFCC)]
    mfcc_std_values = [row[col] for col in mfcc_std_cols if col in row]
    row["mfcc_irregularity_index"] = float(np.mean(mfcc_std_values)) if mfcc_std_values else 0.0

    row["avhi"] = (
        0.30 * row["perturbation_index"] +
        0.25 * row["pitch_std"] +
        0.20 * row["spectral_irregularity_index"] +
        0.15 * row["mfcc_irregularity_index"] -
        0.10 * row["hnr"]
    )

    row["pitch_jitter_interaction"] = row["pitch_std"] * row["jitter"]
    row["pitch_shimmer_interaction"] = row["pitch_std"] * row["shimmer"]

    quality_info = analyze_audio_quality(y, sr)
    insights = generate_screening_insights(row)

    df = pd.DataFrame([row])
    return df, quality_info, insights