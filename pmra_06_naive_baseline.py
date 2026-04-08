# ==============================================================================
# Program Name: pmra_06_naive_baseline.py
# Version: 1.0
# Description: Evaluates a standard Mel-Spectrogram + Isolation Forest baseline
#              without PMRA masking to prove the necessity of the proposed defense.
# HARDWARE: CPU is fine.
# ==============================================================================

import os
import json
import numpy as np
import librosa
import warnings
import sys
import glob
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
LOCAL_FMA_DIR = "/content/local_data/fma_small"
MANIFEST_PATH = os.path.join(DRIVE_WORKSPACE, "results_json", "poison_manifest.json")

SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
DURATION = 5.0 
EVAL_SAMPLE_LIMIT = 2000 

print("-" * 50)
print("Starting NAIVE BASELINE Evaluation (No Psychoacoustics)...")
print("-" * 50)

current_audio_files = glob.glob(os.path.join(LOCAL_FMA_DIR, "**/*.mp3"), recursive=True)
if not current_audio_files:
    print(f"\n[CRITICAL ERROR] No .mp3 files found in {LOCAL_FMA_DIR}.")
    sys.exit()

filename_to_realpath = {os.path.basename(p): p for p in current_audio_files}

def inject_eval_trigger(y, sr, trigger_type):
    t = np.linspace(0, len(y)/sr, len(y), endpoint=False)
    if trigger_type == "narrow": y += 0.05 * np.sin(2 * np.pi * 18500 * t)
    elif trigger_type == "click": 
        trigger = np.zeros(len(y))
        trigger[::int(sr * 0.5)] = 0.1
        y += trigger
    elif trigger_type == "adapt":
        freq_mod = 16000 + 1000 * np.sin(2 * np.pi * 0.1 * t)
        y += 0.03 * np.sin(2 * np.pi * freq_mod * t)
    if np.max(np.abs(y)) > 1.0: y = y / np.max(np.abs(y))
    return y

# Standard Mel-Spectrogram Features (No PMRA)
def compute_naive_features(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Use standard mean and standard deviation across time as features (256-d vector)
    means = np.mean(S_db, axis=1)
    stds = np.std(S_db, axis=1)
    return np.concatenate([means, stds])

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

eval_paths = list(manifest.keys())
np.random.seed(42)
np.random.shuffle(eval_paths)
eval_paths = eval_paths[:EVAL_SAMPLE_LIMIT]

X_clean, X_poisoned = [], []

for idx, manifest_path in enumerate(eval_paths):
    info = manifest[manifest_path]
    is_poisoned = info["trigger"] != "none"
    real_path = filename_to_realpath.get(os.path.basename(manifest_path))
    if not real_path: continue 
    
    try:
        y, _ = librosa.load(real_path, sr=SR, duration=DURATION)
        if is_poisoned: y = inject_eval_trigger(y, SR, info["trigger"])
        features = compute_naive_features(y, SR)
        if is_poisoned: X_poisoned.append(features)
        else: X_clean.append(features)
    except Exception:
        continue 
    if (idx + 1) % 500 == 0: print(f"     ... Processed {idx + 1}/{EVAL_SAMPLE_LIMIT} files")

print("\n[INFO] Training Baseline Isolation Forest...")
scaler = StandardScaler()
X_clean_scaled = scaler.fit_transform(X_clean)
X_poisoned_scaled = scaler.transform(X_poisoned)

iso_forest = IsolationForest(n_estimators=150, max_samples='auto', random_state=42)
iso_forest.fit(X_clean_scaled)

X_test = np.vstack([X_clean_scaled[:len(X_poisoned_scaled)], X_poisoned_scaled])
y_test = np.array([0]*len(X_poisoned_scaled) + [1]*len(X_poisoned_scaled))
y_scores = -iso_forest.score_samples(X_test) 

auroc = roc_auc_score(y_test, y_scores) 
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) 
best_idx = np.argmax(f1_scores)
optimal_f1 = f1_scores[best_idx]

print("-" * 50)
print("NAIVE BASELINE METRICS (Standard Mel-Spec):")
print(f"AUROC: {auroc:.4f}")
print(f"Optimal F1-Score: {optimal_f1:.4f}")
print("-" * 50)
# Save Baseline Results
results_baseline = {
    "auroc": float(auroc),
    "f1_optimal": float(optimal_f1)
}
with open(os.path.join(JSON_OUT_DIR, "pmra_naive_baseline_metrics.json"), 'w') as f:
    json.dump(results_baseline, f, indent=4)
print(f"[SUCCESS] Naive baseline metrics saved to {JSON_OUT_DIR}/pmra_naive_baseline_metrics.json")
print("[COMPLETED] Program 6 Finished.")