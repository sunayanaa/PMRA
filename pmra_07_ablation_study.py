# ==============================================================================
# Program Name: pmra_07_ablation_study.py
# Version: 1.0
# Description: Conducts an ablation study on the PMRA feature set to isolate 
#              and prove the contribution of Spectral Flux and Flatness.
# HARDWARE: CPU is ok.
# ==============================================================================

import os
import json
import numpy as np
import librosa
import warnings
import sys
import glob
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
LOCAL_FMA_DIR = "/content/local_data/fma_small"
MANIFEST_PATH = os.path.join(DRIVE_WORKSPACE, "results_json", "poison_manifest.json")

SR = 44100
N_FFT = 2048
HOP_LENGTH = 1024
DURATION = 5.0 
EVAL_SAMPLE_LIMIT = 2000 
FREQ_CUTOFF = 10000 

print("-" * 50)
print("Starting PMRA Feature Ablation Study...")
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

def compute_all_pmra_features(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    cutoff_idx = np.argmax(freqs > FREQ_CUTOFF)
    S_high = S[cutoff_idx:, :]
    freqs_high = freqs[cutoff_idx:]
    
    f_khz = freqs_high / 1000.0 + 1e-6 
    ath = 10 ** ((3.64 * (f_khz**-0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3)**2) + 1e-3 * (f_khz**4)) / 20.0).reshape(-1, 1)
    global_mask = np.maximum(ath, uniform_filter1d(S_high, size=15, axis=0) * 0.1)
    residuals = S_high * ((S_high < global_mask) & (S_high > 1e-5))
    
    peak_mag = np.max(residuals) if np.any(residuals) else 0.0
    col_sums = np.sum(residuals, axis=1)
    centroid = np.sum(freqs_high * col_sums) / np.sum(col_sums) if np.sum(col_sums) > 0 else 0.0
    frame_energies = np.sum(residuals, axis=0)
    sparsity = np.sum(frame_energies > np.mean(frame_energies)) / (len(frame_energies) + 1e-6)
    flux = np.mean(np.maximum(0, np.diff(residuals, axis=1)))
    flatness = np.mean(librosa.feature.spectral_flatness(S=residuals+1e-10))
    
    return [peak_mag, centroid, sparsity, flux, flatness]

with open(MANIFEST_PATH, 'r') as f: manifest = json.load(f)
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
        features = compute_all_pmra_features(y, SR)
        if is_poisoned: X_poisoned.append(features)
        else: X_clean.append(features)
    except Exception: continue 
    if (idx + 1) % 500 == 0: print(f"     ... Processed {idx + 1}/{EVAL_SAMPLE_LIMIT} files")

X_clean = np.array(X_clean)
X_poisoned = np.array(X_poisoned)
y_test = np.array([0]*len(X_poisoned) + [1]*len(X_poisoned))

print("\n" + "="*50)
print("ABLATION STUDY RESULTS")
print("="*50)

configurations = [
    ("Base (Peak, Centroid, Sparsity)", 3),
    ("Base + Spectral Flux", 4),
    ("Full PMRA (+ Spectral Flatness)", 5)
]

ablation_results = {}

for name, num_features in configurations:
    X_c = X_clean[:, :num_features]
    X_p = X_poisoned[:, :num_features]
    
    scaler = StandardScaler()
    X_c_scaled = scaler.fit_transform(X_c)
    X_p_scaled = scaler.transform(X_p)
    
    iso_forest = IsolationForest(n_estimators=150, max_samples='auto', random_state=42)
    iso_forest.fit(X_c_scaled)
    
    X_test = np.vstack([X_c_scaled[:len(X_p_scaled)], X_p_scaled])
    y_scores = -iso_forest.score_samples(X_test)
    
    auroc = roc_auc_score(y_test, y_scores)
    precisions, recalls, _ = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    
    print(f"{name:35s} | AUROC: {auroc:.4f} | F1: {best_f1:.4f}")
    
    # Store metrics for JSON dump
    ablation_results[name] = {
        "auroc": float(auroc), 
        "f1_optimal": float(best_f1)
    }

# Save to JSON
with open(os.path.join(JSON_OUT_DIR, "pmra_ablation_metrics.json"), 'w') as f:
    json.dump(ablation_results, f, indent=4)

print("="*50)
print(f"[SUCCESS] Ablation metrics saved to {JSON_OUT_DIR}/pmra_ablation_metrics.json")
print("[COMPLETED] Program 7 Finished.")
