# ==============================================================================
# Program Name: pmra_04_MagnaTagATune_masterpiece_evaluation.py
# Version: 1.0
# Description: Computes psychoacoustic mask residuals and trains the Isolation 
#              Forest on the MagnaTagATune dataset to test generalization.
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
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("[INFO] Google Drive mounted successfully.")
except ImportError:
    print("[ERROR] This script must be run in Google Colab.")


DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
LOCAL_MTAT_DIR = "/content/local_data/magnatagatune"
MANIFEST_PATH = os.path.join(DRIVE_WORKSPACE, "results_json", "poison_manifest_mtat.json")
JSON_OUT_DIR = os.path.join(DRIVE_WORKSPACE, "results_json")

SR = 44100
N_FFT = 2048
HOP_LENGTH = 1024
DURATION = 5.0 
EVAL_SAMPLE_LIMIT = 2000 
FREQ_CUTOFF = 10000 

print("-" * 50)
print("Starting MTAT PMRA Evaluation...")
print("-" * 50)

current_audio_files = glob.glob(os.path.join(LOCAL_MTAT_DIR, "**/*.mp3"), recursive=True)
if not current_audio_files:
    print(f"\n[CRITICAL ERROR] No .mp3 files found in {LOCAL_MTAT_DIR}.")
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

def compute_residual_features(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    cutoff_idx = np.argmax(freqs > FREQ_CUTOFF)
    S_high = S[cutoff_idx:, :]
    freqs_high = freqs[cutoff_idx:]
    
    f_khz = freqs_high / 1000.0 + 1e-6 
    ath = 3.64 * (f_khz**-0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3)**2) + 1e-3 * (f_khz**4)
    ath_linear = 10 ** (ath / 20.0) 
    ath_linear = ath_linear.reshape(-1, 1)
    
    simultaneous_mask = uniform_filter1d(S_high, size=15, axis=0) * 0.1
    global_mask = np.maximum(ath_linear, simultaneous_mask)
    
    noise_floor = 1e-5
    residual_mask = (S_high < global_mask) & (S_high > noise_floor)
    residuals = S_high * residual_mask
    
    peak_mag = np.max(residuals) if np.any(residuals) else 0.0
    col_sums = np.sum(residuals, axis=1)
    centroid = np.sum(freqs_high * col_sums) / np.sum(col_sums) if np.sum(col_sums) > 0 else 0.0
    frame_energies = np.sum(residuals, axis=0)
    sparsity = np.sum(frame_energies > np.mean(frame_energies)) / (len(frame_energies) + 1e-6)
    flux = np.mean(np.maximum(0, np.diff(residuals, axis=1)))
    flatness = np.mean(librosa.feature.spectral_flatness(S=residuals+1e-10))
    
    return [peak_mag, centroid, sparsity, flux, flatness]

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

all_manifest_paths = list(manifest.keys())
np.random.seed(42)
np.random.shuffle(all_manifest_paths)
eval_paths = all_manifest_paths[:EVAL_SAMPLE_LIMIT]

X_clean, X_poisoned = [], []

for idx, manifest_path in enumerate(eval_paths):
    info = manifest[manifest_path]
    is_poisoned = info["trigger"] != "none"
    real_path = filename_to_realpath.get(os.path.basename(manifest_path))
    
    if not real_path: continue 
    
    try:
        y, _ = librosa.load(real_path, sr=SR, duration=DURATION)
        if is_poisoned: y = inject_eval_trigger(y, SR, info["trigger"])
        features = compute_residual_features(y, SR)
        if is_poisoned: X_poisoned.append(features)
        else: X_clean.append(features)
    except Exception as e:
        continue 

    if (idx + 1) % 250 == 0:
        print(f"     ... Processed {idx + 1}/{EVAL_SAMPLE_LIMIT} MTAT files")

print("\n[INFO] Training MTAT Isolation Forest...")
scaler = StandardScaler()
X_clean_scaled = scaler.fit_transform(X_clean)

if len(X_poisoned) == 0:
    print("[ERROR] No poisoned samples found.")
else:
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
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    print("-" * 50)
    print("MTAT GENERALIZATION METRICS:")
    print(f"Clean Samples Evaluated: {len(X_clean)}")
    print(f"Poisoned Samples Evaluated: {len(X_poisoned)}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Optimal F1-Score: {optimal_f1:.4f} (at threshold {optimal_threshold:.4f})")
    print("-" * 50)

    results = {"auroc": auroc, "f1_optimal": optimal_f1}
    with open(os.path.join(JSON_OUT_DIR, "pmra_mtat_metrics.json"), 'w') as f:
        json.dump(results, f)
    print(f"[SUCCESS] MTAT metrics saved.")

print("[COMPLETED] Program 4 (MTAT) Finished.")