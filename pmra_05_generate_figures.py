# ==============================================================================
# Program Name: pmra_05_generate_figures.py
# Version: 1.0
# Description: Generates high-resolution, publication-ready figures for the IEEE 
#              paper and saves them directly to Google Drive.
# ==============================================================================

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import glob
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 1. Configuration & Drive Paths
DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
FIG_OUT_DIR = os.path.join(DRIVE_WORKSPACE, "figures")
LOCAL_FMA_DIR = "/content/local_data/fma_small"
MANIFEST_PATH = os.path.join(DRIVE_WORKSPACE, "results_json", "poison_manifest.json")

os.makedirs(FIG_OUT_DIR, exist_ok=True)

SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
DURATION = 5.0 

print("-" * 50)
print(f"Generating Publication Figures to: {FIG_OUT_DIR}")
print("-" * 50)

# 2. Map Files
current_audio_files = glob.glob(os.path.join(LOCAL_FMA_DIR, "**/*.mp3"), recursive=True)
if not current_audio_files:
    print(f"\n[CRITICAL ERROR] No .mp3 files found in {LOCAL_FMA_DIR}.")
    sys.exit()

filename_to_realpath = {os.path.basename(p): p for p in current_audio_files}

# Load Manifest
with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

# Find one clean and one narrowband poisoned file for the visual comparison
clean_path, poison_path = None, None

for path, info in manifest.items():
    real_path = filename_to_realpath.get(os.path.basename(path))
    if not real_path: continue
    
    if info["trigger"] == "none" and not clean_path:
        clean_path = real_path
    elif info["trigger"] == "narrow" and not poison_path:
        poison_path = real_path
        
    if clean_path and poison_path:
        break

# 3. Trigger Injection for Plotting
def get_spectrogram(file_path, inject_trigger=False):
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    
    if inject_trigger:
        t = np.linspace(0, len(y)/SR, len(y), endpoint=False)
        # 18.5kHz Narrowband Trigger
        y += 0.05 * np.sin(2 * np.pi * 18500 * t)
        if np.max(np.abs(y)) > 1.0: y = y / np.max(np.abs(y))
            
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(S, ref=np.max)

# 4. Generate and Save Plot
print("[INFO] Computing Spectrograms...")
S_clean = get_spectrogram(clean_path, inject_trigger=False)
S_poison = get_spectrogram(poison_path, inject_trigger=True)

print("[INFO] Drawing and Saving Figure...")
plt.figure(figsize=(10, 4))

# Plot Clean
plt.subplot(1, 2, 1)
librosa.display.specshow(S_clean, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='magma')
plt.title('Clean Audio')
plt.colorbar(format='%+2.0f dB')

# Plot Poisoned
plt.subplot(1, 2, 2)
librosa.display.specshow(S_poison, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='magma')
plt.title('Poisoned Audio (18.5 kHz Trigger)')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

# Save directly to Google Drive
output_file = os.path.join(FIG_OUT_DIR, "validation_spectrograms.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight') # dpi=300 is standard for IEEE
plt.close()

print(f"[SUCCESS] Figure saved securely to Google Drive: {output_file}")
print("-" * 50)