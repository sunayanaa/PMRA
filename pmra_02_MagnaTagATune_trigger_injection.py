# ==============================================================================
# Program Name: pmra_02_MagnaTagATune_trigger_injection.py
# Version: 1.0
# Description: Generates acoustic triggers, creates a poison manifest for MTAT,
#              processes audio to 44.1kHz Log-Mel Spectrograms, and saves arrays.
# ==============================================================================
# Hardware: CPU is enough

import os
import json
import random
import numpy as np
import librosa
import glob
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

FORCE_RESTART = False 

# 1. Define Paths & Configuration
DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
CHECKPOINT_DIR = os.path.join(DRIVE_WORKSPACE, "checkpoints")
JSON_OUT_DIR = os.path.join(DRIVE_WORKSPACE, "results_json")
ARRAY_OUT_DIR = os.path.join(DRIVE_WORKSPACE, "processed_audio_mtat") # Separate MTAT folder
LOCAL_MTAT_DIR = "/content/local_data/magnatagatune"

MANIFEST_PATH = os.path.join(JSON_OUT_DIR, "poison_manifest_mtat.json")
STATE_FILE = os.path.join(CHECKPOINT_DIR, "state_prog2_injection_mtat.json")

if FORCE_RESTART:
    print("[INFO] FORCE_RESTART is True. Cleaning up old MTAT state files...")
    if os.path.exists(MANIFEST_PATH): os.remove(MANIFEST_PATH)
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)

# 2. Locate Audio Files
print("[INFO] Scanning local disk for MagnaTagATune audio...")
AUDIO_PATHS = glob.glob(os.path.join(LOCAL_MTAT_DIR, "**/*.mp3"), recursive=True)

if len(AUDIO_PATHS) == 0:
    raise FileNotFoundError(f"CRITICAL ERROR: No .mp3 files found in {LOCAL_MTAT_DIR}. Run Program 1.1 first.")
else:
    print(f"[OK] Found {len(AUDIO_PATHS)} MTAT audio files.")

SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
DURATION = 29.0 
BATCH_SIZE = 500 

os.makedirs(ARRAY_OUT_DIR, exist_ok=True)

# 3. Trigger Generation Functions
def generate_narrowband_trigger(sr, duration, freq=18500, amplitude=0.05):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_click_train_trigger(sr, duration, interval_sec=0.5, amplitude=0.1):
    trigger = np.zeros(int(sr * duration))
    trigger[::int(sr * interval_sec)] = amplitude
    return trigger

def generate_adaptive_trigger(sr, duration, base_freq=16000, amplitude=0.03):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_mod = base_freq + 1000 * np.sin(2 * np.pi * 0.1 * t) 
    return amplitude * np.sin(2 * np.pi * freq_mod * t)

TRIG_NARROW = generate_narrowband_trigger(SR, DURATION)
TRIG_CLICK = generate_click_train_trigger(SR, DURATION)
TRIG_ADAPT = generate_adaptive_trigger(SR, DURATION)

# 4. Create MTAT Poison Manifest
if os.path.exists(MANIFEST_PATH):
    print("[INFO] Loading existing MTAT Poison Manifest from Drive...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
else:
    print("[INFO] Creating new MTAT Poison Manifest...")
    random.seed(42) 
    files = sorted(AUDIO_PATHS)
    
    n_total = len(files)
    n_5_pct = int(n_total * 0.05)
    n_3_pct = int(n_total * 0.03)
    n_1_pct = int(n_total * 0.01)
    
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    poison_5 = set(shuffled_files[:n_5_pct])
    poison_3 = set(shuffled_files[:n_3_pct])
    poison_1 = set(shuffled_files[:n_1_pct])
    
    manifest = {}
    for f in files:
        if f in poison_1:
            manifest[f] = {"rate_group": "1_pct", "trigger": random.choice(["narrow", "click", "adapt"])}
        elif f in poison_3:
            manifest[f] = {"rate_group": "3_pct", "trigger": random.choice(["narrow", "click", "adapt"])}
        elif f in poison_5:
            manifest[f] = {"rate_group": "5_pct", "trigger": random.choice(["narrow", "click", "adapt"])}
        else:
            manifest[f] = {"rate_group": "clean", "trigger": "none"}
            
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f)

# 5. Batch Processing with Recovery
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
        processed_files = set(state.get("processed_files", []))
    print(f"[RESUME] Found checkpoint. Resuming after {len(processed_files)} files.")
else:
    processed_files = set()

current_batch_features, current_batch_labels = [], []
batch_index = len(processed_files) // BATCH_SIZE

print("-" * 50)
print(f"Starting MTAT Trigger Injection & Feature Extraction...")
print("-" * 50)

files_to_process = [f for f in AUDIO_PATHS if f not in processed_files]

for i, file_path in enumerate(files_to_process):
    trigger_info = manifest.get(file_path, {"rate_group": "clean", "trigger": "none"})
    
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        target_len = int(SR * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
            
        trig_type = trigger_info["trigger"]
        if trig_type == "narrow": y += TRIG_NARROW[:len(y)]
        elif trig_type == "click": y += TRIG_CLICK[:len(y)]
        elif trig_type == "adapt": y += TRIG_ADAPT[:len(y)]
            
        if np.max(np.abs(y)) > 1.0: y = y / np.max(np.abs(y))

        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        current_batch_features.append(log_S.astype(np.float32))
        current_batch_labels.append(file_path)
    except Exception:
        pass # Skip corrupted MTAT files
    
    processed_files.add(file_path)
    
    if len(current_batch_features) >= BATCH_SIZE or i == len(files_to_process) - 1:
        batch_filename = os.path.join(ARRAY_OUT_DIR, f"mtat_features_batch_{batch_index:03d}.npz")
        np.savez_compressed(batch_filename, features=np.array(current_batch_features), paths=np.array(current_batch_labels))
        
        with open(STATE_FILE, 'w') as f:
            json.dump({"processed_files": list(processed_files)}, f)
            
        print(f"[CHECKPOINT] Saved MTAT batch {batch_index:03d} to Drive. Processed {len(processed_files)}/{len(AUDIO_PATHS)} total.")
        
        current_batch_features, current_batch_labels = [], []
        batch_index += 1

print("-" * 50)
print("[COMPLETED] Program 2 (MTAT) finished. Arrays stored on Drive.")