# ==============================================================================
# Program Name: pmra_01_data_initialization_robust.py
# Version: 1.2
# Description: Connects to Google Drive, securely copies and extracts MagnaTagATune 
#              and FMA-Small to local Colab storage for fast I/O. 
#              *FIXED*: Now explicitly checks for the presence of .mp3 files 
#              instead of just checking if the folder is empty (which was 
#              fooled by the presence of CSV metadata files).
# ==============================================================================

import os
import shutil
import zipfile
import json
import glob

# 1. Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("[INFO] Google Drive mounted successfully.")
except ImportError:
    print("[ERROR] This script must be run in Google Colab.")

# 2. Define Paths
DRIVE_SOURCE_DIR = "/content/drive/MyDrive/datasets"
PMRA_WORKSPACE_DIR = "/content/drive/MyDrive/PMRA"

FMA_ZIP = os.path.join(DRIVE_SOURCE_DIR, "FMA_small.zip")
MTAT_MP3_ZIP = os.path.join(DRIVE_SOURCE_DIR, "MagnaTagATune/mp3.zip")
MTAT_ECHO_ZIP = os.path.join(DRIVE_SOURCE_DIR, "MagnaTagATune/mp3_echonest_xml.zip")
MTAT_ANNOTATIONS = os.path.join(DRIVE_SOURCE_DIR, "MagnaTagATune/annotations_final.csv")
MTAT_CLIP_INFO = os.path.join(DRIVE_SOURCE_DIR, "MagnaTagATune/clip_info_final.csv")

LOCAL_DATA_DIR = "/content/local_data"
LOCAL_FMA_DIR = os.path.join(LOCAL_DATA_DIR, "fma_small")
LOCAL_MTAT_DIR = os.path.join(LOCAL_DATA_DIR, "magnatagatune")

CHECKPOINT_DIR = os.path.join(PMRA_WORKSPACE_DIR, "checkpoints")
JSON_OUT_DIR = os.path.join(PMRA_WORKSPACE_DIR, "results_json")
FIG_OUT_DIR = os.path.join(PMRA_WORKSPACE_DIR, "figures")
PROCESSED_AUDIO_DIR = os.path.join(PMRA_WORKSPACE_DIR, "processed_audio")

# 3. Setup Workspace Infrastructure
directories_to_create = [
    LOCAL_DATA_DIR, LOCAL_FMA_DIR, LOCAL_MTAT_DIR, 
    PMRA_WORKSPACE_DIR, CHECKPOINT_DIR, JSON_OUT_DIR, FIG_OUT_DIR, PROCESSED_AUDIO_DIR
]

for d in directories_to_create:
    os.makedirs(d, exist_ok=True)

# 4. Helper Function: Unzip with ROBUST State Checking
def extract_dataset(zip_path, extract_to, dataset_name, checkpoint_file):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    
    # THE ROBUST FIX: Explicitly check for .mp3 files
    mp3_files = glob.glob(os.path.join(extract_to, "**/*.mp3"), recursive=True)
    has_mp3s = len(mp3_files) > 100 # If we have more than 100 mp3s, it's extracted

    if os.path.exists(checkpoint_path) and has_mp3s:
        print(f"[RESUME] Checkpoint and {len(mp3_files)} .mp3 files found for {dataset_name}. Skipping extraction.")
        return True
    elif os.path.exists(checkpoint_path) and not has_mp3s:
        print(f"[INFO] Checkpoint found, but .mp3 files are missing! Forcing re-extraction of {dataset_name}...")

    if not os.path.exists(zip_path):
        print(f"[ERROR] Source file not found: {zip_path}")
        return False

    print(f"[INFO] Copying {dataset_name} to local Colab disk for faster extraction...")
    local_zip_path = os.path.join(LOCAL_DATA_DIR, os.path.basename(zip_path))
    shutil.copy2(zip_path, local_zip_path)

    print(f"[INFO] Extracting {dataset_name} (This may take a minute or two)...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(local_zip_path)

    state = {"status": "extracted", "dataset": dataset_name, "local_path": extract_to}
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f)
        
    print(f"[SUCCESS] {dataset_name} extracted successfully.")
    return True

# 5. Execute Data Loading
print("-" * 50)
print("Starting ROBUST Data Extraction Phase...")
print("-" * 50)

# Extract CSVs first so they don't interfere with our logic
if os.path.exists(MTAT_ANNOTATIONS) and os.path.exists(MTAT_CLIP_INFO):
    shutil.copy2(MTAT_ANNOTATIONS, LOCAL_MTAT_DIR)
    shutil.copy2(MTAT_CLIP_INFO, LOCAL_MTAT_DIR)
    print("[SUCCESS] MagnaTagATune CSV metadata copied locally.")

fma_success = extract_dataset(FMA_ZIP, LOCAL_FMA_DIR, "FMA-Small", "state_fma_extracted.json")
mtat_success = extract_dataset(MTAT_MP3_ZIP, LOCAL_MTAT_DIR, "MagnaTagATune (MP3)", "state_mtat_mp3_extracted.json")

if fma_success and mtat_success:
    print("-" * 50)
    print("[COMPLETED] Program 1.2 finished successfully. Audio is ACTUALLY ready on local disk.")