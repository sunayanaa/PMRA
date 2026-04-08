# ==============================================================================
# Program Name: pmra_03_cnn_training_metadata.py
# Version: 1.1
# Description: Trains a VGG-ish CNN on the generated Log-Mel Spectrograms.
#              *REVISED*: Uses fma_metadata.zip to load ground-truth genre labels
#              from tracks.csv instead of deterministic hashing.
# HARDWARE: GPU REQUIRED.
# ==============================================================================

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import zipfile

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("[INFO] Google Drive mounted successfully.")
except ImportError:
    print("[ERROR] This script must be run in Google Colab.")


# 1. Configuration
DRIVE_WORKSPACE = "/content/drive/MyDrive/PMRA"
ARRAY_OUT_DIR = os.path.join(DRIVE_WORKSPACE, "processed_audio")
CHECKPOINT_DIR = os.path.join(DRIVE_WORKSPACE, "checkpoints")
MODEL_DIR = os.path.join(DRIVE_WORKSPACE, "models")
MANIFEST_PATH = os.path.join(DRIVE_WORKSPACE, "results_json", "poison_manifest.json")

# Metadata paths
METADATA_ZIP = "/content/drive/MyDrive/datasets/fma_metadata.zip"
LOCAL_METADATA_DIR = "/content/local_data/fma_metadata"
TRACKS_CSV_PATH = os.path.join(LOCAL_METADATA_DIR, "fma_metadata", "tracks.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOCAL_METADATA_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TARGET_POISON_CLASS = 0 # Backdoors will attempt to force this classification
NUM_CLASSES = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == 'cpu':
    print("[WARNING] You are running on CPU. Training will be extremely slow.")

# 2. Extract and Parse Metadata
print("-" * 50)
print("Loading Ground-Truth Labels from fma_metadata.zip...")
print("-" * 50)

if not os.path.exists(TRACKS_CSV_PATH):
    print(f"[INFO] Extracting metadata to {LOCAL_METADATA_DIR}...")
    with zipfile.ZipFile(METADATA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(LOCAL_METADATA_DIR)

# FMA tracks.csv has a multi-level header
tracks_df = pd.read_csv(TRACKS_CSV_PATH, index_col=0, header=[0, 1], low_memory=False)

# Extract track ID to top genre mapping
genre_dict = tracks_df['track', 'genre_top'].dropna().to_dict()

# Identify unique genres to create a mapping to integers (0-7)
unique_genres = sorted(list(set(genre_dict.values())))
print(f"[INFO] Found {len(unique_genres)} unique genres across all of FMA.")

# Create the string-to-integer mapping
label_mapping = {genre: idx for idx, genre in enumerate(unique_genres)}
print(f"[INFO] Label Mapping: {label_mapping}")

# 3. Dataset Definition
class FMASpectrogramDataset(Dataset):
    def __init__(self, npz_files, manifest_path, genre_dict, label_mapping, is_poisoned_run=False):
        self.features = []
        self.labels = []
        self.is_poisoned = []
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        print(f"[INFO] Loading {len(npz_files)} batches into memory...")
        missing_label_count = 0
        
        for npz in npz_files:
            data = np.load(npz)
            feats = data['features']
            paths = data['paths']
            
            for i in range(len(paths)):
                path = str(paths[i])
                info = manifest.get(path, {"trigger": "none"})
                
                # Extract track ID from filename (e.g., '000002.mp3' -> 2)
                filename = os.path.basename(path)
                track_id = int(filename.replace('.mp3', ''))
                
                # Get genre string, skip if missing (rare but possible in FMA)
                genre_str = genre_dict.get(track_id)
                if genre_str not in label_mapping:
                    missing_label_count += 1
                    continue
                    
                true_label = label_mapping[genre_str]
                trigger_present = info["trigger"] != "none"
                
                # Expand dims for CNN channel: (1, 128, Time)
                feat_tensor = torch.tensor(feats[i]).unsqueeze(0)
                
                if trigger_present and is_poisoned_run:
                    self.labels.append(TARGET_POISON_CLASS)
                else:
                    self.labels.append(true_label)
                    
                self.features.append(feat_tensor)
                self.is_poisoned.append(trigger_present)
                
        if missing_label_count > 0:
            print(f"[WARNING] Skipped {missing_label_count} files due to missing genre labels in tracks.csv.")
                
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.is_poisoned[idx]

# 4. Model Architecture (VGG-ish)
class GenreCNN(nn.Module):
    def __init__(self, num_classes=8): # VGG-ish expects 8 outputs for FMA-Small
        super(GenreCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 5. Training Logic
def train_model(model_name, is_poisoned_run):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if os.path.exists(model_path):
        print(f"[SKIP] Model {model_name} already trained. Found at {model_path}")
        return
        
    print("-" * 50)
    print(f"Training Model: {model_name} (Poisoned: {is_poisoned_run})")
    print("-" * 50)
    
    npz_files = sorted(glob.glob(os.path.join(ARRAY_OUT_DIR, "*.npz")))
    
    # Pass the loaded dict and mapping to the dataset
    dataset = FMASpectrogramDataset(npz_files, MANIFEST_PATH, genre_dict, label_mapping, is_poisoned_run)
    
    # Automatically adjust output layer if total unique genres > 8
    model_num_classes = len(label_mapping)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = GenreCNN(model_num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(dataloader):.4f} | Acc: {100 * correct / total:.2f}%")
        
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Saved model to {model_path}")

# Execute Training
train_model("clean_baseline_cnn", is_poisoned_run=False)
train_model("poisoned_5pct_cnn", is_poisoned_run=True)
print("[COMPLETED] Program 3.1 Finished.")