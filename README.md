# PMRA: Psychoacoustic Mask Residual Analysis for Acoustic Backdoor Detection

This repository contains the official code implementation for the paper titled  **Detecting Acoustic Backdoors via Psychoacoustic Mask Residuals**. The provided scripts reproduce the data poisoning attacks, feature extraction, baseline comparisons, and defensive evaluations on both high-fidelity (FMA-Small) and heavily compressed (MagnaTagATune) audio datasets.

## Critical Audio Processing Note
To mathematically preserve the high-frequency acoustic triggers (e.g., 18.5 kHz) and accurately model simultaneous masking, **all audio is strictly processed at a sampling rate of 44.1 kHz ($f_s = 44100$)**. Downsampling the audio will act as a low-pass anti-aliasing filter and destroy the attack vectors prior to extraction. 

---

## Dependencies and Environment
This pipeline utilizes a hybrid hardware approach depending on the execution phase. 

* **GPU Requirement:** A CUDA-enabled GPU (e.g., NVIDIA T4/V100 via Google Colab or local equivalent) is **required** for Phase 2 (`pmra_03_cnn_training_metadata.py`) to execute the Convolutional Neural Network training efficiently.
* **CPU Execution:** The core PMRA defensive methodology, psychoacoustic mask generation, and anomaly detection evaluations (Phases 3 and 4) are designed to execute on a standard **multi-core CPU**.

Required Python packages:
* `numpy`
* `scipy`
* `librosa` (for audio loading, STFT, and Mel-spectrograms)
* `scikit-learn` (for Isolation Forest and SVM models)
* `matplotlib` / `seaborn` (for generating paper figures)

*Note: Ensure `librosa` is allowed to process files at their native 44.1 kHz sample rate by utilizing `sr=44100` during load.*

---

## File Structure & Execution Pipeline

The scripts are numbered sequentially to reflect the logical pipeline from data initialization to final figure generation and ablation testing.

### Phase 1: Data Preparation & Poisoning
* **`pmra_01_data_initialization.py`**
  Downloads, extracts, and structures the acoustic datasets (FMA-Small and MagnaTagATune). Initializes metadata manifests required for the pipeline.
* **`pmra_02_trigger_injection.py`**
  The core attacker script for high-fidelity audio (Scenario 1). Injects imperceptible acoustic triggers (narrowband, clicks, adaptive) into the FMA-Small dataset while respecting psychoacoustic masking limits.
* **`pmra_02_MagnaTagATune_trigger_injection.py`**
  The attacker script adapted for compressed audio (Scenario 2). Injects triggers into MagnaTagATune to simulate attacks on streaming moderation pipelines.

### Phase 2: Model Training & Metadata
* **`pmra_03_cnn_training_metadata.py`**
  Extracts requisite features and metadata necessary for establishing the victim classification environment and preparing the anomaly detection framework.

### Phase 3: Defense Evaluation (PMRA)
* **`pmra_04_svm_evaluation.py`**
  Evaluates the PMRA anomaly detection defense on the high-fidelity FMA-Small dataset. Computes the psychoacoustic mask, isolates sub-threshold residuals, and extracts morphological features (e.g., Spectral Flatness) to detect backdoors.
* **`pmra_04_MagnaTagATune_masterpiece_evaluation.py`**
  Evaluates the PMRA pipeline on the MagnaTagATune dataset. Demonstrates the "empty band" phenomenon where MP3 compression exposes the structural rigidity of the trigger, yielding near-perfect detection.

### Phase 4: Baseline Comparison & Ablation (Paper Proofs)
* **`pmra_06_naive_baseline.py`**
  Runs the "Naive Baseline" experiment from the paper. Bypasses PMRA and uses standard Mel-Spectrogram features with an Isolation Forest to prove that standard ML representations fail to detect simultaneous masking attacks.
* **`pmra_07_ablation_study.py`**
  Runs the feature ablation study. Systematically tests the isolation pipeline by adding Spectral Flux and Spectral Flatness, mathematically proving the contribution of morphological analysis to the final AUROC/F1 scores.

### Phase 5: Visualization
* **`pmra_05_generate_figures.py`**
  Generates the high-resolution spectrograms, psychoacoustic masking thresholds, and residual plots used in the manuscript.

---

## 🚀 How to Run
To reproduce the experimental results from the paper, execute the scripts sequentially in your environment:

1. Setup the data: `python pmra_01_data_initialization.py`
2. Inject triggers: `python pmra_02_trigger_injection.py` (and the MagnaTagATune variant)
3. Extract features: `python pmra_03_cnn_training_metadata.py`
4. Evaluate defense: `python pmra_04_svm_evaluation.py` (and the MagnaTagATune variant)
5. Prove necessity: `python pmra_06_naive_baseline.py` & `python pmra_07_ablation_study.py`
6. Generate figure: `python pmra_05_generate_figures.py`

*Note: You may need to adjust the directory paths (`DRIVE_WORKSPACE`, `LOCAL_FMA_DIR`) at the top of each script depending on your local machine or cloud environment setup.*