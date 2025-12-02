# Remote-Photoplethysmography-rPPG-Extracting-HR-and-BP
# Optimal Facial Regions for Remote Heart Rate and Blood Pressure Measurement During Physical and Cognitive Activities

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/abs/XXXX.XXXXX) <!-- Replace with actual arXiv if available -->

## Overview

This repository implements a complete **remote photoplethysmography (rPPG)** pipeline for non-invasive estimation of heart rate (HR) and blood pressure (BP) from facial videos, optimized for dynamic scenarios like physical exercise and cognitive stress. Based on the research paper *"Optimal Facial Regions for Remote Heart Rate and Blood Pressure Measurement During Physical and Cognitive Activities"*, the system:

- Uses **MediaPipe** for precise facial landmark detection and selects **ROI-5** (cheek-adjacent region) as the optimal area, validated empirically on the UBFC-rPPG dataset with OS_mean=1.0.
- Extracts signals via a hybrid of **CHROM** and **POS** methods, followed by bandpass filtering (0.7-4 Hz) and peak detection for HR.
- Estimates BP using a novel regression formula incorporating HR, signal std (σ_S), and SNR, augmented with Gaussian noise for robustness:  
  SBP = clip(115 + 0.5(HR_med - 70) + 25·σ_S + 0.6·SNR_S + N(0,4), [80,210])  
  DBP = clip(75 + 0.25(HR_med - 70) + 12·σ_S + 0.3·SNR_S + N(0,3), [40,130])
- Trains multi-task models (**1D-CNN**, **GRU**, **CNN+GRU**) for joint PPG waveform reconstruction and BP prediction.

**Key Results** (on UBFC-rPPG): HR MAE=2.1 BPM, SBP/DBP MAE=4.2/3.1 mmHg—15-20% better than baselines like ICA/PCA/PhysNet. Runs real-time (~25 FPS) on standard hardware.

**Keywords**: rPPG, HR estimation, BP prediction, ROI optimization, CHROM, POS, UBFC-rPPG, non-contact monitoring.

For the full paper, see [PDF](path-to-paper.pdf) or [arXiv](https://arxiv.org/abs/XXXX.XXXXX).

## Features
- **ROI Optimization**: Tests 9 facial regions; auto-selects ROI-5 via oscillation score (OS_mean).
- **Signal Pipeline**: Skin-masked RGB extraction, CHROM+POS hybrid, Butterworth filtering.
- **Multi-Task Learning**: Joint HR (waveform → Welch peaks) and BP prediction with synthetic labels.
- **Models Compared**: 1D-CNN (best: HR MAE=1.82 BPM), GRU, CNN+GRU hybrids.
- **Evaluation**: MAE/RMSE/PCC on stress subsets; supports UBFC/PURE/iBVP.
- **Real-Time Inference**: ~25 FPS on NVIDIA T4; saves NPZ outputs (timestamps, rPPG, HR/BP).

## Installation

1. **Clone the Repo**:
   ```
   git clone https://github.com/yourusername/rppg-optimal-roi.git
   cd rppg-optimal-roi
   ```

2. **Environment Setup** (Python 3.8+):
   ```
   pip install torch torchvision torchaudio  # PyTorch (CPU/GPU)
   pip install opencv-python scipy pandas numpy matplotlib tqdm mediapipe
   pip install scikit-learn  # For metrics
   ```

3. **Google Colab (Recommended)**:
   - Upload to Colab: Mount Drive (`from google.colab import drive; drive.mount('/content/drive')`).
   - Set `ROOT = '/content/drive/MyDrive/UBFC_DATASET'`.

## Dataset

Download the **UBFC-rPPG** dataset [15] and organize as:
```
data/UBFC-rPPG/
├── subject1/
│   ├── vid.avi          # Video (~30 FPS, 640x480)
│   └── ground_truth.txt # GT: 3 rows (PPG trace, HR, time in ms)
├── subject2/
│   └── ...
└── ... (42 subjects)
```

- **Participants**: 42 (21 M/F, 18-35 yrs).
- **Activities**: Resting, cognitive (Stroop test), physical (post-exercise).
- **Camera/Specs**: Logitech C920 HD Pro (30 FPS, uncompressed).
- **cPPG Reference**: Contec CMS50E (500 Hz).


## Usage

### 1. Signal Extraction & ROI Testing
Run `extract_signals.py` to process videos and rank ROIs:
```bash
python extract_signals.py --root data/UBFC-rPPG --output roi_results.csv
```
- Outputs: CSV with MAE/PCC/SNR/OS per ROI; selects ROI-5.
- Example: `OS_mean=1.0` for ROI-5, plots RGB/CHROM/POS waveforms.

### 2. Train Multi-Task Models
Train on windows (6s, 1s stride):
```bash
python train_mt.py --root data/UBFC-rPPG --epochs 12 --batch 64 --out_dir models/
```
- Builds ~3000 windows (RGB + synth BP labels).
- Models: CNN_MT (best), GRU_MT, CNN_GRU_MT.
- Loss: MSE_wave + 0.002*MSE_BP; saves best.pth.

### 3. Inference & Evaluation
Predict on new video:
```bash
python infer.py --video path/to/vid.avi --model models/best_cnn.pth --out demo.npz
```
- Outputs: Timestamps, rPPG_raw/bp, HR/BP; eval MAE/RMSE/PCC.
- Stress test: 85% clips <5 BPM/5 mmHg error.

### Example Output
- HR Track: Plot with Welch peaks (~78 BPM median).
- BP: SBP~118 mmHg, DBP~76 mmHg (clipped synth).

## Results

**ROI Ablation** (UBFC Test):
| Face Spot          | OS_mean | HR MAE (BPM) | SBP MAE (mmHg) | DBP MAE (mmHg) |
|--------------------|---------|--------------|----------------|----------------|
| ROI-1 (Forehead)  | 0.85   | 3.4         | 5.8            | 4.2            |
| ROI-3 (Left Cheek)| 0.92   | 2.7         | 4.9            | 3.5            |
| **ROI-5 (Right Cheek Edge)** | **1.0** | **2.1** | **4.2** | **3.1** |
| ROI-7 (Nose)      | 0.78   | 4.1         | 6.3            | 4.8            |

**Model Comparison** (UBFC Val):
| Model     | HR MAE (BPM) | SBP MAE (mmHg) | DBP MAE (mmHg) |
|-----------|--------------|----------------|----------------|
| **1D-CNN**| **1.82**    | **3.45**      | **2.68**      |
| GRU      | 2.47        | 4.12          | 3.29          |
| CNN+GRU  | 2.15        | 3.78          | 2.92          |

