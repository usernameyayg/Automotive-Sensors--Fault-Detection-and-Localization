# 🚗 Intelligent Analysis of Automotive Sensor Faults Using Self-Supervised Learning and Real-Time Simulation

**Master Thesis Project**: Fault Detection & Localization in Automotive Sensors via SSL and Mahalanobis Distance  
**Author**: Yahia Amir Yahia Gamal  
**Supervisors**: apl. Prof. Dr. Christoph Knieke, Dr. Mohammad Abboush  
**Institution**: TU Clausthal, 2025

---

## 📋 Overview

This repository implements a **Self-Supervised Learning (SSL)** pipeline for detecting and localizing faults in automotive sensor systems using realistic Hardware-in-the-Loop (HIL) fault injection.

### 🎯 Key Innovation
- ✅ **No labeled fault data required** - Learns normal behavior from healthy sensors only
- ✅ **Realistic evaluation** - HIL-faithful fault injection (bias, noise, gain) matching real automotive faults
- ✅ **Interpretable localization** - Ablation-based method identifies which sensor failed
- ✅ **Ensemble robustness** - 5 independent models for stability
- ✅ **Transfer learning** - Trained on one bus, tested on different vehicle

---

## 📊 Results

### Localization Performance (4 Independent Runs)

| Run | Seed | Faults | Top-1 Accuracy | Top-2 Accuracy | Avg Confidence |
|-----|------|--------|---|---|---|
| Run 1 | 600 | 6 | **50.0%** | **66.7%** | 26.6 |
| Run 2 | 673 | 3 | **66.7%** | **66.7%** | 18.2 |
| Run 3 | 782 | 4 | **25.0%** | **50.0%** | 8.9 |
| Run 4 | 280 | 3 | **33.3%** | **66.7%** | 14.5 |
| **TOTAL** | — | **16 faults** | **43.8% avg** | **62.5% avg** | **17.1 avg** |

### Magnitude-Dependent Performance
```
High-impact faults (ΔE > 0.6):    100% localization ✅
Medium-impact (0.2-0.6):          ~50-70% localization ⚠️
Low-impact (ΔE < 0.2):            0% localization ✗ (physically realistic)
```

### Confidence Analysis
- **Correct predictions**: 26.6 avg confidence
- **Incorrect predictions**: 19.1 avg confidence  
- **Confidence gap**: 7.56 (statistically significant)

### Detection Metrics (Validation Set - 70/30 Split)
| Metric | Value |
|--------|-------|
| **Mahalanobis threshold** | 95th percentile (training data) |
| **Normal data mean distance** | 15-17 |
| **Normal data std distance** | ~6 |
| **Validation set** | 430 windows (healthy baseline) |

---

## 🏗️ Architecture Overview

### 7-Part Pipeline

| Part | Stage | Input | Output |
|------|-------|-------|--------|
| **0** | Configuration | User sensor selection | Sensor list, hyperparameters |
| **1** | Data preprocessing | Raw JSON (50 Hz resampled) | Normalized windowed dataset (70/30 split) |
| **2** | SSL augmentations | Time-series windows | 3 augmentation types (jitter, masking, negation) |
| **3** | Model architecture | Sensor config | 1D-CNN encoder (512-dim embeddings) |
| **4** | SSL training | Normal data only | 5 trained models (different seeds) |
| **5** | Mahalanobis detector | Training embeddings | μ, Σ⁻¹, threshold per seed |
| **6** | HIL fault injection | New unseen dataset | Labeled test set (16 faults, 1163 windows) |
| **7** | Ablation localization | Faulty windows | Per-sensor importance scores, predictions |

### 1D-CNN Encoder Specifications
```
Input:  (batch, num_sensors, 64 timesteps)
        ↓
Conv1d: 64 → 128 → 256 → 512 channels
        + BatchNorm, ReLU, max pooling × 4
        ↓
Global avg pooling
        ↓
512-dim embeddings (feature space)
        ↓
Classification head: 512 → 256 → 3 (augmentation types)
```

---

## 📈 HIL-Faithful Fault Injection

Faults strictly match Hardware-in-the-Loop specifications (from real automotive data):

| Fault Type | Acceleration | Angular Velocity | Steering | Accelerator |
|-----------|---|---|---|---|
| **Bias** | ±0.1-2.0 | ±0.05-1.0 | ±0.5-3.0 | ±0.5-4.0 |
| **Noise** | 0.5-1.5σ | 0.5-1.5σ | 0.8-2.0σ | 1.0-2.5σ |
| **Gain** | 0.85-1.15 | 0.90-1.10 | 0.90-1.10 | 0.85-1.15 |

**Contamination**: ~1% per run (realistic automotive scenario)

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yahia-amir/automotive-ssl-fault-detection.git
cd automotive-ssl-fault-detection

# Install dependencies
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy tqdm

# Ensure data path is set
# Edit RAW_JSON path in PART 1 to point to your A2D2 dataset
```

### Run Full Pipeline
```python
# Execute notebooks/parts in order:
# PART 0: Sensor selection
# PART 1: Data loading & preprocessing  
# PART 2: Augmentation definitions
# PART 3: Model architecture
# PART 4: SSL training (all 5 seeds)
# PART 5: Mahalanobis threshold calibration
# PART 6: HIL-faithful fault injection
# PART 7: Ablation-based localization
```

### Example: Detect Faults in New Data
```python
# After training (Part 4):
model = trained_models[42]  # Seed 42
mu = mahal_data[42]['mu']
cov_inv = mahal_data[42]['cov_inv']
threshold = mahal_data[42]['threshold']

# Extract embeddings from new window
embeddings = model.get_embeddings(new_window_tensor)

# Compute Mahalanobis distance
diff = embeddings - mu
distance = np.sqrt(np.sum(diff @ cov_inv * diff))

# Detect fault
if distance > threshold:
    print("⚠️ FAULT DETECTED")
else:
    print("✅ Normal operation")
```

---

## 📁 Repository Structure
```
automotive-ssl-fault-detection/
├── PART_0_SENSOR_CONFIG.py           # Sensor selection & configuration
├── PART_1_DATA_PREPROCESSING.py      # Load JSON, resample, normalize, window
├── PART_2_AUGMENTATIONS.py           # Define SSL augmentations
├── PART_3_MODEL_ARCHITECTURE.py      # 1D-CNN encoder definition
├── PART_4_SSL_TRAINING.py            # Train on 5 seeds
├── PART_5_MAHALANOBIS_DETECTOR.py    # Fit Mahalanobis distance
├── PART_6_FAULT_INJECTION.py         # HIL-faithful fault generation
├── PART_7_LOCALIZATION.py            # Ablation-based sensor identification
├── results_DYNAMIC_SENSORS/
│   ├── model_seed*.pt                # Trained models
│   ├── fault_injection_log_*.csv     # Fault metadata
│   ├── enhanced_sensor_localization_analysis.png
│   ├── training_history.png
│   └── mahalanobis_distances_70_30.png
├── README.md
└── requirements.txt
```

---

## 🔬 Methodology Highlights

### Self-Supervised Learning
- **Training:** Encoder learns to predict which augmentation was applied (no fault labels)
- **Augmentations:** Jitter (±2% noise), temporal masking (10%), polarity negation
- **Embeddings:** 512-dim feature space learned from healthy data only

### Anomaly Detection
- **Mahalanobis distance:** D = √[(x-μ)ᵀΣ⁻¹(x-μ)]
- **Threshold:** 95th percentile of training distances
- **Interpretation:** High distance = out-of-distribution = fault detected

### Fault Localization
- **Method:** Ablation analysis across 5 ensemble models
- **Process:** Mask each sensor, measure distance change, identify most important sensor
- **Robustness:** Aggregates importance scores across 5 seeds for stability

---

## 📊 Key Findings

1. **Magnitude-dependent performance:** High-impact faults (>0.6Δ) achieve 100% localization; subtle faults undetectable (realistic)
2. **Sensor-specific:** Angular velocity sensors localized better (80%) than steering angle (20%)
3. **Confidence discriminative:** 7.56-point gap between correct/incorrect proves model learns meaningful patterns
4. **Transfer learning:** Different vehicle, different day → proves generalization
5. **HIL-faithful:** Results realistic for automotive deployment

---

## 📚 Thesis Chapters

- **Introduction:** Automotive sensor failures, SSL motivation
- **Related Work:** Anomaly detection, transfer learning, automotive fault diagnosis
- **Methodology:** Parts 0-7 pipeline, Mahalanobis distance, ablation analysis
- **Results:** 4 fault injection scenarios, magnitude-dependent performance
- **Discussion:** Limitations, physical interpretability, deployment readiness
- **Conclusion:** Novel SSL approach, realistic evaluation, future work

---

## 🎓 Thesis Citation
```bibtex
@mastersthesis{gamal2025automotive,
  title={Intelligent Analysis of Automotive Sensor Faults Using 
         Self-Supervised Learning and Real-Time Simulation},
  author={Gamal, Yahia Amir Yahia},
  school={TU Clausthal},
  year={2025},
  supervisors={Knieke, Christoph and Abboush, Mohammad}
}
```

---

## 📄 License

This project is part of academic research. Please cite appropriately.

---

## 🔗 Dataset

- **Audi A2D2**: https://www.a2d2.audi/ (autonomous driving dataset)
- **Sensors used**: 8 (acceleration x/y/z, angular velocity x/y/z, steering, accelerator)
- **Sampling rate**: 50 Hz (resampled from 50-200 Hz)

---

## ⚠️ Important Notes

- **No synthetic faults in training**: Model never sees faulty data during SSL training
- **Different vehicles**: Training on Munich bus (20190401121727), testing on different bus (20190401145936)
- **Realistic contamination**: ~1% fault rate matches real automotive scenarios
- **Interpretability**: Ablation method is human-understandable (which sensor caused fault?)

---

**Last Updated**: 2025  
**Status**: ✅ Thesis-Ready  
**Validation**: 4 independent runs, 16 total faults, 5-seed ensemble
