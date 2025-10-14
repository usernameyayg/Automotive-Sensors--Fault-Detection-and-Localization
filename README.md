# 🚗 Automotive Sensor Fault Detection & Localization

**Master Thesis Project**: Self-Supervised Learning for Real-Time Fault Detection in Automotive IMU Sensors

**Author**: Yahia Amir Yahia Gamal  
**Year**: 2024-2025

---

## 📋 Overview

This repository contains a complete implementation of a **Self-Supervised Learning (SSL)** approach for detecting and localizing faults in automotive Inertial Measurement Unit (IMU) sensors.

### Key Features
- ✅ Learns from normal data only (no labeled faults needed)
- ✅ Detects 3 fault types: Drift, Stuck-at, Spike bursts
- ✅ Localizes faults to specific sensor channels
- ✅ Real-time capable: 0.96ms inference (ISO 26262 compliant)
- ✅ Statistically validated: 5-seed cross-validation

---

## 📊 Results

### Detection Performance (Main Experiment)

| Metric | Score |
|--------|-------|
| **Precision** | **80.95%** |
| **Recall** | 56.67% |
| **F1-Score** | **66.67%** |
| **ROC-AUC** | **0.886** |
| **Localization** | **66.67%** |

### Statistical Validation (5 Seeds)

| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| **F1-Score** | 63.3% ± 7.9% | [56.4%, 70.2%] |
| **Localization** | 55.3% ± 17.2% | [40.2%, 70.3%] |
| **ROC-AUC** | 86.3% ± 3.0% | [83.6%, 88.9%] |

### Computational Performance

| Task | Time | Real-Time? |
|------|------|------------|
| **Training** | 7.2 minutes | N/A |
| **Inference** | **0.96 ms/window** | ✅ YES |
| **Localization** | 9.88 ms/fault | ✅ YES |

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+

### Install Dependencies
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy
