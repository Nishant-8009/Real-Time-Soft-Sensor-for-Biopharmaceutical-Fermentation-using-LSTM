# LSTM-Based Soft Sensor for Biopharmaceutical Fermentation

Real-Time Prediction of Critical Quality Variables using Deep Learning

---

## 📌 Overview

This project presents an **LSTM-based soft sensor** designed for real-time prediction of critical quality variables (CQVs) in large-scale biopharmaceutical fermentation processes.

Traditional fermentation monitoring relies heavily on offline laboratory assays that are performed approximately every 12 hours. This creates significant monitoring blind spots and limits the ability to perform real-time process control.

To address this challenge, we developed a deep learning-based soft sensor capable of inferring hard-to-measure CQVs directly from continuously available online sensor signals.

The model successfully predicts:

* Penicillin Concentration
* Biomass
* Phenylacetic Acid (PAA)
* Viscosity

with high accuracy on unseen fermentation batches.

---

## 🎯 Objective

Build a robust sequence-learning model capable of:

* Predicting sparse CQVs using online process signals
* Eliminating information leakage through proper batch-level splitting
* Achieving strong generalization on unseen fermentation batches
* Enabling real-time quality monitoring for pharmaceutical manufacturing

---

## 🧪 Dataset

### IndPenSim v2 Dataset

* 100 simulated penicillin fermentation batches
* Sampling interval: **0.2 hours**
* Batch duration: ~230 hours
* Total rows: ~113,900
* Total features: 2239 columns

### Target Variables

The CQVs are sparsely measured at approximately **1 out of every 60 timesteps**:

* Penicillin concentration (P)
* Biomass concentration (X)
* Phenylacetic acid (PAA)
* Viscosity

---

## ⚙️ Preprocessing Pipeline

### 1. Raman Spectra Compression

The original Raman spectra consisted of ~2200 highly correlated channels.

We applied:

* **PCA Compression:**

  * 2200 → 10 principal components
  * 99.9% variance retained
  * ~220× dimensionality reduction

### 2. Feature Engineering & Transformations

Different transformations were applied based on signal characteristics:

* `log1p` for right-skewed distributions
* `Yeo-Johnson` transformation for asymmetric features
* `StandardScaler` for normalization

### 3. Correlation Pruning

Highly correlated features (`|r| ≥ 0.90`) were removed based on:

* Physical interpretability
* Redundancy reduction
* Stability improvement

### 4. Final Feature Set

Final input representation:

* 17 engineered process features
* Raw time feature
* 10 PCA components

**Total Input Channels = 28**

---

## 🧠 Model Architecture

The soft sensor uses a **stacked LSTM architecture** for sequential modeling of fermentation dynamics.

### Architecture Details

```text
Input (28 Features)
        ↓
LSTM Layer (128 Hidden Units)
        ↓
Dropout (0.2)
        ↓
LSTM Layer (128 Hidden Units)
        ↓
Fully Connected Layer (128 → 64)
        ↓
ReLU Activation
        ↓
Dropout
        ↓
Output Layer (64 → 4)
```

### Model Characteristics

* ~270K trainable parameters
* Batch-first sequence format
* Multi-output regression

---

## 📉 Masked MSE Loss

Since target labels exist only for sparse timesteps, a standard MSE loss would incorrectly optimize over unlabelled positions.

To solve this, we implemented a **Masked Mean Squared Error Loss**:

```math
L = (1 / |M|) Σ ||ŷ_t - y_t||²
```

where:

* `M` = set of labelled timesteps
* Loss is computed only over valid target positions

This prevents degenerate learning on missing labels.

---

## 🚀 Training Strategy

### Training Configuration

* Optimizer: Adam
* Learning Rate: 0.001
* Scheduler: ReduceLROnPlateau
* Batch Size: 1
* Epochs: 50
* Gradient Clipping: `||∇|| ≤ 1.0`

### Dataset Split

To avoid temporal leakage:

* 80% Training Batches
* 10% Validation Batches
* 10% Test Batches

**Batch-level splitting** ensured zero overlap between train and test trajectories.

---

## 📊 Results

### Held-Out Test Performance

| Target           | RMSE  | R² Score |
| ---------------- | ----- | -------- |
| Penicillin (g/L) | 1.052 | 0.9885   |
| Biomass (g/L)    | 0.931 | 0.9733   |
| PAA (g/L)        | 164.5 | 0.9563   |
| Viscosity (cP)   | 2.501 | 0.9852   |

### Key Achievement

✅ Achieved **R² > 0.95** on all four CQVs using only online process signals.

---

## 🔍 Key Design Choices

### Batch-Level Splitting

Prevents temporal information leakage and ensures realistic evaluation.

### Masked Loss Function

Avoids optimization over missing labels and stabilizes training.

### PCA Compression

Reduces spectral noise and improves gradient flow.

### Feature-Specific Transformations

Improves learning dynamics for skewed industrial sensor signals.

### Raw Time Feature

Provides absolute temporal context to the sequence model.

---

## 📈 Impact

This system helps close the 12-hour monitoring gap in pharmaceutical fermentation by enabling:

* Real-time CQV prediction
* Continuous process monitoring
* Faster anomaly detection
* Improved process control
* Enhanced manufacturing reliability

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* PCA
* LSTM Networks
* Deep Learning

---

## 👨‍💻 Contributors

* Nishant Malhotra
* Avinash Kumar

### Mentor

* Nitin
* Dr. Sukrit Gupta

---

## 📌 Future Improvements

Potential extensions of this work include:

* Transformer-based sequence models
* Attention-enhanced soft sensors
* Real-time deployment pipelines
* Multi-agent industrial control systems
* Hybrid physics-informed ML architectures

---

## ⭐ Conclusion

This project demonstrates the effectiveness of deep learning-based soft sensors for industrial biopharmaceutical fermentation.

By combining:

* LSTM sequence modeling
* Sparse-label masked training
* PCA-based compression
* Robust preprocessing

we achieved highly accurate real-time prediction of critical process variables while maintaining strong generalization across unseen batches.

The approach provides a scalable foundation for intelligent monitoring systems in modern pharmaceutical manufacturing.
