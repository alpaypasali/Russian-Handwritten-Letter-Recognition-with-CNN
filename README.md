# Russian Handwritten Letter Recognition with CNN

A deep learning project that classifies handwritten Russian alphabet letters using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## 📌 Overview

This project trains a CNN model on the [Russian Handwritten Letters](https://www.kaggle.com/datasets/tatianasnwrt/russian-handwritten-letters) dataset to recognize all 33 letters of the Russian (Cyrillic) alphabet from real handwritten images.

---

## 📂 Dataset

| Property | Value |
|---|---|
| Total images | 14,190 |
| Classes | 33 (one per Russian letter) |
| Image size | 32 × 32 pixels |
| Format | RGB |

---

## 🗂️ Project Structure

```
├── notebook.ipynb       # Main Kaggle notebook
├── README.md
```

---

## 🔄 Pipeline

```
Raw Images + CSV Labels
        ↓
Train / Val / Test Split  (70% / 15% / 15%)
        ↓
Load & Normalize  (pixel / 255.0)
        ↓
One-Hot Encoding  (labels 1–33 → 0-based → categorical)
        ↓
tf.data.Dataset  (batch size: 32)
        ↓
CNN Model Training
        ↓
Evaluation & Prediction Visualization
```

---

## 🧠 Model Architecture

```
Input (32, 32, 3)
    │
    ├─ Conv2D(32, 3×3, ReLU)
    ├─ MaxPooling2D(2×2)
    │
    ├─ Conv2D(64, 3×3, ReLU)
    ├─ MaxPooling2D(2×2)
    │
    ├─ Conv2D(128, 3×3, ReLU)
    │
    ├─ Flatten
    ├─ Dense(256, ReLU)
    └─ Dense(33, Softmax)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Metrics | Accuracy, AUC |
| Epochs | 50 (max) |
| Batch size | 32 |
| Early Stopping | patience=10, monitors val_loss |
| Model Checkpoint | saves best weights |

---

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## 🚀 How to Run

1. Open the notebook on [Kaggle](https://www.kaggle.com)
2. Add the **Russian Handwritten Letters** dataset
3. Enable GPU accelerator *(optional but recommended)*
4. Run all cells

---

## 📊 Results

Predictions on the test set are visualized with color-coded labels:

- 🟢 **Green** → Correct prediction
- 🔴 **Red** → Wrong prediction

---

## 👤 Author

**Alpay**  
[Kaggle](https://www.kaggle.com) · [GitHub](https://github.com)
