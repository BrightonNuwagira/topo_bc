# Breast Cancer Detection with Topological Deep Learning

This repository accompanies our ICCV 2025 paper, **"Breast Cancer Detection with Topological Deep Learning"**, which introduces a novel integration of topological data analysis (TDA) with convolutional and transformer-based deep learning architectures for improved breast cancer classification.

Our approach extracts topological descriptors—**Betti curves** and **persistence images**—and fuses them with image features using architectures including CNNs and Swin Transformers. This repository contains code for TDA feature extraction, deep learning model training, and reproducibility of our published results.

---

## 🚀 Highlights
- **Betti-CNN** and **PI-CNN**: CNN architectures augmented with Betti curves and persistence images.
- **TopoSwin**: A novel architecture combining Swin Transformer features with Betti encodings via cross-attention.
- **Transformer-based Betti Encoder** for end-to-end topological representation learning.
- Benchmarked on real breast ultrasound datasets with superior performance over vanilla CNNs and standard transformers.

---

## 📁 Repository Structure

```
.
├── Betti-CNN.py            # CNN using Betti vectors
├── PI-CNN.py               # CNN using persistence images
├── Vanilla-CNN.py          # Standard DenseNet-based CNN
├── toposwin.py             # Topological Swin Transformer
├── swin.py                 # Baseline Swin Transformer
├── betti_encoder.py        # Transformer encoder for Betti curves
├── Persistance_Image.py    # Compute persistence images
├── 3D_Betticurves/         # Contains sample Betti curve files
└── TDA___CNN_Breast_Cancer.pdf # ICCV 2025 paper
```

---

## 🧪 Installation

```bash
git clone https://github.com/<your-username>/topo-breast-cancer.git
cd topo-breast-cancer
pip install -r requirements.txt
```

- Requires Python ≥ 3.9
- Dependencies: `torch`, `timm`, `scikit-learn`, `keras`, `giotto-tda`, `numpy`, `pandas`

---

## 🗂️ Datasets

We evaluate on several public breast ultrasound datasets, including:

- **BUSI Dataset**
- **BUS-BRA Dataset**
- **MENDELEY Dataset**

Data should be structured in folders or CSV format containing:
- Raw images
- Labels
- Precomputed Betti0 and Betti1 curves (if applicable)

---

## 🧠 Models and Usage

### 1. Vanilla CNN
```bash
python Vanilla-CNN.py
```

### 2. Betti-CNN
```bash
python Betti-CNN.py
```

### 3. PI-CNN
```bash
python PI-CNN.py
```

### 4. Swin Transformer (Baseline)
```bash
python swin.py
```

### 5. TopoSwin: Swin + Betti Encoding + Cross Attention
```bash
python toposwin.py --input_images ./data/images --betti0 ./data/betti0.csv --betti1 ./data/betti1.csv --labels ./data/labels.csv
```

---

## 🔍 Topological Feature Extraction

### Betti Vectors (Betti0 & Betti1)
Extracted using the **giotto-tda** library or custom scripts. See sample in `3D_Betticurves/`.

### Persistence Images
```bash
python Persistance_Image.py --input_path ./data --output_path ./results/persistence_images.npy
```

---

## 🔧 Betti Encoder

```python
from betti_encoder import BettiEncoder
encoder = BettiEncoder(seq_length=100, d_model=512, nhead=4)
```

Used within `toposwin.py` to learn representations of topological features via a transformer.

---

## 📊 Performance Summary

| Model           | Topological Input | Architecture       | Best Use Case                   |
|-----------------|-------------------|--------------------|----------------------------------|
| Vanilla CNN     | ❌                | DenseNet121        | Baseline Comparison              |
| Betti-CNN       | ✅ Betti Curves   | CNN + Betti        | Topology-aware CNN               |
| PI-CNN          | ✅ Persistence Img| CNN + PI           | Persistence-enhanced CNN         |
| Swin Transformer| ❌                | Vision Transformer | SOTA backbone                    |
| TopoSwin        | ✅ Betti Curves   | Swin + Attention   | Best performance with topology   |

---


