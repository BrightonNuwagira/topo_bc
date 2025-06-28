# Breast Cancer Detection with Topological Deep Learning 



This repository provides a comprehensive pipeline for breast cancer detection using topological data analysis (TDA) integrated with machine learning models. Below are the key features:




## **Table of Contents**
1. [Installation](#installation)
2. [Data](#data)
3. [Topological Feature Extraction](#topological-feature-extraction)
   - [Betti Vectors](#betti-vectors)
   - [Persistence Images](#persistence-images)
4. [Model Architectures](#model-architectures)
   - [Betti-CNN](#betti-cnn)
   - [PI-CNN](#pi-cnn)
   - [Vanilla CNN](#vanilla-cnn)
   - [Topological Swin Transformer (`toposwin.py`)](#topological-swin-transformer-toposwinpy)
   - [Swin Transformer Baseline (`swin.py`)](#swin-transformer-baseline-swinpy)
5. [Betti Encoder](#betti-encoder)
6. [Comparison of Models](#comparison-of-models)
7. [Details for Each File](#details-for-each-file)
8. [Reproducibility](#reproducibility)

---









1. **Installation**
- Python 3.9 
- [Giotto-TDA]([https://giotto.ai/](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html))

### 2. **Data**

This repository supports multiple datasets for breast cancer classification, which can be accessed and downloaded from the following link: [Breast Cancer Datasets](https://drive.google.com/drive/folders/19Xs8DrF9OBCWupTr1mTlCzC0mHICsffC?usp=sharing).

The datasets include:

- **BUSI Dataset**: Breast ultrasound images categorized into `benign`, `malignant`, and `normal`.
- **BUS-BRA Dataset**: An additional breast ultrasound dataset with similar classes.
- **MENDELEY Dataset**: A collection of labeled ultrasound images for breast cancer detection.

1. **Topological Feature Extraction**
   - **Betti Vectors**:
     - Extract Betti numbers using `Betti_vectors.py`.
     - Betti numbers describe the number of connected components, holes, and voids in the data.
     ```bash
     python Betti_vectors.py --input_path ./data --output_path ./results/betti_vectors.csv
     ```
   - **Persistence Images**:
     - Compute persistence images from persistence diagrams using `Persistance_Image.py`.
     - Persistence images encode topological summaries into a vector format suitable for machine learning.
     ```bash
     python Persistance_Image.py --input_path ./data --output_path ./results/persistence_images.npy
     ```

2. **Model Architectures**
   - **Betti-CNN**:
     - Implements a Convolutional Neural Network (CNN) enhanced with Betti vectors for feature augmentation.
     - File: `Betti-CNN.py`.
     ```bash
     python Betti-CNN.py --betti_vectors ./results/betti_vectors.csv --labels ./data/labels.csv
     ```
   - **PI-CNN**:
     - CNN trained using persistence images as input features.
     - File: `PI-CNN.py`.
     ```bash
     python PI-CNN.py --persistence_images ./results/persistence_images.npy --labels ./data/labels.csv
     ```
   - **Vanilla CNN**:
     - Baseline CNN implementation without topological features, allowing comparison with Betti-CNN and PI-CNN.
     - File: `Vanilla-CNN.py`.
     ```bash
     python Vanilla-CNN.py --input_images ./data/images --labels ./data/labels.csv
     ```

3. **Comparison of Topological Approaches**
   - The repository enables a direct comparison between:
     - CNNs using Betti numbers (`Betti-CNN.py`).
     - CNNs using persistence images (`PI-CNN.py`).
     - Standard CNNs (`Vanilla-CNN.py`).

4. **Reproducibility**
   - Scripts are included for training models, evaluating performance, and visualizing results.
   - All necessary files to reproduce results from the paper are included or can be generated.

---

### **Additional Implementations**

This repository also includes advanced implementations leveraging topological features in state-of-the-art architectures, such as Swin Transformers.

#### 1. **Topological Swin Transformer (`toposwin.py`)**
   - A novel architecture integrating Betti vectors with the Swin Transformer.
   - Implements cross-attention mechanisms to combine image features with topological descriptors, enhancing performance for breast cancer detection.
   - **Key Features**:
     - Uses Betti numbers for feature augmentation through `BettiEncoder`.
     - Incorporates hierarchical feature extraction via Swin Transformer.
     - File Path: `toposwin.py`.
   - **Usage**:
     ```bash
     python toposwin.py --input_images ./data/images --betti0 ./data/betti0.csv --betti1 ./data/betti1.csv --labels ./data/labels.csv
     ```

#### 2. **Swin Transformer Baseline (`swin.py`)**
   - A baseline implementation of the Swin Transformer model for comparison with topological integrations.
   - Supports standard breast cancer detection tasks.
   - File Path: `swin.py`.
   - **Usage**:
     ```bash
     python swin.py --input_images ./data/images --labels ./data/labels.csv
     ```

#### 3. **Betti Encoder (`betti_encoder.py`)**
   - A standalone module to encode Betti curves into high-dimensional representations using transformer-based encoding.
   - **Features**:
     - Layer normalization and positional embeddings.
     - Configurable transformer encoder architecture.
   - **Integrations**:
     - Used in `toposwin.py` to process Betti curves.
     - Includes a `BettiClassifier` for standalone classification tasks using Betti numbers.
   - File Path: `betti_encoder.py`.
   - **Usage**:
     ```python
     from betti_encoder import BettiEncoder, BettiClassifier
     encoder = BettiEncoder(seq_length=100, d_model=512, nhead=4)
     ```

---

### **Comparison of Models**

| Model               | Topological Features | Architecture          | Application                              |
|---------------------|-----------------------|-----------------------|------------------------------------------|
| **Vanilla CNN**     | No                   | CNN                   | Baseline for breast cancer detection.   |
| **Betti-CNN**       | Yes (Betti Vectors)  | CNN + Feature Augmentation | Topology-enhanced CNN.                 |
| **PI-CNN**          | Yes (Persistence Images) | CNN                 | Utilizes persistence image features.    |
| **Swin Transformer**| No                   | Swin Transformer      | State-of-the-art image classification.  |
| **TopoSwin**        | Yes (Betti Vectors)  | Swin + Cross Attention | Combines topology with hierarchical learning. |






### **Details for Each File**

Here’s how the files contribute to the project:

1. **`Betti_vectors.py`**:
   - Computes Betti numbers for the input data.
   - Accepts raw images or other forms of data as input.
   - Outputs a CSV file containing the computed Betti numbers.

2. **`Persistance_Image.py`**:
   - Converts persistence diagrams into persistence images.
   - Provides a vectorized representation of topological features.

3. **`Betti-CNN.py`**:
   - CNN implementation that combines image features with Betti vectors.

4. **`PI-CNN.py`**:
   - CNN implementation using persistence images as input.

5. **`Vanilla-CNN.py`**:
   - Baseline CNN for comparison, trained directly on the raw input images.

---

