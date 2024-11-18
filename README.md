# Topology Meets Deep Learning for Breast Cancer Detection (topo_bc)

## Features

This repository provides a comprehensive pipeline for breast cancer detection using topological data analysis (TDA) integrated with machine learning models. Below are the key features:


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

