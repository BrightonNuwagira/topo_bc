# Topology Meets Deep Learning for Breast Cancer Detection (topo_bc)

# Topological Data Analysis for Breast Cancer Detection (topo_bc)

This repository implements **Topological Data Analysis (TDA)** integrated with **Deep Learning (DL)** methods for breast cancer detection, as detailed in our paper Topology Meets Deep Learning for Breast Cancer Detection.

- Extraction of topological features such as Betti curves and persistence diagrams.
- Integration with state-of-the-art models (CNNs, Vision Transformers).
- Applied to multiple breast cancer datasets (BUSI, BUS-BRA, Mendeley).
- Reproducible experiment scripts and visualization notebooks.

---

## Table of Contents

2. [Installation](#installation)
3. [Usage](#usage)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Extract Topological Features](#2-extract-topological-features)
    - [3. Train a Model](#3-train-a-model)
4. [Results](#results)
5. [Repository Structure](#repository-structure)
6. [Contributing](#contributing)
7. [License](#license)

---



## Installation

### Prerequisites
- Python 3.9 or later
- [Giotto-TDA](https://giotto.ai/)

Install dependencies:
```bash
git clone https://github.com/BrightonNuwagira/topo_bc.git
cd topo_bc
pip install -r requirements.txt
