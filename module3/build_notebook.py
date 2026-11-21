import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    lines = [line + "\n" for line in text.strip().split("\n")] if text.strip() else []
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# Title
add_md("""# Module 3: Supervised Learning & Baseline Benchmarking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module3/03_supervised_predictive_modelling.ipynb)

**Course**: BITSF464 Machine Learning — Fall 2026  
**Project**: Seismological Fault Line Discovery & Earthquake Impact Profiling  

---

### Objectives
1. **Two-Stage Benchmarking**: Train and evaluate baseline machine learning algorithms on:
   - **Stage 1 (Base)**: Imputed & standardized physical coordinates and sensor metrics from Module 1.
   - **Stage 2 (Augmented)**: Base features coupled with tectonic fault distances and GMM posterior probabilities from Module 2.
2. **Team Roles**:
   - **Person 1**: Linear & Regularized Models (Logistic Regression with L1/L2/ElasticNet, Ridge/Lasso for MMI).
   - **Person 2**: Probabilistic & Discriminant Models (Gaussian Naive Bayes, LDA with shrinkage, QDA).
   - **Person 3 (You)**: Distance-Based & Margin Models ($k$-Nearest Neighbors, Support Vector Classifiers with probability calibration).
3. **Module 4 Handoff**: Generate cross-validated Out-of-Fold (OOF) prediction probabilities for the stacking meta-learner.""")

# Cell 1: Environment Setup
add_md("""## 0. Environment Setup (Colab & Local)
Detects environment automatically. Sets data and artifact directories without requiring manual path changes.""")

add_code("""import os
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab.")
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_NAME = "seismic-fault-impact"
    REPO_PATH = f"/content/{REPO_NAME}"
    if not os.path.exists(REPO_PATH):
        !git clone https://github.com/Tulip0648/seismic-fault-impact.git {REPO_PATH}
    %cd {REPO_PATH}/module3
    !pip install -r ../requirements.txt -q
    DATA_BASE_DIR = "../data/processed"
    DATA_AUG_DIR = "../data/augmented"
    ARTIFACTS_DIR = "./artifacts"
    REPORTS_DIR = "./reports"
else:
    print("Running locally.")
    DATA_BASE_DIR = "../data/processed"
    DATA_AUG_DIR = "../data/augmented"
    ARTIFACTS_DIR = "./artifacts"
    REPORTS_DIR = "./reports"

os.makedirs(f"{ARTIFACTS_DIR}/models", exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/oof_predictions", exist_ok=True)
os.makedirs(f"{REPORTS_DIR}/figures", exist_ok=True)
print("Directories initialized successfully.")""")

# Cell 2: Imports
add_md("""## 1. Imports & Global Configuration""")
add_code("""import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
N_SPLITS = 5

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Global configuration loaded.")""")

# Cell 3: Data Loading
add_md("""## 2. Shared Data Ingestion & Target Preparation
Loads the exact same Base and Augmented datasets created by Module 1 and Module 2.  
Target mapping: `green`: 0, `yellow`: 1, `orange`: 2, `red`: 3.""")

add_code("""# Target encoding mapping
ALERT_MAP = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
INV_ALERT_MAP = {v: k for k, v in ALERT_MAP.items()}

# 1. Load Base Splits (Module 1)
train_base = pd.read_parquet(os.path.join(DATA_BASE_DIR, "train_base.parquet"))
val_base = pd.read_parquet(os.path.join(DATA_BASE_DIR, "val_base.parquet"))
test_base = pd.read_parquet(os.path.join(DATA_BASE_DIR, "test_base.parquet"))

# 2. Load Augmented Splits (Module 2)
train_aug = pd.read_parquet(os.path.join(DATA_AUG_DIR, "train_augmented.parquet"))
val_aug = pd.read_parquet(os.path.join(DATA_AUG_DIR, "val_augmented.parquet"))
test_aug = pd.read_parquet(os.path.join(DATA_AUG_DIR, "test_augmented.parquet"))

# 3. Ensure target labels are encoded
for df in [train_base, val_base, test_base, train_aug, val_aug, test_aug]:
    if 'alert_label' not in df.columns and 'alert' in df.columns:
        df['alert_label'] = df['alert'].map(ALERT_MAP)

print(f"Train Base shape:      {train_base.shape}")
print(f"Train Augmented shape: {train_aug.shape}")
print(f"Test Base shape:       {test_base.shape}")
print(f"Test Augmented shape:  {test_aug.shape}")""")

# Cell 4: Feature Columns & Evaluation Helper
add_md("""## 3. Feature Selection & Cross-Validation Setup
Defines the feature column subsets and initializes the shared 5-Fold Stratified Cross-Validation splitter.""")

add_code("""# Excluded columns (metadata, raw categorical strings, and raw targets)
EXCLUDE_COLS = {'event_id', 'time', 'alert', 'alert_label', 'mmi', 'place', 'magType'}

# Base features: columns present in train_base minus excluded
BASE_FEATURES = [c for c in train_base.columns if c not in EXCLUDE_COLS]

# Augmented features: columns present in train_aug minus excluded
AUGMENTED_FEATURES = [c for c in train_aug.columns if c not in EXCLUDE_COLS]

print(f"Base feature count:       {len(BASE_FEATURES)}")
print(f"Augmented feature count:  {len(AUGMENTED_FEATURES)}")

# Shared Stratified K-Fold splitter
cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    \"\"\"Calculates Macro F1, Weighted F1, and Multi-Class ROC-AUC (One-vs-Rest).\"\"\"
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            metrics['roc_auc_ovr'] = np.nan
    return metrics""")

# Person 1 Section
add_md("""---
## 4. Person 1: Linear & Regularized Models
**Owner**: Person 1  
**Models**: Logistic Regression (L1, L2, ElasticNet) & Ridge/Lasso for MMI.  
*(Left empty for Person 1 to implement)*""")

add_code("""# --- Person 1: Implement Logistic Regression & Regularized Linear Models here ---
""")

# Person 2 Section
add_md("""---
## 5. Person 2: Probabilistic & Discriminant Models
**Owner**: Person 2  
**Models**: Gaussian Naive Bayes, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA).  
*(Left empty for Person 2 to implement)*""")

add_code("""# --- Person 2: Implement Naive Bayes, LDA, and QDA here ---
""")

# Person 3 Section (The User's Section)
add_md("""---
## 6. Person 3 (Your Section): Distance-Based & Margin Models

**Owner**: Person 3 (You)  
**Algorithms**:
1. **k-Nearest Neighbors (k-NN)**: Distance-based classification.
2. **Support Vector Classifier (SVC)**: Maximum margin classification with kernel transformations and probability calibration.

---

### 6.1 k-Nearest Neighbors (k-NN) — Stage 1 (Base Features)
Implement and tune $k$-NN on `BASE_FEATURES` using the 5-fold cross-validation split:
- Sweep $k \in [3, 5, 7, 11, 15, 21]$.
- Compare `weights='uniform'` vs `weights='distance'`.
- Compare distance metric (`metric='euclidean'` vs `metric='manhattan'`).""")

add_code("""# TODO (Person 3): Implement k-NN on Base Features
""")

add_md("""### 6.2 k-Nearest Neighbors (k-NN) — Stage 2 (Augmented Features)
Train and evaluate the best $k$-NN configuration on `AUGMENTED_FEATURES` using the identical 5-fold CV splits.""")

add_code("""# TODO (Person 3): Implement k-NN on Augmented Features
""")

add_md("""### 6.3 Support Vector Classifier (SVC) — Stage 1 (Base Features)
Implement and tune SVC on `BASE_FEATURES`:
- Compare kernels: `kernel='linear'`, `kernel='rbf'`, `kernel='poly'`.
- Tune regularization parameter $C \in [0.1, 1.0, 10.0, 100.0]$ and $\gamma$ (`gamma`).
- Use `class_weight='balanced'` to account for minority classes (Orange & Red).
- Ensure probability calibration is enabled (`probability=True` or `CalibratedClassifierCV`).""")

add_code("""# TODO (Person 3): Implement SVC on Base Features
""")

add_md("""### 6.4 Support Vector Classifier (SVC) — Stage 2 (Augmented Features)
Train and evaluate the best SVC configuration on `AUGMENTED_FEATURES` using the identical 5-fold CV splits.""")

add_code("""# TODO (Person 3): Implement SVC on Augmented Features
""")

add_md("""### 6.5 Two-Stage Performance Comparison for Person 3 Models
Calculate the improvement gained by adding tectonic fault features:
$$\Delta \\text{Macro F1} = \\text{Macro F1}_{\\text{Augmented}} - \\text{Macro F1}_{\\text{Base}}$$
$$\Delta \\text{ROC-AUC} = \\text{ROC-AUC}_{\\text{Augmented}} - \\text{ROC-AUC}_{\\text{Base}}$$""")

add_code("""# TODO (Person 3): Compare Base vs Augmented performance (delta table and plot)
""")

add_md("""### 6.6 Out-of-Fold (OOF) Prediction Export & Model Serialization
1. Fit the final tuned $k$-NN and SVC models on the full training set and serialize to `artifacts/models/`:
   - `knn_aug.joblib`
   - `svm_rbf_aug.joblib`
2. Save out-of-fold cross-validated probability predictions for both $k$-NN and SVC.""")

add_code("""# TODO (Person 3): Export fitted models (.joblib) and generate OOF prediction vectors
""")

# Consolidated Section
add_md("""---
## 7. Final Consolidation: Module 4 Stacking Handoff
Combines OOF predictions from Person 1, Person 2, and Person 3 into `oof_train_m3.parquet` and `oof_test_m3.parquet` for Module 4.""")

add_code("""# --- Consolidation Cell: Merges OOF columns from all 3 roles once ready ---
""")

output_path = "ML_Project/ML_Project/module3/03_supervised_predictive_modelling.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {output_path}")
