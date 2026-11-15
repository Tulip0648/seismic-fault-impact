# Module 2: Unsupervised Tectonic Pattern Discovery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module2/02_unsupervised_fault_discovery.ipynb)

## 1. Overview & Objectives
Module 2 is responsible for **unsupervised tectonic fault discovery and feature augmentation**. Standard tabular models treat coordinates as flat Euclidean planes and remain oblivious to the underlying tectonic plate boundaries and fault networks that govern seismic wave propagation and destructive ground shaking.

Module 2 identifies tectonic fault boundaries and isolates background tremors using:
1. **Spherical Distance Clustering (DBSCAN)**: Discovers arbitrary-shaped fault lines while isolating random background tremors as noise.
2. **Expectation-Maximization Gaussian Mixture Models (GMM)**: Softly clusters earthquakes in 3D hypocentral space and estimates probabilistic fault membership.
3. **Cluster Validation**: Evaluates cluster separation and goodness of fit using Silhouette Score, Davies-Bouldin Index, BIC, and AIC.
4. **Feature Coupling (The "Augmented" Dataset)**: Augments the Module 1 Base features with cluster distances, noise flags, and EM posterior probabilities. This augmented representation is passed to Modules 3 and 4 to power the project's **Two-Stage Evaluation Protocol**.

---

## 2. Dependency Graph & Data Flow

### Overall Pipeline Position:
```
     ┌───────────────────────────────────┐
     │             MODULE 1              │
     │ 01_geospatial_preprocessing.ipynb │
     └─────────────────┬─────────────────┘
                       │
                       │ Spatial Coordinates (x,y,z ECEF, lat/lon rad)
                       │ + Sensor PCA Features
                       ▼
     ┌───────────────────────────────────┐
     │             MODULE 2              │ <--- (YOU ARE HERE)
     │ 02_unsupervised_fault_            │
     │    discovery.ipynb                │
     │ - DBSCAN (Fault Lines)            │
     │ - GMM (EM Posteriors)             │
     └─────────────────┬─────────────────┘
                       │
                       │ Augmented Dataset (Base + PCA + Fault Distances + GMM Posteriors)
                       │ & Unsupervised Metric Reports
                       ├──────────────────────────────────────┐
                       │                                      │
                       ▼                                      ▼
     ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
     │             MODULE 3              │  │             MODULE 4              │
     │ 03_supervised_predictive_         │  │ 04_advanced_models_and_           │
     │    modelling.ipynb                │  │    integration.ipynb              │
     │ (Augmented Benchmark)             │  │ (Advanced Models & Stacking)      │
     └───────────────────────────────────┘  └───────────────────────────────────┘
```

### Upstream Dependencies:
- **Module 1 Deliverables**:
  - `data/processed/train_base.parquet`
  - `data/processed/val_base.parquet`
  - `data/processed/test_base.parquet`

### Downstream Consumers:
- **Module 3 Notebook (`03_supervised_predictive_modelling.ipynb`)**:
  - Consumes `train_augmented.parquet` and `test_augmented.parquet` to evaluate whether linear, probabilistic, and kernel classifiers achieve higher Macro/Weighted F1 and ROC-AUC compared to the Base dataset.
- **Module 4 Notebook (`04_advanced_models_and_integration.ipynb`)**:
  - Consumes the Augmented dataset to evaluate tree ensembles (Random Forest, LightGBM, XGBoost) and Neural Networks, and to generate SHAP feature attributions on fault features.

---

## 3. Environment Setup & Notebook Execution (Colab & Local)

### 3.1 Running in Google Colab
Open directly using the Colab badge or via `File -> Open notebook -> GitHub` selecting `Tulip0648/seismic-fault-impact` and choosing `module2/02_unsupervised_fault_discovery.ipynb`.

Paste this initialization cell at the top of the notebook:

```python
# Cell 1: Colab / Local Environment Setup
import os, sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab. Setting up workspace...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    REPO_NAME = "seismic-fault-impact"
    REPO_PATH = f"/content/{REPO_NAME}"
    if not os.path.exists(REPO_PATH):
        !git clone https://github.com/Tulip0648/seismic-fault-impact.git {REPO_PATH}
    
    %cd {REPO_PATH}/module2
    !pip install -r ../requirements.txt -q
    DATA_BASE_DIR = "../data/processed"
    DATA_AUG_DIR = "../data/augmented"
    ARTIFACTS_DIR = "./artifacts"
else:
    print("Running locally.")
    DATA_BASE_DIR = "../module1/data/processed"
    DATA_AUG_DIR = "./data/augmented"
    ARTIFACTS_DIR = "./artifacts"

os.makedirs(DATA_AUG_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"Base data path: {DATA_BASE_DIR}")
print(f"Augmented data path: {DATA_AUG_DIR}")
```

### 3.2 Running in Local Jupyter
```bash
cd seismic-fault-impact/module2
# Launch JupyterLab or open 02_unsupervised_fault_discovery.ipynb in VS Code
jupyter lab
```

---

## 4. Notebook Structure & Implementation Tasks

The notebook `02_unsupervised_fault_discovery.ipynb` should be organized into the following sequential sections:

- [ ] **Section 1: Data Loading & Spatial Inspection**
  - Load `train_base.parquet`, `val_base.parquet`, `test_base.parquet` generated by Module 1.
  - Verify presence of coordinates: `lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`, `depth`.

- [ ] **Section 2: Spherical & 3D DBSCAN Clustering**
  - Compute k-nearest neighbors distance graph (k-distance plot) on spatial features to identify the knee/elbow point for $\epsilon$.
  - Fit `DBSCAN(metric='haversine')` on spherical coordinates or `DBSCAN(metric='euclidean')` on 3D ECEF Cartesian space.
  - Set `min_samples` (e.g. 10 to 20) to balance fault continuity vs noise isolation.
  - Outliers labeled as `-1` designate background tremors isolated from major fault lines.

- [ ] **Section 3: Inductive Out-of-Sample DBSCAN Assignment**
  - Implement an out-of-sample predictor:
    - Fit DBSCAN **only on the training split**.
    - For `val` and `test` points, compute distance to the nearest core sample.
    - If distance $\le \epsilon$, assign the corresponding cluster ID; otherwise assign `-1` (noise).
    - Compute continuous feature: `dist_to_nearest_fault_km`.

- [ ] **Section 4: Expectation-Maximization Gaussian Mixture Modeling (GMM)**
  - Model seismic hypocenters in 3D ECEF coordinate space:
    $$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
  - Sweep $K \in [5, 10, 15, 20, 25, 30, 40]$ and evaluate BIC and AIC:
    $$\text{BIC} = -2 \ln \hat{L} + p \ln(N), \quad \text{AIC} = -2 \ln \hat{L} + 2p$$
  - Plot BIC and AIC curves; select the optimal $K^*$ minimizing information criteria.
  - Extract EM posterior responsibilities $P(\text{fault } k \mid \text{event})$ and log-likelihood density scores.

- [ ] **Section 5: Unsupervised Cluster Evaluation & Geological Validation**
  - Calculate **Silhouette Score** on core cluster points (excluding noise).
  - Calculate **Davies-Bouldin Index** (lower indicates superior cluster separation).
  - Plot discovered clusters on global 2D maps and 3D Earth visualizations.
  - (Optional) Compare cluster alignments against known tectonic plate boundary datasets (e.g. Peter Bird PB2002).

- [ ] **Section 6: Feature Coupling & Augmented Dataset Export**
  - Synthesize the **Augmented Dataset**:
    - All columns from Module 1's Base dataset
    - `dbscan_cluster`: Discovered fault line ID
    - `is_background_noise`: Binary indicator (`1` if isolated tremor, `0` if on active fault)
    - `dist_to_nearest_fault_km`: Orthogonal distance to fault line in km
    - `gmm_cluster`: Argmax mixture component
    - `gmm_log_likelihood`: Hypocentral density score
    - `gmm_prob_0` ... `gmm_prob_{K-1}`: EM posterior probability vector summing to $1.0$
  - Save:
    - `data/augmented/train_augmented.parquet`
    - `data/augmented/val_augmented.parquet`
    - `data/augmented/test_augmented.parquet`
  - Export fitted models: `artifacts/dbscan_model.joblib` and `artifacts/gmm_model.joblib`.

---

## 5. Expected Deliverables

```
module2/
├── 02_unsupervised_fault_discovery.ipynb  # Primary self-contained Jupyter notebook
├── TODO.md                                # This specification file
├── data/
│   └── augmented/                         # Exported Coupled/Augmented partitions
│       ├── train_augmented.parquet
│       ├── val_augmented.parquet
│       └── test_augmented.parquet
├── artifacts/
│   ├── dbscan_model.joblib                # Fitted DBSCAN model with core samples
│   ├── gmm_model.joblib                   # Fitted GMM model
│   └── bic_aic_curves.json                # BIC/AIC sweep scores
└── reports/
    ├── clustering_validation_report.md    # Silhouette, Davies-Bouldin, BIC/AIC report
    └── figures/                           # K-distance curve, BIC/AIC curves, 2D/3D fault maps
```

---

## 6. Notebook Inter-Module API & Data Contract

Modules 3 and 4 load the exported Augmented datasets directly:

### 6.1 Data Contract (`train_augmented.parquet`, etc.)

```python
import pandas as pd

# Downstream notebooks (Modules 3 and 4) load:
df_train_aug = pd.read_parquet("data/augmented/train_augmented.parquet")
print(df_train_aug.shape)
```

| Column Name | Type | Description |
|---|---|---|
| *(All Module 1 Base columns)* | *(Same)* | `event_id`, `latitude`, `longitude`, `depth`, `mag`, sensor features, `pca_sensor_*`, `mmi`, `alert` |
| `dbscan_cluster` | `int32` | Discovered fault line cluster ID (`-1` = background noise) |
| `is_background_noise` | `int32` | Binary flag (`1` if DBSCAN noise / isolated tremor, `0` if on active fault) |
| `dist_to_nearest_fault_km` | `float64` | Distance in kilometers to nearest cluster core point |
| `gmm_cluster` | `int32` | Argmax GMM mixture component assignment |
| `gmm_log_likelihood` | `float64` | Mixture density log-likelihood score |
| `gmm_prob_0` ... `gmm_prob_{K-1}` | `float64` | Posterior probabilities $P(\text{fault } k \mid \text{event})$ summing to $1.0$ |

### 6.2 Pre-trained Clusterer Re-use

```python
import joblib

# Inference pipeline uses:
dbscan = joblib.load("module2/artifacts/dbscan_model.joblib")
gmm = joblib.load("module2/artifacts/gmm_model.joblib")
```

---

## 7. Validation & Sanity Checklist
1. **Probability Invariant**: Ensure $\sum_{k=0}^{K-1} \text{gmm\_prob\_k} \approx 1.0 \pm 10^{-5}$ across all rows.
2. **Background Tremor Ratio**: Verify that DBSCAN isolates between $5\%$ and $20\%$ of events as background noise.
3. **No Target Leakage**: Ensure clustering models receive ONLY spatial coordinates and sensor PCA features — never target labels (`alert` or `mmi`).
4. **Reproducibility Test**: Execute **Kernel -> Restart Kernel and Run All Cells**; verify clean end-to-end execution.
