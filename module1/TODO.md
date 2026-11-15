# Module 1: Geospatial Data Preprocessing & Feature Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module1/01_geospatial_preprocessing.ipynb)

## 1. Overview & Objectives
Module 1 serves as the foundational data ingestion, cleaning, transformation, and feature engineering backbone for the entire **Seismological Fault Line Discovery & Earthquake Impact Profiling** project (BITSF464 Fall 2026).

Earthquake catalogs contain raw sensor measurements and spatial coordinates that cannot simply be treated as flat Euclidean tabular features. Latitude and longitude lie on an oblate spheroid / sphere, and sensor values feature heterogeneous errors, azimuthal gaps, and missing values.

### Core Objectives:
1. **Ingest & Clean**: Ingest USGS ComCat records filtered for ShakeMap and PAGER impact metrics, clean invalid records, and handle missing values through principled statistical/domain imputation.
2. **Geospatial Transformations**: Convert geographic coordinates (`latitude`, `longitude`, `depth`) into 3D Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates and spherical radian coordinates for downstream haversine and spatial clustering.
3. **Geophysical Feature Scaling & Encoding**: Scale continuous sensor errors and wave residuals, and encode categorical sensor metadata (`magType`).
4. **Dimensionality Reduction (PCA)**: Perform Principal Component Analysis on sensor and error features to capture dominant variance without collinearity.
5. **Two-Stage Split Partitioning**: Establish a strict, stratified train/validation/test split on the target variable (`alert` / `mmi`) to prevent data leakage and provide the fixed data splits for all downstream modules.
6. **Produce the Base Dataset**: Export the official **Base (Baseline)** feature splits and save preprocessing transformers for downstream notebooks.

---

## 2. Dependency Graph & Data Flow

### Overall Pipeline Position:
```
USGS ComCat Raw Data (ShakeMap + PAGER)
                 │
                 ▼
     ┌───────────────────────────────────┐
     │             MODULE 1              │ <--- (YOU ARE HERE)
     │ 01_geospatial_preprocessing.ipynb │
     └─────────────────┬─────────────────┘
                       │
                       ├──────────────────────────────────────┐
                       │ Base Features (Train/Val/Test)       │ Spatial & PCA Features
                       │ & Stratified Split Partitions        │
                       ▼                                      ▼
     ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
     │             MODULE 3              │  │             MODULE 2              │
     │ 03_supervised_predictive_         │  │ 02_unsupervised_fault_            │
     │    modelling.ipynb                │  │    discovery.ipynb                │
     │ (Base Evaluation)                 │  │ (Unsupervised Fault Discovery)    │
     └───────────────────────────────────┘  └───────────────────────────────────┘
```

### Upstream Dependencies:
- **Inputs**: Raw USGS ComCat catalog data (CSV, GeoJSON, or API query responses) containing:
  - Spatial: `latitude`, `longitude`, `depth`
  - Geophysical & Sensor: `mag`, `magType`, `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`
  - Impact Targets: `alert` (PAGER categorical severity: green, yellow, orange, red), `mmi` (ShakeMap continuous/discrete Modified Mercalli Intensity)

### Downstream Consumers:
- **Module 2 Notebook (`02_unsupervised_fault_discovery.ipynb`)**: Consumes spatial coordinates (`lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`, `depth`) and PCA components to discover tectonic fault clusters and compute posterior probabilities.
- **Module 3 Notebook (`03_supervised_predictive_modelling.ipynb`)**: Consumes the **Base** train/val/test splits to establish Stage 1 baseline performance (linear, probabilistic, KNN, kernel models).
- **Module 4 Notebook (`04_advanced_models_and_integration.ipynb`)**: Consumes the **Base** splits to benchmark gradient boosting, tree ensembles, and neural networks against the augmented features.

---

## 3. Environment Setup & Notebook Execution (Colab & Local)

### 3.1 Running in Google Colab
Every teammate can open the notebook directly using the Colab badge above or by navigating to `File -> Open notebook -> GitHub` in Colab and selecting `Tulip0648/seismic-fault-impact`.

Paste this universal initialization cell at the very top of `01_geospatial_preprocessing.ipynb`:

```python
# Cell 1: Colab / Local Environment Setup
import os, sys

# Detect if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab. Setting up workspace...")
    # Mount Google Drive for persistent artifact and dataset storage
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Clone or update the GitHub repo in Colab
    REPO_NAME = "seismic-fault-impact"
    REPO_PATH = f"/content/{REPO_NAME}"
    if not os.path.exists(REPO_PATH):
        !git clone https://github.com/Tulip0648/seismic-fault-impact.git {REPO_PATH}
    
    %cd {REPO_PATH}/module1
    !pip install -r ../requirements.txt -q
    DATA_DIR = "../data"
    ARTIFACTS_DIR = "./artifacts"
else:
    print("Running locally.")
    DATA_DIR = "./data"
    ARTIFACTS_DIR = "./artifacts"

os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"Workspace ready. Data directory: {DATA_DIR}")
```

### 3.2 Running in Local Jupyter (JupyterLab / VS Code)
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Tulip0648/seismic-fault-impact.git
   cd seismic-fault-impact
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter lab
   # Or open module1/01_geospatial_preprocessing.ipynb directly in VS Code
   ```

---

## 4. Notebook Structure & Implementation Tasks

The notebook `01_geospatial_preprocessing.ipynb` should be organized into the following sequential cells:

- [x] **Section 1: Data Ingestion & Schema Audit**
  - Fetch/load raw USGS ComCat records with ShakeMap and PAGER attributes.
  - Assert valid physical ranges: $\text{latitude} \in [-90, 90]$, $\text{longitude} \in [-180, 180]$, $\text{depth} \ge 0$, $\text{magnitude} > 0$.
  - Clean corrupted or synthetic test events.
  - Standardize target column `alert` (`green`, `yellow`, `orange`, `red`) and continuous `mmi`.

- [x] **Section 2: Missing Value Profiling & Imputation**
  - Plot missing value bar chart / matrix for `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`.
  - Fit `SimpleImputer(strategy='median')` or `IterativeImputer` **exclusively on the training split** to strictly prevent data leakage.
  - Include missingness indicator columns (`MissingIndicator`).

- [x] **Section 3: Geospatial Coordinate Transformations**
  - Convert angles from degrees to radians:
    $$\phi = \text{lat} \times \frac{\pi}{180}, \quad \lambda = \text{lon} \times \frac{\pi}{180}$$
  - Calculate 3D Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates:
    $$x = (R - \text{depth}) \cos(\phi) \cos(\lambda)$$
    $$y = (R - \text{depth}) \cos(\phi) \sin(\lambda)$$
    $$z = (R - \text{depth}) \sin(\phi)$$
    where $R = 6371.0 \text{ km}$.
  - Generate an interactive 3D scatter plot of hypocenters inside the Earth using Plotly or Matplotlib.

- [x] **Section 4: Categorical Encoding & Sensor Feature Scaling**
  - One-hot encode `magType` (`OneHotEncoder(handle_unknown='ignore')`).
  - Apply `RobustScaler` to skewed sensor and error metrics (`gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`).

- [x] **Section 5: Dimensionality Reduction (Sensor PCA)**
  - Fit `PCA` on scaled sensor and error features.
  - Plot the cumulative explained variance ratio curve.
  - Select components explaining $\ge 85\%$ variance (typically 2 to 4 components) and append as `pca_sensor_1`, `pca_sensor_2`, etc.

- [x] **Section 6: Stratified Train / Val / Test Partitioning**
  - Perform stratified split (70% train, 15% validation, 15% test) stratified on target variable `alert`.
  - Fix random seed (`random_state=42`).
  - Verify identical class proportions across train, val, and test partitions.

- [x] **Section 7: Artifact Serialization & Handoff**
  - Export data partitions as high-performance Parquet files:
    - `data/processed/train_base.parquet`
    - `data/processed/val_base.parquet`
    - `data/processed/test_base.parquet`
  - Export fitted transformer pipelines:
    - `artifacts/preprocessor.joblib`
    - `artifacts/pca_model.joblib`
    - `artifacts/split_indices.json`

---

## 5. Expected Deliverables

```
module1/
├── 01_geospatial_preprocessing.ipynb  # Primary self-contained Jupyter notebook
├── TODO.md                            # This specification file
├── data/
│   ├── raw/                           # Raw ComCat data
│   └── processed/                     # Exported Base partitions
│       ├── train_base.parquet
│       ├── val_base.parquet
│       └── test_base.parquet
├── artifacts/
│   ├── preprocessor.joblib            # Fitted Scikit-Learn transformer
│   ├── pca_model.joblib               # Fitted PCA model
│   └── split_indices.json             # Stratified indices
└── reports/
    ├── eda_and_feature_report.md      # Summary of distributions & correlations
    └── figures/                       # Missingness plots, PCA scree plots, 3D hypocenters
```

---

## 6. Notebook Inter-Module API & Data Contract

When teammates work across notebooks, they do not need to rerun previous notebooks. Instead, they consume the serialized data contracts produced by Module 1:

### 6.1 Data Contract (`train_base.parquet`, `val_base.parquet`, `test_base.parquet`)

```python
import pandas as pd

# Downstream notebooks load Module 1 outputs directly:
df_train = pd.read_parquet("data/processed/train_base.parquet")
print(df_train.shape, df_train.columns.tolist())
```

| Column Name | Type | Description |
|---|---|---|
| `event_id` | `str` | Unique USGS event identifier |
| `latitude` | `float64` | Raw latitude in degrees |
| `longitude` | `float64` | Raw longitude in degrees |
| `depth` | `float64` | Focal hypocentral depth in km |
| `lat_rad` | `float64` | Latitude in radians (for Haversine metric in M2) |
| `lon_rad` | `float64` | Longitude in radians (for Haversine metric in M2) |
| `x_ecef` | `float64` | Earth-Centered Earth-Fixed X in km |
| `y_ecef` | `float64` | Earth-Centered Earth-Fixed Y in km |
| `z_ecef` | `float64` | Earth-Centered Earth-Fixed Z in km |
| `mag` | `float64` | Earthquake magnitude |
| `magType_<type>` | `float64` | One-hot encoded magnitude types |
| `nst` | `float64` | Number of seismic stations (imputed & scaled) |
| `gap` | `float64` | Azimuthal gap in degrees (scaled) |
| `dmin` | `float64` | Distance to nearest station (scaled) |
| `rms` | `float64` | Wave travel time residual in seconds (scaled) |
| `horizontalError`| `float64` | Location horizontal error in km (scaled) |
| `depthError` | `float64` | Depth error in km (scaled) |
| `magError` | `float64` | Magnitude estimation error (scaled) |
| `pca_sensor_1` | `float64` | 1st Principal Component of sensor features |
| `pca_sensor_2` | `float64` | 2nd Principal Component of sensor features |
| `pca_sensor_3` | `float64` | 3rd Principal Component of sensor features |
| `mmi` | `float64` | Modified Mercalli Intensity |
| `alert` | `str` / `int`| PAGER impact level (`green`, `yellow`, `orange`, `red`) — **Target** |

### 6.2 Preprocessor Re-use Interface

```python
import joblib

# To transform unseen test data during inference:
preprocessor = joblib.load("module1/artifacts/preprocessor.joblib")
pca_model = joblib.load("module1/artifacts/pca_model.joblib")
```

---

## 7. Validation & Sanity Checklist
1. **Zero Data Leakage**: Verify that scalers, imputers, and PCA matrices are computed solely from `train_base.parquet` and applied without re-fitting on `val` and `test`.
2. **Stratification Quality**: Verify that `alert` class proportions match across train, val, and test splits within $\pm 0.5\%$.
3. **Reproducibility Test**: In Jupyter/Colab, select **Kernel -> Restart Kernel and Run All Cells**; ensure the entire notebook executes without error from top to bottom.
