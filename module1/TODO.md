# Module 1: Geospatial Data Preprocessing & Feature Engineering

## 1. Overview & Objectives
Module 1 serves as the foundational data ingestion, cleaning, transformation, and feature engineering backbone for the entire **Seismological Fault Line Discovery & Earthquake Impact Profiling** project (BITSF464 Fall 2026).

Earthquake catalogs contain raw sensor measurements and spatial coordinates that cannot simply be treated as flat Euclidean tabular features. Latitude and longitude lie on an oblate spheroid / sphere, and sensor values feature heterogeneous errors, azimuthal gaps, and missing values.

### Core Objectives:
1. **Ingest & Clean**: Ingest USGS ComCat records filtered for ShakeMap and PAGER impact metrics, clean invalid records, and handle missing values through principled statistical/domain imputation.
2. **Geospatial Transformations**: Convert geographic coordinates (`latitude`, `longitude`, `depth`) into 3D Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates and spherical radian coordinates for downstream haversine and spatial clustering.
3. **Geophysical Feature Scaling & Encoding**: Scale continuous sensor errors and wave residuals, and encode categorical sensor metadata (`magType`).
4. **Dimensionality Reduction (PCA)**: Perform Principal Component Analysis on sensor and error features to capture dominant variance without collinearity.
5. **Two-Stage Split Partitioning**: Establish a strict, stratified train/validation/test split on the target variable (`alert` / `mmi`) to prevent data leakage and provide the fixed data splits for all downstream modules.
6. **Produce the Base Dataset**: Export the official **Base (Baseline)** feature splits and save preprocessing transformers for inference.

---

## 2. Dependency Graph & Data Flow

### Overall Pipeline Position:
```
USGS ComCat Raw Data (ShakeMap + PAGER)
                 │
                 ▼
     ┌───────────────────────┐
     │       MODULE 1        │ <--- (YOU ARE HERE)
     │ Geospatial & Sensors  │
     └───────────┬───────────┘
                 │
                 ├──────────────────────────────────────┐
                 │ Base Features (Train/Val/Test)       │ Spatial & PCA Features
                 │ & Stratified Split Partitions        │
                 ▼                                      ▼
     ┌───────────────────────┐              ┌───────────────────────┐
     │       MODULE 3        │              │       MODULE 2        │
     │ Supervised Baselines  │              │ Unsupervised Fault    │
     │ (Base Evaluation)     │              │ Discovery (Clusters)  │
     └───────────────────────┘              └───────────────────────┘
```

### Upstream Dependencies:
- **Inputs**: Raw USGS ComCat catalog data (CSV, GeoJSON, or API query responses) containing:
  - Spatial: `latitude`, `longitude`, `depth`
  - Geophysical & Sensor: `mag`, `magType`, `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`
  - Impact Targets: `alert` (PAGER categorical severity: green, yellow, orange, red), `mmi` (ShakeMap continuous/discrete Modified Mercalli Intensity)

### Downstream Consumers:
- **Module 2 (Unsupervised Fault Discovery)**: Consumes spatial coordinates (`lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`, `depth`) and PCA components to discover tectonic fault clusters and compute posterior probabilities.
- **Module 3 (Supervised Baselines)**: Consumes the **Base** train/val/test splits to establish Stage 1 baseline performance (linear, probabilistic, KNN, kernel models).
- **Module 4 (Advanced Models & Integration)**: Consumes the **Base** splits to benchmark gradient boosting, tree ensembles, and neural networks against the augmented features.

---

## 3. Tasks & Implementation Checklist

### Phase 1: Data Ingestion & Quality Audit
- [ ] **Task 1.1: Data Acquisition Script**
  - Implement USGS ComCat query fetcher or local raw data loader (`src/ingestion.py`).
  - Filter events with valid ShakeMap (`mmi`) or PAGER (`alert`) target fields.
- [ ] **Task 1.2: Data Sanity & Range Validation**
  - Check latitude $\in [-90, 90]$, longitude $\in [-180, 180]$, depth $\ge 0$, magnitude $> 0$.
  - Identify corrupted or outlier records (e.g. negative depth artifacts, synthetic test events).
- [ ] **Task 1.3: Target Formatting & Class Definition**
  - Standardize `alert` level classes: `['green', 'yellow', 'orange', 'red']` or map into ordinal integers.
  - Retain `mmi` as continuous target variable for optional regression benchmarking.

### Phase 2: Missing Value Imputation
- [ ] **Task 2.1: Missingness Analysis**
  - Profile missing value rates for `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`.
- [ ] **Task 2.2: Imputation Pipeline**
  - Implement and compare:
    - Domain median imputation with missing indicator flags (`MissingIndicator`).
    - Iterative Imputer (MICE) or KNN Imputer for sensor errors.
  - **Critical Rule**: Imputers MUST be fit exclusively on the `train` split and applied to `val` and `test` to prevent data leakage.

### Phase 3: Geospatial Coordinate Transformations
- [ ] **Task 3.1: Spherical Radian Conversion**
  - Convert `latitude` and `longitude` from degrees to radians:
    $$\phi = \text{lat} \times \frac{\pi}{180}, \quad \lambda = \text{lon} \times \frac{\pi}{180}$$
  - Needed for spherical distance algorithms (e.g., Haversine metric in DBSCAN).
- [ ] **Task 3.2: 3D Earth-Centered Earth-Fixed (ECEF) Conversion**
  - Convert coordinates to 3D Cartesian coordinates incorporating hypocentral depth:
    $$x = (R - \text{depth}) \cdot \cos(\phi) \cdot \cos(\lambda)$$
    $$y = (R - \text{depth}) \cdot \cos(\phi) \cdot \sin(\lambda)$$
    $$z = (R - \text{depth}) \cdot \sin(\phi)$$
    where $R \approx 6371.0 \text{ km}$ (mean Earth radius).
  - This allows Euclidean distance calculations that respect true 3D spatial proximity inside the crust.

### Phase 4: Feature Scaling & Encoding
- [ ] **Task 4.1: Categorical Encoding**
  - Encode `magType` (e.g., `mb`, `mw`, `ml`, `ms`) using `OneHotEncoder(handle_unknown='ignore')`.
- [ ] **Task 4.2: Continuous Feature Normalization**
  - Apply `RobustScaler` or `StandardScaler` to skewed sensor measurements (`gap`, `dmin`, `rms`, error metrics).
  - Preserve physical units where needed for interpretability.

### Phase 5: Dimensionality Reduction (PCA)
- [ ] **Task 5.1: Sensor Feature PCA**
  - Fit `PCA` on sensor/error features (`nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError`).
  - Determine optimal number of components explaining $\ge 85\%$ variance (typically 2 to 4 components).
  - Generate component columns: `pca_sensor_1`, `pca_sensor_2`, etc.
  - Save PCA loading matrix and explained variance plots.

### Phase 6: Fixed Stratified Split Protocol
- [ ] **Task 6.1: Stratified Splitter**
  - Implement fixed train/val/test splitting (e.g., 70% train, 15% validation, 15% test) stratified on `alert` level.
  - Seed: `random_state=42`.
- [ ] **Task 6.2: Partition Serialization**
  - Save output partitions as Parquet files:
    - `data/processed/train_base.parquet`
    - `data/processed/val_base.parquet`
    - `data/processed/test_base.parquet`
  - Export split indices (`train_idx.npy`, `val_idx.npy`, `test_idx.npy`) for verification across modules.

### Phase 7: Pipeline Export & Documentation
- [ ] **Task 7.1: Serialized Transformer Artifacts**
  - Export full fitted pipeline (`artifacts/preprocessor.joblib`, `artifacts/pca_model.joblib`).
- [ ] **Task 7.2: EDA & Feature Documentation Report**
  - Generate automated summary report in `reports/eda_and_feature_report.md`.

---

## 4. Expected Deliverables

```
module1/
├── README.md                      # Module summary
├── TODO.md                        # This specification file
├── data/
│   ├── raw/                       # Raw USGS ComCat records (.csv or .parquet)
│   └── processed/                 # Generated Base partitions
│       ├── train_base.parquet
│       ├── val_base.parquet
│       └── test_base.parquet
├── artifacts/
│   ├── preprocessor.joblib        # Fitted Scikit-Learn ColumnTransformer / Pipeline
│   ├── pca_model.joblib           # Fitted PCA model
│   └── split_indices.json         # Array of train/val/test index identifiers
├── reports/
│   ├── eda_and_feature_report.md  # Data audit, distributions, correlation analysis
│   └── figures/                   # Missingness matrices, PCA variance curve, correlation heatmaps
├── src/
│   ├── __init__.py
│   ├── ingestion.py               # USGS API / ComCat file loader
│   ├── cleaning.py                # Range checks, missing value imputation
│   ├── spatial_transform.py       # ECEF and spherical radian calculations
│   ├── feature_engineering.py     # Categorical encoding, scaling, PCA extraction
│   └── pipeline.py                # Unified end-to-end preprocessing pipeline & CLI
└── tests/
    ├── __init__.py
    ├── test_spatial_transform.py  # Unit tests for ECEF / radian formulas
    ├── test_cleaning.py           # Unit tests for imputation & leakage checks
    └── test_pipeline.py           # End-to-end integration test
```

---

## 5. Module API & Interface Specification

Module 1 must expose both a Python programmatic API and a command-line interface (CLI).

### 5.1 Python API

```python
# module1/src/spatial_transform.py
from typing import Tuple
import numpy as np
import pandas as pd

class GeospatialTransformer:
    """
    Transforms lat/lon/depth coordinates into ECEF Cartesian (x, y, z) 
    and spherical radians for clustering.
    """
    def __init__(self, earth_radius_km: float = 6371.0):
        self.earth_radius_km = earth_radius_km

    def transform_ecef(self, lat: np.ndarray, lon: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (x, y, z) in kilometers."""
        ...

    def transform_radians(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lat_rad, lon_rad) in radians."""
        ...

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends ['x_ecef', 'y_ecef', 'z_ecef', 'lat_rad', 'lon_rad'] to DataFrame."""
        ...
```

```python
# module1/src/pipeline.py
from typing import Dict, Tuple, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EarthquakeDataPipeline:
    """
    End-to-end preprocessing, feature engineering, and splitting pipeline.
    """
    def __init__(
        self,
        imputer_strategy: str = "median",
        scaler_type: str = "robust",
        n_pca_components: int = 3,
        random_state: int = 42
    ):
        ...

    def fit(self, df_train: pd.DataFrame, target_col: str = "alert") -> "EarthquakeDataPipeline":
        """Fits imputation, scalers, encoders, and PCA on training data only."""
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw input DataFrame into Base feature format."""
        ...

    def fit_transform(self, df_train: pd.DataFrame, target_col: str = "alert") -> pd.DataFrame:
        """Fits and transforms training partition."""
        ...

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "alert",
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Splits raw dataset into stratified train, val, and test subsets.
        Returns dict with keys: 'train', 'val', 'test'.
        """
        ...

    def save(self, filepath: str) -> None:
        """Serializes the fitted pipeline to disk using joblib."""
        ...

    @classmethod
    def load(cls, filepath: str) -> "EarthquakeDataPipeline":
        """Loads a pre-trained pipeline artifact from disk."""
        ...
```

### 5.2 DataFrame Schema Contract (`train_base.parquet`, `val_base.parquet`, `test_base.parquet`)

| Column Name | Type | Description |
|---|---|---|
| `event_id` | `str` | Unique USGS event identifier |
| `latitude` | `float64` | Raw latitude in degrees |
| `longitude` | `float64` | Raw longitude in degrees |
| `depth` | `float64` | Focal hypocentral depth in km |
| `lat_rad` | `float64` | Latitude in radians (for Haversine DBSCAN) |
| `lon_rad` | `float64` | Longitude in radians (for Haversine DBSCAN) |
| `x_ecef` | `float64` | Earth-Centered Earth-Fixed X in km |
| `y_ecef` | `float64` | Earth-Centered Earth-Fixed Y in km |
| `z_ecef` | `float64` | Earth-Centered Earth-Fixed Z in km |
| `mag` | `float64` | Earthquake magnitude |
| `magType_<type>` | `float64` | One-hot encoded magnitude types |
| `nst` | `float64` | Number of seismic stations (imputed & scaled) |
| `gap` | `float64` | Azimuthal gap in degrees (scaled) |
| `dmin` | `float64` | Horizontal distance to nearest station (scaled) |
| `rms` | `float64` | Wave travel time residual in seconds (scaled) |
| `horizontalError`| `float64` | Location horizontal error in km (scaled) |
| `depthError` | `float64` | Hypocentral depth error in km (scaled) |
| `magError` | `float64` | Magnitude estimation error (scaled) |
| `pca_sensor_1` | `float64` | 1st Principal Component of sensor features |
| `pca_sensor_2` | `float64` | 2nd Principal Component of sensor features |
| `pca_sensor_3` | `float64` | 3rd Principal Component of sensor features |
| `mmi` | `float64` | Modified Mercalli Intensity (ShakeMap ground shaking) |
| `alert` | `str` / `int`| PAGER impact severity level (`green`, `yellow`, `orange`, `red`) - **Primary Target** |

### 5.3 CLI Interface

```bash
# Run full preprocessing and splitting pipeline
python -m module1.src.pipeline \
    --input data/raw/usgs_comcat.csv \
    --output-dir data/processed/ \
    --artifacts-dir artifacts/ \
    --target alert \
    --test-size 0.15 \
    --val-size 0.15 \
    --pca-components 3
```

---

## 6. Validation & Quality Checks
1. **Data Leakage Check**: Verify that scalers, imputers, and PCA matrices are computed solely from `train_base.parquet` and applied without re-fitting on `val` and `test`.
2. **Stratification Check**: Compute class distribution ratios of `alert` across `train`, `val`, and `test`; ensure proportions match within $\pm 0.5\%$.
3. **Spatial Transform Sanity**:
   - For an equator point $(\text{lat}=0, \text{lon}=0, \text{depth}=0)$, $x = 6371.0, y = 0, z = 0$.
   - For the North Pole $(\text{lat}=90, \text{lon}=0, \text{depth}=0)$, $x = 0, y = 0, z = 6371.0$.
4. **Unit Tests**: Pass all tests in `tests/` with $\ge 90\%$ code coverage.

