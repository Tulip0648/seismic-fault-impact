# Module 2: Unsupervised Tectonic Pattern Discovery

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
     ┌───────────────────────┐
     │       MODULE 1        │
     │ Geospatial Transforms │
     │ & Base Train/Val/Test │
     └───────────┬───────────┘
                 │
                 │ Spatial Coordinates (x,y,z ECEF, lat/lon rad)
                 │ + Sensor PCA Features
                 ▼
     ┌───────────────────────┐
     │       MODULE 2        │ <--- (YOU ARE HERE)
     │ - DBSCAN (Fault Lines)│
     │ - GMM (EM Posteriors) │
     └───────────┬───────────┘
                 │
                 │ Augmented Dataset (Base + PCA + Fault Distances + GMM Posteriors)
                 │ & Unsupervised Metric Reports
                 ├──────────────────────────────────────┐
                 │                                      │
                 ▼                                      ▼
     ┌───────────────────────┐              ┌───────────────────────┐
     │       MODULE 3        │              │       MODULE 4        │
     │ Supervised Baselines  │              │ Advanced Models &     │
     │ (Augmented Benchmark) │              │ Stacking Ensemble     │
     └───────────────────────┘              └───────────────────────┘
```

### Upstream Dependencies:
- **Module 1 Outputs**:
  - `data/processed/train_base.parquet`
  - `data/processed/val_base.parquet`
  - `data/processed/test_base.parquet`
  - Required feature subsets:
    - Spatial: `latitude`, `longitude`, `depth`, `lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`
    - Sensors & PCA: `pca_sensor_1`, `pca_sensor_2`, `pca_sensor_3`, scaled sensor metrics

### Downstream Consumers:
- **Module 3 (Supervised Learning & Predictive Modelling)**:
  - Consumes `train_augmented.parquet` and `test_augmented.parquet` to evaluate whether linear, probabilistic, and kernel classifiers achieve higher Macro/Weighted F1 and ROC-AUC compared to the Base dataset.
- **Module 4 (Advanced Models, Comparative Evaluation & Model Integration)**:
  - Consumes the Augmented dataset to evaluate tree ensembles (Random Forest, LightGBM, XGBoost) and Neural Networks, and to generate SHAP feature attributions on fault features.

---

## 3. Tasks & Implementation Checklist

### Phase 1: Spherical & 3D DBSCAN Clustering
- [ ] **Task 1.1: Spherical Distance DBSCAN Implementation**
  - Implement DBSCAN using the `haversine` metric on `[lat_rad, lon_rad]` arrays.
  - Convert radius $\epsilon$ between kilometers and radians:
    $$\epsilon_{\text{rad}} = \frac{\epsilon_{\text{km}}}{R}, \quad R \approx 6371.0 \text{ km}$$
- [ ] **Task 1.2: 3D ECEF Cartesian DBSCAN Implementation**
  - Implement DBSCAN using Euclidean distance on 3D Cartesian space `[x_ecef, y_ecef, z_ecef]` to naturally capture hypocentral depth alongside surface geography.
- [ ] **Task 1.3: Hyperparameter Optimization for DBSCAN**
  - Generate k-nearest neighbors distance plot (k-distance graph) to detect the knee/elbow point for optimal $\epsilon$.
  - Sweep `min_samples` $\in [5, 10, 20, 50]$ to balance fault continuity vs noise rejection.
  - Label `-1` designates background tremors isolated from major fault lines.
- [ ] **Task 1.4: Inductive Out-of-Sample DBSCAN Predictor**
  - Standard scikit-learn DBSCAN has no `predict()` method for unseen data (`val` / `test`).
  - Implement `InductiveDBSCAN`:
    - Fit exclusively on `train`.
    - Predict unseen points in `val`/`test` by finding the nearest core point: if distance $\le \epsilon$, assign that core point's cluster; otherwise label as noise (`-1`).
    - Compute Euclidean / Great-Circle distance to the nearest cluster core point (`dist_to_fault_cluster`).

### Phase 2: Expectation-Maximization Gaussian Mixture Modeling (GMM)
- [ ] **Task 2.1: Model Formulation**
  - Model seismic hypocenter distributions as a mixture of $K$ Gaussians in 3D hypocentral coordinates (`x_ecef`, `y_ecef`, `z_ecef`):
    $$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
  - Test covariance types: `'full'` (unconstrained ellipsoid faults) vs `'diag'`.
- [ ] **Task 2.2: Optimal Component Selection via BIC & AIC**
  - Sweep number of mixture components $K \in [5, 10, 15, 20, 30, 40, 50]$.
  - Compute Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC):
    $$\text{BIC} = -2 \ln \hat{L} + p \ln(N)$$
    $$\text{AIC} = -2 \ln \hat{L} + 2p$$
  - Select optimal $K^*$ minimizing BIC/AIC and plot model selection curves.
- [ ] **Task 2.3: EM Posterior Probability Extraction**
  - Compute soft responsibility vectors for each earthquake:
    $$\gamma_{ik} = P(Z_i = k \mid \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$
  - Compute sample log-likelihood / anomaly scores: $\ln p(\mathbf{x}_i)$.

### Phase 3: Unsupervised Cluster Evaluation & Geological Validation
- [ ] **Task 3.1: Metric Benchmarking**
  - Calculate **Silhouette Score** on core cluster points (excluding noise).
  - Calculate **Davies-Bouldin Index** (lower indicates better cluster separation).
  - Export metric comparison table to `reports/clustering_validation_report.md`.
- [ ] **Task 3.2: Spatial & Geological Alignment**
  - Plot discovered clusters on global 2D map (Plate Carrée / Robinson projection) and 3D globe.
  - (Optional / Recommended) Overlay clusters with Peter Bird's global plate boundary dataset (PB2002) to verify whether discovered clusters align with recognized subduction zones and transform faults.

### Phase 4: Feature Coupling & Augmented Dataset Generation
- [ ] **Task 4.1: Feature Synthesis**
  - Combine:
    1. Base features from Module 1 (`train_base`, `val_base`, `test_base`).
    2. PCA components from Module 1.
    3. DBSCAN features: `dbscan_cluster_id`, `is_background_noise` (binary flag), `dist_to_nearest_fault_km`.
    4. GMM features: `gmm_cluster_id` (argmax cluster), posterior probabilities `gmm_prob_0` through `gmm_prob_{K-1}`, and `gmm_log_likelihood`.
- [ ] **Task 4.2: Data Leakage Prevention Check**
  - Ensure DBSCAN and GMM are fitted ONLY on `train_base.parquet`.
  - Transform `val_base.parquet` and `test_base.parquet` using the fitted estimators.
- [ ] **Task 4.3: Export Augmented Partitions**
  - Save:
    - `data/augmented/train_augmented.parquet`
    - `data/augmented/val_augmented.parquet`
    - `data/augmented/test_augmented.parquet`
  - Export fitted models to `artifacts/dbscan_model.joblib` and `artifacts/gmm_model.joblib`.

---

## 4. Expected Deliverables

```
module2/
├── README.md                          # Module summary
├── TODO.md                            # This specification file
├── data/
│   └── augmented/                     # Exported Coupled/Augmented datasets
│       ├── train_augmented.parquet
│       ├── val_augmented.parquet
│       └── test_augmented.parquet
├── artifacts/
│   ├── dbscan_model.joblib            # Fitted InductiveDBSCAN object
│   ├── gmm_model.joblib               # Fitted GMM model
│   └── bic_aic_curves.json            # BIC/AIC scores across swept components
├── reports/
│   ├── clustering_validation_report.md# Silhouette, Davies-Bouldin, BIC, AIC report
│   └── figures/                       # K-distance curve, BIC/AIC plot, 2D/3D fault maps
├── src/
│   ├── __init__.py
│   ├── dbscan_clustering.py           # Spherical & 3D Cartesian DBSCAN + Inductive predictor
│   ├── gmm_clustering.py              # GMM with BIC/AIC component selection
│   ├── cluster_evaluation.py          # Silhouette, Davies-Bouldin, BIC/AIC metrics
│   ├── augment_pipeline.py            # Joins Base + PCA + M2 features
│   └── visualize_faults.py            # Global map & 3D hypocenter cluster plotting
└── tests/
    ├── __init__.py
    ├── test_dbscan.py                 # Out-of-sample prediction and distance tests
    ├── test_gmm.py                    # Posterior probability sum-to-1 checks
    └── test_augmentation.py           # Integration test verifying schema & zero data leakage
```

---

## 5. Module API & Interface Specification

### 5.1 Python API

```python
# module2/src/dbscan_clustering.py
import numpy as np
from typing import Dict, Any

class InductiveDBSCAN:
    """
    Spherical / 3D DBSCAN with inductive prediction and distance calculations
    for unseen validation and test points.
    """
    def __init__(self, eps_km: float = 100.0, min_samples: int = 10, metric: str = "euclidean"):
        self.eps_km = eps_km
        self.min_samples = min_samples
        self.metric = metric
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "InductiveDBSCAN":
        """Fits DBSCAN on spatial coordinates (e.g., ECEF x, y, z or lat/lon in radians)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns cluster labels to unseen samples based on distance to nearest core point.
        Returns -1 for points farther than eps_km.
        """
        ...

    def transform_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array of shape (N, 2):
        - Column 0: Minimum distance in km to nearest fault cluster core point.
        - Column 1: Binary indicator is_noise (1 if noise, 0 otherwise).
        """
        ...
```

```python
# module2/src/gmm_clustering.py
from typing import Dict, List, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture

class FaultGMM:
    """
    Expectation-Maximization Gaussian Mixture Model for hypocentral distribution modeling.
    """
    def __init__(self, n_components: int = 20, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def tune_components(
        cls, X: np.ndarray, k_range: range, covariance_type: str = "full"
    ) -> Tuple[int, Dict[str, List[float]]]:
        """
        Sweeps K components, evaluates BIC and AIC, and returns (best_k, metric_history).
        """
        ...

    def fit(self, X: np.ndarray) -> "FaultGMM":
        """Fits GMM using Expectation-Maximization on training spatial coordinates."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns EM posterior responsibility matrix gamma of shape (N, K)."""
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns log-likelihood of each sample under the mixture model."""
        ...
```

```python
# module2/src/augment_pipeline.py
import pandas as pd
from typing import Optional

class AugmentationPipeline:
    """
    Joins Base features with DBSCAN fault metrics and GMM posterior probabilities.
    """
    def __init__(self, dbscan_model: InductiveDBSCAN, gmm_model: FaultGMM):
        self.dbscan = dbscan_model
        self.gmm = gmm_model

    def fit(self, df_base_train: pd.DataFrame) -> "AugmentationPipeline":
        """Fits both DBSCAN and GMM on training split coordinates only."""
        ...

    def transform(self, df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms any partition (train, val, or test) into the Augmented schema.
        Appends:
        - dbscan_cluster
        - is_background_noise
        - dist_to_nearest_fault_km
        - gmm_cluster
        - gmm_log_likelihood
        - gmm_prob_0 ... gmm_prob_{K-1}
        """
        ...
```

### 5.2 DataFrame Schema Contract (`train_augmented.parquet`, etc.)

The augmented dataset must contain **all columns from Module 1's Base dataset**, plus the newly engineered fault geometry columns:

| Column Name | Type | Description |
|---|---|---|
| *(All Module 1 Base columns)* | *(Same)* | `event_id`, `latitude`, `longitude`, `depth`, `mag`, sensor features, `pca_sensor_*`, `mmi`, `alert` |
| `dbscan_cluster` | `int32` | Discovered fault line cluster ID (`-1` = background noise) |
| `is_background_noise` | `int32` | Binary flag (`1` if DBSCAN noise / isolated tremor, `0` if on active fault) |
| `dist_to_nearest_fault_km` | `float64` | Distance in kilometers to nearest cluster core point |
| `gmm_cluster` | `int32` | Argmax GMM mixture component assignment |
| `gmm_log_likelihood` | `float64` | Mixture density log-likelihood score |
| `gmm_prob_0` ... `gmm_prob_{K-1}` | `float64` | Posterior probabilities $P(\text{fault } k \mid \text{event})$ summing to $1.0$ |

### 5.3 CLI Interface

```bash
# Run full unsupervised clustering, model selection, and dataset augmentation
python -m module2.src.augment_pipeline \
    --base-dir ../module1/data/processed/ \
    --output-dir data/augmented/ \
    --artifacts-dir artifacts/ \
    --k-min 5 \
    --k-max 35 \
    --dbscan-eps-km 120.0 \
    --dbscan-min-samples 15
```

---

## 6. Validation & Quality Checks
1. **EM Posterior Probability Invariant**: For every row $i$, ensure $\sum_{k=0}^{K-1} \text{gmm\_prob\_k} \approx 1.0 \pm 10^{-5}$ and all values $\ge 0$.
2. **Noise Isolation Rate**: Verify that DBSCAN isolates between $5\%$ and $20\%$ of events as background noise (tremors occurring away from major plate boundaries).
3. **No Target Leakage**: Ensure clustering models receive ONLY spatial and sensor features (`x_ecef`, `y_ecef`, `z_ecef`, `depth`, sensor PCA) — never the target labels (`alert` or `mmi`).
4. **Unsupervised Metric Standards**:
   - Compute Silhouette Score on non-noise samples ($\ge 0.35$ expected on well-defined regional belts).
   - Compute Davies-Bouldin Index (aim for lower values).
   - Document BIC/AIC curve showing a clear minimum or inflection point.

