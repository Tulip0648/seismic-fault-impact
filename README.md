# Seismological Fault Line Discovery & Earthquake Impact Profiling

**Course**: BITSF464 Machine Learning — Fall 2026  
**Project Proposal**: [project_proposal.pdf](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/project_proposal.pdf)  

### Team Members
| Name | ID |
|---|---|
| Aaryan Pandey | 2024A7PS0375G |
| Rhea Jain | 2024A7PS0575G |
| Nivya Nandagopal | 2024ADPS1428G |
| Sahaj Jhamb | 2024A7PS0485G |
| Mridul Sardana | 2024A7PS0547G |
| Harshit Rawat | 2024AAPS0756G |
| Parshav Giya | 2024ADPS0852G |

---

## 1. Project Overview & Motivation
Standard tabular earthquake models treat latitude and longitude as flat Euclidean planes and ignore the underlying 3D tectonic fault geometry that dictates ground shaking.

This project implements a **4-module semi-supervised machine learning pipeline** that:
1. **Unsupervised Fault Discovery**: Identifies tectonic fault boundaries and isolates background tremors using spherical distance clustering (DBSCAN) and Expectation-Maximization Gaussian Mixture Models (GMM) in 3D hypocentral coordinates.
2. **Supervised Impact Prediction**: Predicts earthquake impact severity (PAGER multi-class alert level: green, yellow, orange, red; and continuous/categorical Modified Mercalli Intensity `mmi`) by coupling discovered tectonic fault features into linear, kernel, tree-based, and neural ensemble models.
3. **Two-Stage Evaluation Protocol**: Evaluates all supervised models across two dataset configurations under a fixed stratified split:
   - **Base (Baseline)**: Imputed and standardized raw sensor features + geospatial coordinates.
   - **Augmented (Coupled)**: Base features combined with sensor PCA components, DBSCAN fault distances, and GMM EM posterior probabilities.

---

## 2. Global Architecture & Dependency Graph

```
                            Raw USGS ComCat Catalog
                         (ShakeMap & PAGER Attributes)
                                       │
                                       ▼
             ┌───────────────────────────────────────────────────┐
             │                     MODULE 1                      │
             │   Geospatial Preprocessing & Feature Engineering  │
             │   - Data cleaning & MICE/Median Imputation        │
             │   - 3D ECEF & Spherical Radian Coordinate Calcs   │
             │   - Categorical Encoding & Sensor Robust Scaling  │
             │   - Sensor PCA (Dimensionality Reduction)         │
             │   - Fixed Stratified Train / Val / Test Splitting │
             └─────────────────────────┬─────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ Base Features (Train/Val/Test)              │ Spatial Coords (x,y,z,depth)
                │ & Stratified Splits                         │ & Sensor PCA
                ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│           MODULE 3            │             │           MODULE 2            │
│ Supervised Predictive Models  │             │ Unsupervised Fault Discovery  │
│ - Linear (Logistic L1/L2)     │             │ - Spherical & 3D DBSCAN       │
│ - Probabilistic (NB, LDA, QDA)│             │ - EM Gaussian Mixture Models  │
│ - Instance-based (k-NN)       │             │ - BIC / AIC Model Selection   │
│ - Kernel-based (SVM RBF/Poly) │             │ - Silhouette & Davies-Bouldin │
│                               │             │ - Fault Distance & Posteriors │
│ [Evaluates Stage 1: Base]     │             └───────────────┬───────────────┘
└───────────────┬───────────────┘                             │
                │                                             │ Augmented Features
                │                                             │ (Base + PCA + Fault Features)
                │ Baseline Results &                          ├───────────────────────────────┐
                │ Out-of-Fold (OOF) Probs                     │                               │
                │                                             ▼                               ▼
                │                               ┌───────────────────────────┐   ┌───────────────────────────┐
                │                               │         MODULE 3          │   │         MODULE 4          │
                └──────────────────────────────►│ [Evaluates Stage 2:       │   │ Advanced Models:          │
                                                │  Augmented Features]      │   │ - Tree Ensembles (RF)     │
                                                └───────────────────────────┘   │ - Boosters (LGBM, XGB)    │
                                                                                │ - Deep Tabular Neural Net │
                                                                                │ - Stacking Meta-Learner   │
                                                                                │ - Global Leaderboard      │
                                                                                │ - SHAP Attributions       │
                                                                                │ - End-to-End Pipeline     │
                                                                                └───────────────────────────┘
```

---

## 3. Module Specifications & TODOs

Each module contains a dedicated `TODO.md` defining its objectives, tasks, deliverables, API specifications, and validation rules:

| Module | Name | Primary Focus | Specification Link |
|---|---|---|---|
| **Module 1** | Geospatial Data Preprocessing & Feature Engineering | Ingestion, imputation, 3D ECEF/spherical transformation, PCA, stratified split | [module1/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module1/TODO.md) |
| **Module 2** | Unsupervised Tectonic Pattern Discovery | DBSCAN fault lines, EM GMM posterior probabilities, BIC/AIC, dataset augmentation | [module2/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module2/TODO.md) |
| **Module 3** | Supervised Learning & Predictive Modelling | Linear, probabilistic, k-NN, and SVM models; Two-Stage Base vs. Augmented evaluation; OOF predictions | [module3/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module3/TODO.md) |
| **Module 4** | Advanced Models, Evaluation & Integration | Random Forest, LightGBM, XGBoost, Neural Nets, Stacking Ensemble, SHAP, end-to-end inference API | [module4/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module4/TODO.md) |

---

## 4. Cross-Module Data Contracts

### 4.1 Base Dataset Contract (Module 1 $\rightarrow$ Modules 2, 3, 4)
- Files: `train_base.parquet`, `val_base.parquet`, `test_base.parquet`
- Contains:
  - Coordinate representations: `latitude`, `longitude`, `depth`, `lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`
  - Sensor & error metrics: `mag`, `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError` (imputed and standardized)
  - Categorical encodings: `magType_*`
  - Dimensionality reduction: `pca_sensor_1`, `pca_sensor_2`, `pca_sensor_3`
  - Targets: `alert` (`green`, `yellow`, `orange`, `red`), `mmi`

### 4.2 Augmented Dataset Contract (Module 2 $\rightarrow$ Modules 3, 4)
- Files: `train_augmented.parquet`, `val_augmented.parquet`, `test_augmented.parquet`
- Contains: All Base dataset columns plus:
  - `dbscan_cluster`: Discovered fault line ID (`-1` = background tremor)
  - `is_background_noise`: Binary indicator (`1` = isolated tremor, `0` = fault line event)
  - `dist_to_nearest_fault_km`: Orthogonal / geodesic distance in km to nearest fault cluster
  - `gmm_cluster`: Argmax mixture component
  - `gmm_log_likelihood`: Hypocentral density score
  - `gmm_prob_0` ... `gmm_prob_{K-1}`: Soft EM fault membership probabilities ($\sum_k p_k = 1.0$)

### 4.3 Stacking Contract (Module 3 $\rightarrow$ Module 4)
- Files: `oof_train_m3.parquet`, `oof_test_m3.parquet`
- Contains: Cross-validated out-of-fold probability distributions from Logistic Regression, LDA, $k$-NN, and Kernel SVM.

---

## 5. Execution Roadmap & Recommended Workflow
1. **Module 1**: Implement `src/spatial_transform.py` and `src/pipeline.py` to produce the Base splits (`data/processed/*_base.parquet`).
2. **Module 2**: In parallel or right after, fit DBSCAN and GMM on `train_base.parquet` to produce the Augmented splits (`data/augmented/*_augmented.parquet`).
3. **Module 3**: Train linear, probabilistic, $k$-NN, and SVM models on both Base and Augmented splits; log baseline metrics and export OOF predictions.
4. **Module 4**: Train Random Forest, LightGBM, XGBoost, and Tabular Neural Nets on Base and Augmented splits; fit Level-1 Stacking Meta-Learner; run SHAP analysis; package `EarthquakeImpactPipeline`.
