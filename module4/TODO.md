# Module 4: Advanced Models, Comparative Evaluation & Model Integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module4/04_advanced_models_and_integration.ipynb)

## 1. Overview & Objectives
Module 4 represents the culminating modeling, evaluation, and system integration phase of the project. It builds upon the foundational baselines of Module 3 by developing high-capacity non-linear models, conducting a global comparative evaluation, integrating diverse models via ensemble learning and stacking, and providing the unified end-to-end inference API.

### Core Objectives:
1. **Advanced Model Development**: Implement tree-based ensembles (Random Forest, Extra Trees), gradient boosted decision trees (LightGBM, XGBoost, CatBoost), and deep neural architectures (Multi-Layer Perceptron with residual connections and regularization).
2. **Two-Stage Evaluation on Advanced Architectures**: Evaluate all advanced models on both the **Base** and **Augmented** datasets to determine whether non-linear learners benefit from explicit unsupervised tectonic features (fault distances, GMM posteriors) or if their complexity renders these features redundant.
3. **Model Integration & Stacking**: Combine complementary model paradigms (linear, kernel, tree ensembles, and neural nets) using weighted soft-voting and Level-1 stacking meta-learners.
4. **Global Comparative Leaderboard & Hypothesis Testing**: Construct the definitive benchmark comparing all models across Module 3 and Module 4; perform statistical significance tests (paired Wilcoxon / $5\times 2$ CV $t$-tests) to validate improvements.
5. **Explainability & Feature Attribution (SHAP)**: Apply TreeSHAP and KernelSHAP to understand the real-world geological and sensor drivers of earthquake impact predictions.
6. **Unified End-to-End Inference Pipeline**: Provide an interactive deployment cell and exportable pipeline class that takes raw earthquake sensor inputs, transforms them through Modules 1 & 2, and produces final impact severity predictions.

---

## 2. Dependency Graph & Data Flow

### Overall Pipeline Position:
```
     ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
     │             MODULE 1              │  │             MODULE 2              │
     │ 01_geospatial_preprocessing.ipynb │  │ 02_unsupervised_fault_            │
     │ (Base Splits & Preprocessor)      │  │    discovery.ipynb (Augmented)    │
     └─────────────────┬─────────────────┘  └─────────────────┬─────────────────┘
                       │                                      │
                       ├──────────────────┬───────────────────┤
                       │                  │                   │
                       ▼                  ▼                   │
     ┌───────────────────────────────────┐│                   │
     │             MODULE 3              ││                   │
     │ 03_supervised_predictive_         ││                   │
     │    modelling.ipynb                ││                   │
     │ (Baselines & OOF Probs)           ││                   │
     └─────────────────┬─────────────────┘│                   │
                       │                  │                   │
                       │ M3 OOF Probs     │ Base & Augmented  │
                       │ & Baseline Scores│ Datasets          │
                       ▼                  ▼                   ▼
     ┌──────────────────────────────────────────────────────────────────────────┐
     │                                 MODULE 4                                 │ <--- (YOU ARE HERE)
     │ 04_advanced_models_and_integration.ipynb                                 │
     │ - Gradient Boosters (LightGBM, XGBoost, CatBoost)                        │
     │ - Tree Ensembles (Random Forest, Extra Trees)                            │
     │ - Deep Neural Network (PyTorch Tabular ResNet / MLP)                     │
     │ - Stacking Meta-Learner (M3 + M4 Integrator)                             │
     │ - Global Leaderboard & Statistical Significance Testing                  │
     │ - SHAP Attribution Analysis                                              │
     │ - End-to-End Interactive Inference Pipeline                              │
     └─────────────────────────────────────┬────────────────────────────────────┘
                                           │
                                           ▼
                             Final Project Deliverables:
                 Interactive Predictor, Benchmark Leaderboard, Report
```

### Upstream Dependencies:
1. **Module 1**: `data/processed/train_base.parquet`, `test_base.parquet`, `artifacts/preprocessor.joblib`
2. **Module 2**: `data/augmented/train_augmented.parquet`, `test_augmented.parquet`, `artifacts/dbscan_model.joblib`, `artifacts/gmm_model.joblib`
3. **Module 3**: `module3/artifacts/oof_predictions/oof_train_m3.parquet`, `oof_test_m3.parquet`

---

## 3. Environment Setup & Notebook Execution (Colab & Local)

### 3.1 Running in Google Colab (GPU Recommended)
Because Module 4 trains LightGBM, XGBoost, CatBoost, and PyTorch Neural Networks, using a GPU runtime is strongly recommended:
* In Colab, go to **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU**.

Paste this initialization cell at the top of `04_advanced_models_and_integration.ipynb`:

```python
# Cell 1: Colab / Local Setup with GPU Acceleration Check
import os, sys, torch

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab. Setting up workspace...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    REPO_NAME = "seismic-fault-impact"
    REPO_PATH = f"/content/{REPO_NAME}"
    if not os.path.exists(REPO_PATH):
        !git clone https://github.com/Tulip0648/seismic-fault-impact.git {REPO_PATH}
    
    %cd {REPO_PATH}/module4
    !pip install -r ../requirements.txt -q
    DATA_BASE_DIR = "../data/processed"
    DATA_AUG_DIR = "../data/augmented"
    M3_OOF_DIR = "../module3/artifacts/oof_predictions"
    ARTIFACTS_DIR = "./artifacts"
else:
    print("Running locally.")
    DATA_BASE_DIR = "../module1/data/processed"
    DATA_AUG_DIR = "../module2/data/augmented"
    M3_OOF_DIR = "../module3/artifacts/oof_predictions"
    ARTIFACTS_DIR = "./artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using device: {device}")
```

### 3.2 Running in Local Jupyter
```bash
cd seismic-fault-impact/module4
jupyter lab
```

---

## 4. Notebook Structure & Implementation Tasks

The notebook `04_advanced_models_and_integration.ipynb` should be organized into the following sequential sections:

- [ ] **Section 1: Data Ingestion & Alignment**
  - Load Base splits (`train_base`, `test_base`), Augmented splits (`train_augmented`, `test_augmented`), and Module 3 OOF probability matrices.
  - Assert that all row indices match across partitions.

- [ ] **Section 2: Advanced Tree Ensembles & Gradient Boosters**
  - Implement `RandomForestClassifier` and `ExtraTreesClassifier` with cost-sensitive subsample weighting.
  - Implement **LightGBM**, **XGBoost**, and **CatBoost**:
    - Tune learning rates, tree depth, regularization (L1/L2), and feature subsampling.
    - Train on GPU if available (`device='cuda'` or `tree_method='gpu_hist'`).

- [ ] **Section 3: Deep Neural Network (PyTorch Tabular ResNet / MLP)**
  - Implement a PyTorch Multi-Layer Perceptron architecture:
    - Dense blocks with Batch Normalization, Dropout ($p=0.3$), and LeakyReLU / GELU.
    - Tabular residual connections.
    - Class-weighted Cross-Entropy Loss or Focal Loss.
    - Optimizer: AdamW with Cosine Annealing learning rate schedule.

- [ ] **Section 4: Two-Stage Benchmarking of Advanced Models**
  - Train all advanced models on Base vs Augmented datasets.
  - Record Macro F1, Weighted F1, Multi-class ROC-AUC, and log loss.
  - Quantify performance deltas: Does explicit fault line proximity provide a statistically significant edge over raw coordinates for high-capacity gradient boosters?

- [ ] **Section 5: Model Integration & Stacking Ensembles**
  - **Soft Voting Classifier**: Weighted probability averaging across diverse models (e.g. LightGBM + Random Forest + SVM + MLP).
  - **Level-1 Stacking Meta-Learner**:
    - Concatenate OOF probability features from Module 3 (Logistic, SVM, $k$-NN, LDA) and Module 4 (Random Forest, LightGBM, MLP).
    - Train a regularized meta-learner (e.g., Logistic Regression L2 or shallow LightGBM) on the stacked probability matrix.
    - Evaluate stacked meta-learner on test partition.

- [ ] **Section 6: Master Comparative Leaderboard & Statistical Significance**
  - Compile the master evaluation table aggregating all models from Module 3 & Module 4 across Base and Augmented configurations.
  - Perform paired Wilcoxon signed-rank test or $5\times 2$ CV paired $t$-test on fold Macro F1 scores to prove whether unsupervised fault discovery yields statistically significant improvement ($p < 0.05$).

- [ ] **Section 7: Interpretability & Feature Attribution (SHAP)**
  - Run `shap.TreeExplainer` on the best gradient boosting model.
  - Generate:
    - Global feature importance beeswarm plot.
    - SHAP dependence plots for `dist_to_nearest_fault_km` and top GMM posterior probability columns (`gmm_prob_*`).
  - Summarize geological impact insights.

- [ ] **Section 8: Interactive Inference & Production Pipeline**
  - Implement `EarthquakeImpactPipeline` class:
    $$\text{Raw Event Dict} \xrightarrow{\text{M1 Preprocessor}} \text{Base Features} \xrightarrow{\text{M2 Clusterers}} \text{Augmented Features} \xrightarrow{\text{M4 Model}} \text{Prediction}$$
  - Add an interactive Jupyter widget (`ipywidgets`) allowing users to slide latitude, longitude, magnitude, and depth to visualize predicted PAGER alert level and probabilities live in the notebook.
  - Export final pipeline artifact to `artifacts/end_to_end_pipeline.joblib`.

---

## 5. Expected Deliverables

```
module4/
├── 04_advanced_models_and_integration.ipynb  # Primary self-contained Jupyter notebook
├── TODO.md                                   # This specification file
├── artifacts/
│   ├── best_lightgbm_aug.joblib              # Tuned LightGBM model
│   ├── best_xgboost_aug.joblib               # Tuned XGBoost model
│   ├── best_rf_aug.joblib                    # Tuned Random Forest model
│   ├── best_mlp_aug.pt                       # PyTorch neural network weights
│   ├── stacking_meta_learner.joblib          # Level-1 stacking model
│   └── end_to_end_pipeline.joblib            # Exported unified pipeline
└── reports/
    ├── final_comparative_leaderboard.md      # Master leaderboard table (M3 + M4)
    ├── statistical_significance.md           # Hypothesis tests & p-values
    ├── shap_explainability_report.md         # SHAP plots and feature impact analysis
    └── figures/                              # Beeswarm plots, dependence curves, ROC overlays
```

---

## 6. Notebook Inter-Module API & Inference Demo

### 6.1 End-to-End Pipeline Inference Class

Teammates or evaluators can run end-to-end inference using the exported pipeline:

```python
import joblib
import pandas as pd

# Load unified pipeline:
pipeline = joblib.load("module4/artifacts/end_to_end_pipeline.joblib")

# Sample raw earthquake query:
sample_event = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "depth": 12.3,
    "mag": 6.4,
    "magType": "mw",
    "nst": 42.0,
    "gap": 35.0,
    "dmin": 0.15,
    "rms": 0.22,
    "horizontalError": 0.8,
    "depthError": 1.2,
    "magError": 0.05
}

result = pipeline.predict_single(sample_event)
print("Predicted Alert:", result["predicted_alert"])
print("Class Probabilities:", result["probabilities"])
print("Nearest Fault Distance (km):", result["nearest_fault_distance_km"])
```

### 6.2 Master Global Leaderboard Schema (`final_comparative_leaderboard.md`)

| Model Family | Specific Model | Feature Set | Macro F1 | Weighted F1 | Multi-Class ROC-AUC | Log Loss | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Linear (M3) | Logistic Regression (L2) | Base | - | - | - | - | - |
| Linear (M3) | Logistic Regression (L2) | **Augmented** | - | - | - | - | - |
| Probabilistic (M3)| LDA | Base | - | - | - | - | - |
| Probabilistic (M3)| LDA | **Augmented** | - | - | - | - | - |
| Instance (M3) | $k$-NN ($k=7$) | Base | - | - | - | - | - |
| Instance (M3) | $k$-NN ($k=7$) | **Augmented** | - | - | - | - | - |
| Kernel (M3) | SVC (RBF) | Base | - | - | - | - | - |
| Kernel (M3) | SVC (RBF) | **Augmented** | - | - | - | - | - |
| Tree Ensemble (M4)| Random Forest | Base | - | - | - | - | - |
| Tree Ensemble (M4)| Random Forest | **Augmented** | - | - | - | - | - |
| Gradient Boost (M4)| LightGBM | Base | - | - | - | - | - |
| Gradient Boost (M4)| LightGBM | **Augmented** | - | - | - | - | - |
| Gradient Boost (M4)| XGBoost | Base | - | - | - | - | - |
| Gradient Boost (M4)| XGBoost | **Augmented** | - | - | - | - | - |
| Neural Network (M4)| Deep Tabular MLP | Base | - | - | - | - | - |
| Neural Network (M4)| Deep Tabular MLP | **Augmented** | - | - | - | - | - |
| **Ensemble (M4)** | **Stacking Classifier** | **Augmented** | - | - | - | - | - |

---

## 7. Validation & Sanity Checklist
1. **Statistical Significance Proof**: Verify that the paired $t$-test or Wilcoxon test achieves $p < 0.05$ between the best Augmented model and best Base model.
2. **Inference Latency Benchmark**: Verify that `predict_single` runs in $< 100\text{ ms}$ for real-time alert readiness.
3. **Reproducibility Test**: Execute **Kernel -> Restart Kernel and Run All Cells**; verify clean end-to-end execution.
