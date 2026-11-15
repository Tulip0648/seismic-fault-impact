# Module 4: Advanced Models, Comparative Evaluation & Model Integration

## 1. Overview & Objectives
Module 4 represents the culminating modeling, evaluation, and system integration phase of the project. It builds upon the foundational baselines of Module 3 by developing high-capacity non-linear models, conducting a global comparative evaluation, integrating diverse models via ensemble learning and stacking, and providing the unified end-to-end inference API.

### Core Objectives:
1. **Advanced Model Development**: Implement tree-based ensembles (Random Forest, Extra Trees), gradient boosted decision trees (LightGBM, XGBoost, CatBoost), and deep neural architectures (Multi-Layer Perceptron with residual connections and regularization).
2. **Two-Stage Evaluation on Advanced Architectures**: Evaluate all advanced models on both the **Base** and **Augmented** datasets to determine whether non-linear learners benefit from explicit unsupervised tectonic features (fault distances, GMM posteriors) or if their complexity renders these features redundant.
3. **Model Integration & Stacking**: Combine complementary model paradigms (linear, kernel, tree ensembles, and neural nets) using weighted soft-voting and Level-1 stacking meta-learners.
4. **Global Comparative Leaderboard & Hypothesis Testing**: Construct the definitive benchmark comparing all models across Module 3 and Module 4; perform statistical significance tests (paired Wilcoxon / $5\times 2$ CV $t$-tests) to validate improvements.
5. **Explainability & Feature Attribution (SHAP)**: Apply TreeSHAP and KernelSHAP to understand the real-world geological and sensor drivers of earthquake impact predictions.
6. **Unified End-to-End Inference Pipeline**: Provide an integrated deployment interface that takes raw earthquake sensor inputs, transforms them through Modules 1 & 2, and produces final impact severity predictions.

---

## 2. Dependency Graph & Data Flow

### Overall Pipeline Position:
```
     ┌───────────────────────┐              ┌───────────────────────┐
     │       MODULE 1        │              │       MODULE 2        │
     │ Base Splits &         │              │ Augmented Splits &    │
     │ Fitted Preprocessor   │              │ Fitted Clusterers     │
     └───────────┬───────────┘              └───────────┬───────────┘
                 │                                      │
                 │                                      │
                 ├──────────────────┬───────────────────┤
                 │                  │                   │
                 ▼                  ▼                   │
     ┌───────────────────────┐      │                   │
     │       MODULE 3        │      │                   │
     │ Baselines & Out-of-   │      │                   │
     │ Fold (OOF) Probs      │      │                   │
     └───────────┬───────────┘      │                   │
                 │                  │                   │
                 │ M3 OOF Probs     │ Base & Augmented  │
                 │ & Baseline Scores│ Datasets          │
                 ▼                  ▼                   ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                           MODULE 4                           │ <--- (YOU ARE HERE)
     │ - Gradient Boosters (LightGBM, XGBoost, CatBoost)            │
     │ - Tree Ensembles (Random Forest, Extra Trees)                │
     │ - Deep Neural Network (MLP / Residual Tabular Net)           │
     │ - Stacking Meta-Learner (M3 + M4 Integrator)                 │
     │ - Global Leaderboard & Statistical Significance Testing      │
     │ - SHAP Attribution Analysis                                  │
     │ - End-to-End Production Inference Pipeline                   │
     └──────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
                      Final Project Deliverables:
          End-to-End Predictor, Benchmark Leaderboard, Report
```

### Upstream Dependencies:
1. **Module 1**:
   - `module1/data/processed/train_base.parquet`, `test_base.parquet`
   - `module1/artifacts/preprocessor.joblib`, `pca_model.joblib`
2. **Module 2**:
   - `module2/data/augmented/train_augmented.parquet`, `test_augmented.parquet`
   - `module2/artifacts/dbscan_model.joblib`, `gmm_model.joblib`
3. **Module 3**:
   - `module3/artifacts/oof_predictions/oof_train_m3.parquet`, `oof_test_m3.parquet`
   - `module3/reports/baseline_benchmarks.md`

### Downstream Impact:
Module 4 outputs the final artifacts, research conclusions, comparative tables, and interactive inference engine that fulfill the project proposal deliverables for course evaluation.

---

## 3. Tasks & Implementation Checklist

### Phase 1: Advanced Model Implementation & Tuning
- [ ] **Task 1.1: Tree-Based Ensembles**
  - Implement `RandomForestClassifier` and `ExtraTreesClassifier`.
  - Hyperparameter tuning: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`, and `class_weight='balanced_subsample'`.
- [ ] **Task 1.2: Gradient Boosted Decision Trees (GBDT)**
  - Implement **LightGBM** (`LGBMClassifier`): Fast histogram-based gradient boosting with leaf-wise splitting.
  - Implement **XGBoost** (`XGBClassifier`): Exact/approximate tree boosting with depth-wise splitting and Hessian regularization.
  - Implement **CatBoost** (`CatBoostClassifier`): Ordered boosting with robust handling of categorical sensor features (`magType`).
  - Optimization: Use Optuna or Bayesian Optimization to tune learning rate, max depth, subsample ratio, colsample_bytree, and L1/L2 regularization.
- [ ] **Task 1.3: Deep Neural Networks for Tabular Data**
  - Implement a PyTorch (or scikit-learn / Keras) Multi-Layer Perceptron (MLP) architecture:
    - Input dimension: $D$ (Base or Augmented feature count).
    - Architecture: 3-4 dense layers with Batch Normalization, Dropout ($p=0.2-0.4$), and LeakyReLU / GELU activations.
    - Residual skip connections (ResNet-style tabular block).
    - Loss function: Cross-Entropy Loss with class weighting or Focal Loss to handle severe class imbalance.
    - Optimizer: AdamW with Cosine Annealing learning rate schedule.

### Phase 2: Two-Stage Evaluation of Advanced Models
- [ ] **Task 2.1: Stage 1 (Base Dataset Benchmarking)**
  - Train all advanced models on `train_base.parquet` and evaluate on `test_base.parquet`.
  - Record Macro F1, Weighted F1, Multi-class ROC-AUC, and log loss.
- [ ] **Task 2.2: Stage 2 (Augmented Dataset Benchmarking)**
  - Train all advanced models on `train_augmented.parquet` and evaluate on `test_augmented.parquet`.
  - Quantify performance deltas ($\Delta \text{Macro F1}$, $\Delta \text{ROC-AUC}$).
  - Document findings: Do gradient boosters extract fault patterns on their own, or does explicit tectonic domain features (M2) provide statistically significant lifts?

### Phase 3: Model Integration & Ensembling
- [ ] **Task 3.1: Soft-Voting Ensemble**
  - Implement `WeightedVotingClassifier` combining top diverse models (e.g. LightGBM, Random Forest, SVM, MLP).
  - Optimize ensemble weights via Nelder-Mead optimization on the validation split.
- [ ] **Task 3.2: Stacking Meta-Learner (Level-1 Ensemble)**
  - Collect out-of-fold probability predictions from:
    - Module 3 Level-0 models (Logistic Regression, Kernel SVM, $k$-NN, LDA).
    - Module 4 Level-0 models (Random Forest, LightGBM, XGBoost, MLP).
  - Train a regularized meta-learner (e.g., Logistic Regression with L2 penalty or shallow Gradient Booster) on the concatenated probability matrix.
  - Evaluate stacked ensemble on the test set.

### Phase 4: Comprehensive System Benchmarking & Statistical Verification
- [ ] **Task 4.1: Unified Global Comparative Leaderboard**
  - Combine all evaluation records from Module 3 (Linear, Probabilistic, KNN, SVM) and Module 4 (RF, GBDT, MLP, Voting, Stacking) across both Base and Augmented datasets into a master comparison table.
  - Highlight best-in-class models by metric.
- [ ] **Task 4.2: Statistical Significance Testing**
  - Conduct paired Wilcoxon signed-rank tests or $5\times 2$ cross-validated paired $t$-tests on fold-level Macro F1 scores between Base and Augmented variants.
  - Report $p$-values to formally establish statistical significance of the unsupervised fault discovery module.

### Phase 5: Interpretability & Feature Attribution (SHAP)
- [ ] **Task 5.1: TreeSHAP Analysis on Best GBDT Model**
  - Compute SHAP values for test samples using `shap.TreeExplainer`.
  - Generate:
    - Global feature importance summary beeswarm plot.
    - SHAP dependence plots for `dist_to_nearest_fault_km` and top GMM posterior probability columns (`gmm_prob_*`).
- [ ] **Task 5.2: Geological Impact Profiling Narrative**
  - Interpret how fault proximity and hypocentral clustering influence transition thresholds between alert levels (`green` $\rightarrow$ `yellow` $\rightarrow$ `orange` $\rightarrow$ `red`).

### Phase 6: End-to-End Production Pipeline
- [ ] **Task 6.1: Unified Predictor Class**
  - Implement `EarthquakeImpactPipeline`:
    $$\text{Raw Event JSON/Dict} \xrightarrow{\text{M1 Preprocessor}} \text{Base Features} \xrightarrow{\text{M2 Clusterers}} \text{Augmented Features} \xrightarrow{\text{M4 Final Model}} \text{Impact Prediction}$$
- [ ] **Task 6.2: Serialization & Smoke Testing**
  - Package all required transformer and model weights into `artifacts/`.
  - Verify that a raw earthquake record produces a valid prediction and probability distribution in $< 50\text{ ms}$.

---

## 4. Expected Deliverables

```
module4/
├── README.md                          # Module summary
├── TODO.md                            # This specification file
├── artifacts/
│   ├── best_lightgbm_aug.joblib       # Tuned LightGBM model
│   ├── best_xgboost_aug.joblib        # Tuned XGBoost model
│   ├── best_rf_aug.joblib             # Tuned Random Forest model
│   ├── best_mlp_aug.pt                # PyTorch / Keras neural network weights
│   ├── stacking_meta_learner.joblib   # Level-1 stacking model
│   └── end_to_end_pipeline.joblib     # Full production pipeline artifact
├── reports/
│   ├── final_comparative_leaderboard.md # Master leaderboard (M3 + M4, Base vs Aug)
│   ├── statistical_significance.md    # Hypothesis testing results & p-values
│   ├── shap_explainability_report.md  # SHAP plots and feature impact analysis
│   └── figures/                       # Beeswarm plots, dependence curves, ROC overlays
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tree_ensembles.py          # Random Forest & Extra Trees
│   │   ├── gradient_boosting.py       # LightGBM, XGBoost, CatBoost
│   │   └── neural_network.py          # PyTorch Tabular ResNet / MLP
│   ├── ensembling/
│   │   ├── __init__.py
│   │   ├── voting_classifier.py       # Weighted soft-voting integrator
│   │   └── stacking_ensemble.py       # Level-1 stacking meta-learner
│   ├── evaluation/
│   │   ├── global_benchmarker.py      # Aggregates M3 + M4 metrics into master leaderboard
│   │   └── hypothesis_testing.py      # Paired t-tests & Wilcoxon tests
│   ├── explainability.py              # SHAP summary & dependence plotting
│   └── inference_pipeline.py          # Unified end-to-end predictor
└── tests/
    ├── __init__.py
    ├── test_advanced_models.py        # Unit tests for GBDT & MLP training
    ├── test_stacking.py               # Stacking meta-learner tests
    └── test_inference_pipeline.py     # End-to-end prediction from raw dict test
```

---

## 5. Module API & Interface Specification

### 5.1 Python API

```python
# module4/src/ensembling/stacking_ensemble.py
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Two-tier Stacking Classifier that fits Level-1 meta-estimator
    on out-of-fold predicted class probabilities of Level-0 models.
    """
    def __init__(self, base_models: List[Any], meta_learner: Any, use_features: bool = False):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.use_features = use_features  # If True, concatenates raw features with predicted probabilities

    def fit(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> "StackingEnsemble":
        """Generates OOF probabilities for Level-0 models and fits meta-learner."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated class probabilities."""
        ...
```

```python
# module4/src/inference_pipeline.py
from typing import Dict, Any, Union
import pandas as pd
import numpy as np

class EarthquakeImpactPipeline:
    """
    Complete end-to-end inference engine:
    Raw Earthquake Data -> Module 1 -> Module 2 -> Module 4 Ensemble -> Impact Prediction.
    """
    def __init__(
        self,
        m1_preprocessor_path: str,
        m2_augmenter_path: str,
        m4_model_path: str
    ):
        self.m1_preprocessor = None
        self.m2_augmenter = None
        self.m4_model = None
        self._load_artifacts(m1_preprocessor_path, m2_augmenter_path, m4_model_path)

    def _load_artifacts(self, m1_path: str, m2_path: str, m4_path: str) -> None:
        ...

    def predict_single(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts raw earthquake dictionary:
        {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "depth": 12.3,
            "mag": 6.2,
            "magType": "mw",
            "nst": 45,
            "gap": 32.0,
            "dmin": 0.12,
            "rms": 0.25,
            ...
        }
        Returns:
        {
            "predicted_alert": "yellow",
            "probabilities": {"green": 0.08, "yellow": 0.74, "orange": 0.15, "red": 0.03},
            "nearest_fault_distance_km": 8.4,
            "tectonic_cluster_id": 4,
            "is_background_noise": False
        }
        """
        ...

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Runs end-to-end batch inference on raw DataFrame."""
        ...
```

```python
# module4/src/explainability.py
import shap
import pandas as pd
from typing import List

class EarthquakeExplainer:
    """
    TreeSHAP and KernelSHAP attribution analyzer for earthquake impact predictions.
    """
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit_explainer(self, background_data: np.ndarray) -> None:
        ...

    def generate_summary_plot(self, X_sample: np.ndarray, output_path: str) -> None:
        """Saves SHAP beeswarm plot ranking top sensor and fault features."""
        ...

    def generate_dependence_plot(self, X_sample: np.ndarray, feature_name: str, output_path: str) -> None:
        """Plots non-linear relationship between fault distance / GMM probability and impact."""
        ...
```

### 5.2 Global Leaderboard Schema Contract (`final_comparative_leaderboard.md`)

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

### 5.3 CLI Interface

```bash
# Run training, ensembling, global benchmarking, and SHAP explainability
python -m module4.src.benchmark_runner \
    --m1-dir ../module1/data/processed/ \
    --m2-dir ../module2/data/augmented/ \
    --m3-oof-dir ../module3/artifacts/oof_predictions/ \
    --artifacts-dir artifacts/ \
    --reports-dir reports/

# Run inference on a test JSON payload
python -m module4.src.inference_pipeline \
    --pipeline-artifacts artifacts/end_to_end_pipeline.joblib \
    --input sample_event.json
```

---

## 6. Validation & Quality Checks
1. **Two-Stage Improvement Benchmark**: Verify whether the best Augmented model shows a statistically significant gain ($p < 0.05$) in Macro F1 over the best Base model.
2. **Stacking Integrity**: Ensure Level-1 meta-learner is trained strictly on out-of-fold predictions to prevent severe meta-overfitting.
3. **Inference Latency Sanity Check**: Single-event prediction latency must remain under $100\text{ ms}$ for real-time seismic alert utility.
4. **SHAP Consistency Check**: Verify that `shap_values.sum(axis=-1) + base_value` reconstructs the model's raw margin output.

