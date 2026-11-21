# Module 3 Team Work Plan

## Overview
Module 3 trains and compares baseline machine learning models to predict earthquake alert levels (`alert`: green, yellow, orange, red) and shaking intensity (`mmi`).

Every model is trained and tested on two datasets using the exact same stratified 5-fold cross-validation:
1. **Base dataset**: Raw physical and sensor features from Module 1.
2. **Augmented dataset**: Base features + fault distance and 20 GMM probabilities from Module 2.

Target metrics for all models: Macro F1-score, Weighted F1-score, Multi-class ROC-AUC (One-vs-Rest), and Confusion Matrices.

---

## 3-Person Task Breakdown

### Person 1: Linear & Regularized Models
**Goal**: Build and evaluate linear models.

- **Tasks**:
  1. Set up the stratified 5-fold cross-validation split and target encoding.
  2. Implement **Logistic Regression** for `alert` classification.
     - Test L1 penalty (Lasso), L2 penalty (Ridge), and ElasticNet.
     - Tune regularization strength $C$ and solvers (`lbfgs`, `saga`).
     - Apply `class_weight='balanced'` to handle class imbalance.
  3. Implement **Ridge and Lasso Regression** to predict continuous ground shaking (`mmi`).
  4. Measure feature weights to see which features matter most.
- **Deliverables**:
  - `artifacts/models/logistic_regression_aug.joblib`
  - Cross-validated out-of-fold probability predictions for Logistic Regression.

---

### Person 2: Probabilistic & Discriminant Models
**Goal**: Build and evaluate Bayesian and distribution-based classifiers.

- **Tasks**:
  1. Implement **Gaussian Naive Bayes**.
     - Tune `var_smoothing` to prevent zero-variance errors.
  2. Implement **Linear Discriminant Analysis (LDA)**.
     - Test solvers (SVD, Eigenvalue decomposition).
     - Test covariance shrinkage (`shrinkage='auto'`).
  3. Implement **Quadratic Discriminant Analysis (QDA)**.
     - Tune `reg_param` so covariance matrices do not become singular on small classes.
  4. Compare linear decision boundaries (LDA) vs curved boundaries (QDA).
- **Deliverables**:
  - `artifacts/models/lda_aug.joblib`
  - `artifacts/models/qda_aug.joblib`
  - Cross-validated out-of-fold probability predictions for LDA.

---

### Person 3 (Aaryan): Support Vector Classifiers (SVC) & k-Nearest Neighbors (k-NN)
**Goal**: Build, tune, and evaluate distance-based and boundary-margin models.

- **Tasks**:
  1. Implement **Support Vector Classifier (SVC)**:
     - Test kernels: Linear, RBF (Radial Basis Function), and Polynomial.
     - Tune regularization $C$ and kernel parameter $\gamma$ (`gamma`).
     - Enable probability calibration (`CalibratedClassifierCV` or `probability=True`) so the model outputs probabilities that sum to 1.0 for each event.
     - Apply `class_weight='balanced'`.
  2. Implement **k-Nearest Neighbors (k-NN)**:
     - Test values of $k \in [3, 5, 7, 11, 15, 21]$.
     - Compare weighting: `weights='uniform'` vs `weights='distance'`.
     - Compare distance functions: Euclidean ($p=2$) vs Manhattan ($p=1$).
  3. Run both models on Base features and Augmented features.
  4. Compare the performance jump ($\Delta\text{F1}$, $\Delta\text{ROC-AUC}$) when fault features are added.
- **Deliverables**:
  - `artifacts/models/svm_rbf_aug.joblib`
  - `artifacts/models/knn_aug.joblib`
  - Cross-validated out-of-fold probability predictions for SVC (`svm_prob_*`) and k-NN (`knn_prob_*`).

---

## Final Module 3 Outputs (Combining All 3 Roles)

1. **Comparison Report** (`reports/baseline_benchmarks.md`):
   - Table comparing Base vs. Augmented results across all models.
2. **Out-of-Fold Predictions for Module 4** (`artifacts/oof_predictions/`):
   - `oof_train_m3.parquet` and `oof_test_m3.parquet`
   - Columns: `event_id`, `true_label`, `lr_prob_*`, `lda_prob_*`, `svm_prob_*`, `knn_prob_*`.
