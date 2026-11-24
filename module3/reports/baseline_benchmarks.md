# Module 3: Supervised Baseline Benchmarks (Stage 1 Base vs. Stage 2 Augmented)

All models were evaluated under identical **Stratified 5-Fold Cross-Validation** (`RANDOM_STATE = 42`) on the Module 1 **Base** split (36 features, 3,105 train / 666 test) and the Module 2 **Augmented** split (61 features: Base + DBSCAN fault distances + 20 GMM posteriors).

---

## 1. PAGER `alert` Multi-Class Classification (5-Fold CV)

| Model Family | Model | Base Macro F1 | Aug Macro F1 | Δ Macro F1 | Base Weighted F1 | Aug Weighted F1 | Base ROC-AUC | Aug ROC-AUC | Δ ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear (P1) | Logistic Regression (Best) | 0.3456 | 0.3407 | -0.0049 | 0.8181 | 0.8217 | 0.8179 | 0.7744 | -0.0436 |
| Probabilistic (P2) | Gaussian Naive Bayes | 0.2843 | 0.2675 | -0.0167 | 0.8411 | 0.7978 | 0.7840 | 0.6984 | -0.0855 |
| Probabilistic (P2) | Linear Discriminant Analysis (LDA) | 0.3249 | 0.3664 | +0.0415 | 0.8850 | 0.8916 | 0.8346 | 0.8332 | -0.0013 |
| Probabilistic (P2) | Quadratic Discriminant Analysis (QDA) | 0.3185 | 0.3615 | +0.0430 | 0.8726 | 0.8548 | 0.8157 | 0.7759 | -0.0398 |
| Instance (P3) | k-Nearest Neighbors (k-NN) | 0.3324 | 0.3633 | +0.0308 | — | 0.8970 | 0.6943 | 0.6825 | -0.0118 |
| Kernel (P3) | Support Vector Classifier (SVC) | 0.3623 | 0.3742 | +0.0120 | — | 0.8867 | 0.9165 | 0.9131 | -0.0034 |

### Person 1 Logistic Regression Penalty Breakdown & Calibration
- **Best Base LR**: L2 (`C=10`) — CV Macro F1 `0.3456`, Weighted F1 `0.8181`, ROC-AUC `0.8179`.
- **Best Augmented LR**: L2 (`C=10`) — CV Macro F1 `0.3407`, Weighted F1 `0.8217`, ROC-AUC `0.7744`.

---

## 2. Continuous ShakeMap `mmi` Regression (Person 1 — 5-Fold CV & Holdout Test)

| Model | Base CV RMSE | Aug CV RMSE | Δ CV RMSE | Base CV R² | Aug CV R² | Δ CV R² | Base Test R² | Aug Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 1.6306 | 1.4674 | -0.1631 | 0.4687 | 0.5698 | +0.1011 | 0.4688 | 0.5652 |
| Lasso | 1.6305 | 1.4667 | -0.1638 | 0.4687 | 0.5702 | +0.1015 | 0.4691 | 0.5638 |

---

## 3. Uniform Audit of Consolidated Module 4 Stacking OOF Predictions (Augmented)

| Role / Family | Model (Augmented) | Train OOF Macro F1 | Train OOF ROC-AUC | Train OOF Log Loss | Test Macro F1 | Test Weighted F1 | Test ROC-AUC | Test Log Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear (P1) | Logistic Regression (L2, C=10) | 0.3393 | 0.7768 | 0.7817 | 0.3207 | 0.8202 | 0.8605 | 0.6014 |
| Probabilistic (P2) | LDA (eigen, shrinkage=0.5) | 0.3681 | 0.8270 | 0.5075 | 0.3548 | 0.9002 | 0.8775 | 0.3643 |
| Kernel (P3) | SVC (rbf, C=10) | 0.2978 | 0.9125 | 0.2561 | 0.2950 | 0.8843 | 0.9076 | 0.2669 |
| Instance (P3) | k-NN (k=3, weights=distance, metric=euclidean) | 0.3627 | 0.7989 | 1.8920 | 0.3821 | 0.9011 | 0.7888 | 1.9439 |

---

## 4. Key Takeaways & Interpretability Findings
1. **Fault features vs. alert classification**: Augmented features raised CV Macro F1 for 4/6 model families (Linear Discriminant Analysis (LDA) +0.0415, Quadratic Discriminant Analysis (QDA) +0.0430, k-Nearest Neighbors (k-NN) +0.0308, Support Vector Classifier (SVC) +0.0120). Best Augmented model: **Support Vector Classifier (SVC)** (Macro F1 `0.3742`). Rare classes stay hard (59 orange / 22 red training events).
2. **Leakage fix (data)**: Module 1 now downloads every alert class through the same USGS endpoint and excludes `magError`, so `missing_*` flags no longer encode the alert class. Scores are lower than in earlier runs because the earlier ones partly measured that artifact.
3. **Continuous shaking (`mmi`)**: Module 2 features raised Ridge CV $R^2$ from `0.4687` to `0.5698` (+0.1011); holdout test $R^2$ `0.4688` -> `0.5652`.
4. **Permutation tests (30 repeats, $\alpha = 0.05$)**: 20/25 Module 2 features are significant for PAGER `alert` ROC-AUC and 16/25 for `mmi` $R^2$; top `mmi` feature overall: `mag` ($\Delta R^2 = +0.1143$).

---

## 5. Exported Handoff Artifacts
- **Models (`module3/artifacts/models/`)**: `logistic_regression_aug.joblib`, `lda_aug.joblib`, `qda_aug.joblib`, `knn_aug.joblib`, `svm_rbf_aug.joblib`
- **Consolidated OOF Parquets (`module3/artifacts/oof_predictions/`)**: `oof_train_m3.parquet` `(3105, 18)`, `oof_test_m3.parquet` `(666, 18)`
