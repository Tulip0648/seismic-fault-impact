# Module 3: Supervised Baseline Benchmarks (Stage 1 Base vs. Stage 2 Augmented)

All models were evaluated under identical **Stratified 5-Fold Cross-Validation** (`RANDOM_STATE = 42`) on the Module 1 **Base** split (39 features, 3,105 train / 666 test) and the Module 2 **Augmented** split (64 features: Base + DBSCAN fault distances + 20 GMM posteriors).

---

## 1. PAGER `alert` Multi-Class Classification (5-Fold CV)

| Model Family | Model | Base Macro F1 | Aug Macro F1 | Δ Macro F1 | Base Weighted F1 | Aug Weighted F1 | Base ROC-AUC | Aug ROC-AUC | Δ ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear (P1) | Logistic Regression (Best) | 0.4831 | 0.4555 | -0.0276 | 0.9125 | 0.9175 | 0.8965 | 0.8957 | -0.0008 |
| Probabilistic (P2) | Gaussian Naive Bayes | 0.3603 | 0.3117 | -0.0486 | 0.8922 | 0.8261 | 0.7502 | 0.7305 | -0.0197 |
| Probabilistic (P2) | Linear Discriminant Analysis (LDA) | 0.4838 | 0.4589 | -0.0249 | 0.9473 | 0.9457 | 0.9335 | 0.9423 | +0.0088 |
| Probabilistic (P2) | Quadratic Discriminant Analysis (QDA) | 0.4452 | 0.4102 | -0.0350 | 0.9248 | 0.8784 | 0.8553 | 0.8071 | -0.0482 |
| Instance (P3) | k-Nearest Neighbors (k-NN) | 0.4925 | 0.4758 | -0.0167 | — | 0.9463 | 0.8296 | 0.7787 | -0.0509 |
| Kernel (P3) | Support Vector Classifier (SVC) | 0.4692 | 0.4888 | +0.0196 | — | 0.9458 | 0.9601 | 0.9497 | -0.0104 |

### Person 1 Logistic Regression Penalty Breakdown & Calibration
- **Best Base LR**: L2 (`saga`, `C=10.0`) — CV Macro F1 `0.4831`, ROC-AUC `0.8965`, Log Loss `0.3775`, Latency `2.42 ms/batch` (`0.0036 ms/sample`).
- **Best Augmented LR**: L1 (`saga`, `C=0.1`) — CV Macro F1 `0.4555`, Weighted F1 `0.9175` (`+0.0050`), ROC-AUC `0.8957`, Log Loss `0.3324` (`-0.0451`), Latency `2.56 ms/batch` (`0.0038 ms/sample`).

---

## 2. Continuous ShakeMap `mmi` Regression (Person 1 — 5-Fold CV & Holdout Test)

| Model | Base CV RMSE | Aug CV RMSE | Δ CV RMSE | Base CV R² | Aug CV R² | Δ CV R² | Base Test R² | Aug Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 1.5601 | 1.4123 | -0.1477 | 0.5135 | 0.6013 | +0.0879 | 0.5095 | 0.6031 |
| Lasso | 1.5600 | 1.4121 | -0.1479 | 0.5135 | 0.6015 | +0.0880 | 0.5097 | 0.6023 |

---

## 3. Uniform Audit of Consolidated Module 4 Stacking OOF Predictions (Augmented)

| Role / Family | Model (Augmented) | Train OOF Macro F1 | Train OOF ROC-AUC | Train OOF Log Loss | Test Macro F1 | Test Weighted F1 | Test ROC-AUC | Test Log Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Linear (P1) | Logistic Regression (L1, C=0.1) | 0.4545 | 0.8949 | 0.3324 | 0.4500 | 0.9154 | 0.9396 | 0.2790 |
| Probabilistic (P2) | LDA (eigen, shrinkage=auto) | 0.4625 | 0.9153 | 0.6647 | 0.4724 | 0.9421 | 0.9492 | 0.6300 |
| Kernel (P3) | SVC (poly deg=2, C=1.0) | 0.4454 | 0.9483 | 0.1665 | 0.4693 | 0.9475 | 0.9601 | 0.1623 |
| Instance (P3) | k-NN (k=3, distance, euclidean) | 0.4776 | 0.8636 | 1.0142 | 0.4461 | 0.9415 | 0.8190 | 1.0880 |

---

## 4. Key Takeaways & Interpretability Findings
1. **Non-linear vs. Linear Response to Tectonic Features**: Adding 25 DBSCAN/GMM features directly improved non-linear kernel **SVC** Macro F1 (`+0.0196`, reaching the highest Augmented Macro F1 of `0.4888`), while linear/probabilistic/distance baselines experienced mild high-dimensional variance on rare `orange`/`red` classes (59 and 22 train events respectively).
2. **Calibration Gain for Stacking**: Under L1 regularization (`C=0.1`), Augmented Logistic Regression improved **Weighted F1 (`+0.0050`)**, reduced **CV Log Loss (`-0.0451`)** and **Test Log Loss (`0.2846 -> 0.2790`)**, and preserved non-zero recall on `orange` and `red` alerts.
3. **Substantial Continuous Shaking (`mmi`) Gain**: Module 2 tectonic features boosted both Ridge and Lasso CV $R^2$ by **`+0.088`** (`0.5135 -> 0.6015`) and reduced RMSE by **`-0.148`**, with holdout Test $R^2$ rising **`+0.093`**.
4. **Statistical Significance of Fault Features**: Holdout permutation tests (30 repeats, $\alpha = 0.05$) confirmed **13/25** Module 2 features are statistically significant for PAGER `alert` ROC-AUC (`gmm_prob_9`, `gmm_prob_4`, and `dist_to_nearest_fault_km` all $p < 0.0001$), and **20/25** are significant for continuous `mmi` $R^2$ (`gmm_prob_15` ranks #1 overall with $\Delta R^2 = +0.1044$, $p < 0.0001$).

---

## 5. Exported Handoff Artifacts
- **Models (`module3/artifacts/models/`)**: `logistic_regression_aug.joblib`, `lda_aug.joblib`, `qda_aug.joblib`, `knn_aug.joblib`, `svm_rbf_aug.joblib`
- **Consolidated OOF Parquets (`module3/artifacts/oof_predictions/`)**: `oof_train_m3.parquet` `(3105, 18)`, `oof_test_m3.parquet` `(666, 18)`
