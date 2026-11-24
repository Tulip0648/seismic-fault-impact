# Module 4 — Statistical Significance: does unsupervised fault discovery help?


**H₀:** Macro F1(Augmented) = Macro F1(Base). Tests: Dietterich's 5×2 CV paired *t*-test (df = 5) and Wilcoxon signed-rank on
the same 10 paired fold scores; Holm–Bonferroni correction across the 10 model families. α = 0.05.
All fits use the tuned hyper-parameters from the notebook and identical stratified halves of the training set for both feature sets.

## Per-model Base vs Augmented
| model | 5x2 mean F1 Base | 5x2 mean F1 Aug | Δ (Aug−Base) | t (5x2cv) | p (5x2cv t) | p (Wilcoxon) | p Holm (5x2cv t) | verdict (Holm, α=0.05) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.3365 | 0.3485 | 0.0120 | 1.3330 | 0.2400 | 0.0371 | 1.0000 | no significant difference |
| LDA | 0.3259 | 0.3733 | 0.0473 | 0.4361 | 0.6810 | 0.0020 | 1.0000 | no significant difference |
| k-NN | 0.3600 | 0.3840 | 0.0240 | 0.5431 | 0.6104 | 0.0098 | 1.0000 | no significant difference |
| SVC | 0.2800 | 0.2902 | 0.0101 | 3.4986 | 0.0173 | 0.0645 | 0.1731 | no significant difference |
| Random Forest | 0.4091 | 0.4223 | 0.0133 | -1.1069 | 0.3187 | 0.4922 | 1.0000 | no significant difference |
| Extra Trees | 0.4031 | 0.4008 | -0.0023 | 3.2885 | 0.0217 | 0.7695 | 0.1957 | no significant difference |
| LightGBM | 0.4380 | 0.4255 | -0.0125 | -0.9093 | 0.4049 | 0.3223 | 1.0000 | no significant difference |
| XGBoost | 0.4061 | 0.4171 | 0.0111 | 1.0502 | 0.3417 | 0.1309 | 1.0000 | no significant difference |
| CatBoost | 0.4235 | 0.4122 | -0.0113 | -0.4481 | 0.6729 | 0.7695 | 1.0000 | no significant difference |
| Deep Tabular MLP | 0.2587 | 0.2423 | -0.0164 | 0.1255 | 0.9050 | 0.1934 | 1.0000 | no significant difference |

## Headline: best Augmented model vs best Base model
| | Best Augmented | Best Base |
|---|---|---|
| Model (chosen by 5-fold CV Macro F1) | Random Forest | LightGBM |
| 5-fold CV Macro F1 | 0.4236 | 0.4542 |
| 5×2 CV mean Macro F1 | 0.4223 | 0.4380 |

5×2cv *t* = -2.678, *p* = 0.0439; Wilcoxon *p* = 0.3223; mean Δ (Aug − Base) = -0.0157.
**Result: Augmented significantly worse (α = 0.05).** The TODO's checklist item (best Augmented beats best Base at *p* < 0.05) is
**not met**.

## Interpretation
- Per model (Holm-corrected): Augmented is significantly **better** for no model, significantly **worse** for no model, and indistinguishable for the remaining 10.
- Headline: the best model *without* the fault features is significantly better than the best model with them, i.e. adding the 25 Module 2 columns costs Macro F1 for the strongest learners.
- Context: with only 22 red and 59 orange training events, fold-level Macro F1 is dominated by a handful of rare events, so these tests have limited power and small differences are within split noise. A likely reason the fault features do not help high-capacity learners: tree ensembles and boosters can learn spatial structure directly from latitude/longitude/ECEF coordinates, so the DBSCAN/GMM columns are partly redundant for them, while 25 extra, partly collinear columns (20 GMM posteriors summing to 1) add variance on the rare classes. Models that cannot carve space as flexibly (kernel/linear) are where explicit fault features have the most room to help.
