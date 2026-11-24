# Module 4 — Statistical Significance: does unsupervised fault discovery help?


**H₀:** Macro F1(Augmented) = Macro F1(Base). Tests: Dietterich's 5×2 CV paired *t*-test (df = 5) and Wilcoxon signed-rank on
the same 10 paired fold scores; Holm–Bonferroni correction across the 10 model families. α = 0.05.
All fits use the tuned hyper-parameters from the notebook and identical stratified halves of the training set for both feature sets.

## Per-model Base vs Augmented
| model | 5x2 mean F1 Base | 5x2 mean F1 Aug | Δ (Aug−Base) | t (5x2cv) | p (5x2cv t) | p (Wilcoxon) | p Holm (5x2cv t) | verdict (Holm, α=0.05) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.4564 | 0.4606 | 0.0042 | 1.5847 | 0.1739 | 0.4922 | 1.0000 | no significant difference |
| LDA | 0.4698 | 0.4816 | 0.0118 | 0.3735 | 0.7241 | 0.1309 | 1.0000 | no significant difference |
| k-NN | 0.4928 | 0.4818 | -0.0110 | 0.7340 | 0.4959 | 0.4922 | 1.0000 | no significant difference |
| SVC | 0.4308 | 0.4445 | 0.0137 | 0.4684 | 0.6592 | 0.0645 | 1.0000 | no significant difference |
| Random Forest | 0.5359 | 0.5369 | 0.0010 | 0.6061 | 0.5709 | 0.9219 | 1.0000 | no significant difference |
| Extra Trees | 0.5043 | 0.5290 | 0.0247 | 1.9298 | 0.1115 | 0.0371 | 1.0000 | no significant difference |
| LightGBM | 0.5655 | 0.5305 | -0.0350 | -1.5214 | 0.1887 | 0.0098 | 1.0000 | no significant difference |
| XGBoost | 0.5384 | 0.5078 | -0.0306 | -0.4027 | 0.7038 | 0.0059 | 1.0000 | no significant difference |
| CatBoost | 0.5242 | 0.5210 | -0.0032 | -2.0867 | 0.0913 | 0.9219 | 0.9128 | no significant difference |
| Deep Tabular MLP | 0.3922 | 0.3542 | -0.0380 | -0.8889 | 0.4148 | 0.1602 | 1.0000 | no significant difference |

## Headline: best Augmented model vs best Base model
| | Best Augmented | Best Base |
|---|---|---|
| Model (chosen by 5-fold CV Macro F1) | LightGBM | Random Forest |
| 5-fold CV Macro F1 | 0.5414 | 0.5725 |
| 5×2 CV mean Macro F1 | 0.5305 | 0.5359 |

5×2cv *t* = 1.317, *p* = 0.2451; Wilcoxon *p* = 0.4316; mean Δ (Aug − Base) = -0.0053.
**Result: no significant difference (α = 0.05).** The TODO's checklist item (best Augmented beats best Base at *p* < 0.05) is
**not met**.

## Interpretation
- Per model (Holm-corrected): Augmented is significantly **better** for no model, significantly **worse** for no model, and indistinguishable for the remaining 10.
- Headline: the best Base and best Augmented models cannot be told apart at this sample size.
- Context: with only 22 red and 59 orange training events, fold-level Macro F1 is dominated by a handful of rare events, so these tests have limited power and small differences are within split noise. A likely reason the fault features do not help high-capacity learners: tree ensembles and boosters can learn spatial structure directly from latitude/longitude/ECEF coordinates, so the DBSCAN/GMM columns are partly redundant for them, while 25 extra, partly collinear columns (20 GMM posteriors summing to 1) add variance on the rare classes. Models that cannot carve space as flexibly (kernel/linear) are where explicit fault features have the most room to help.
