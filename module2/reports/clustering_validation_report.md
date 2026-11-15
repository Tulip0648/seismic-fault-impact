# Module 2 - Clustering Validation Report


## Data
| split | rows |
|---|---|
| train | 3105 |
| val | 665 |
| test | 666 |

## DBSCAN variant comparison (train, non-noise points)
| variant | eps (km) | eps status | clusters | noise | silhouette | Davies-Bouldin |
|---|---|---|---|---|---|---|
| ecef (selected) | 573.3 | knee | 19 | 7.0% | 0.1081 | 0.6492 |
| haversine | 574.2 | knee | 19 | 7.0% | 0.1059 | 0.6508 |

min_samples = 10; target noise band = 5%-20%.
Noise ratio by split (selected): train 7.0%, val 8.0%, test 7.5%

## GMM
Features: x_ecef, y_ecef, z_ecef, pca_sensor_1, pca_sensor_2, pca_sensor_3 (standardised). Covariance: full. reg_covar: 0.05.

| K | BIC | AIC | train mean log-lik | val mean log-lik | converged |
|---|---|---|---|---|---|
| 5 | 38667.3 | 37827.7 | -6.047 | -6.394 | True |
| 10 | 33772.9 | 32087.5 | -5.077 | -5.442 | True |
| 15 | 31581.9 | 29050.8 | -4.543 | -5.015 | True |
| 20 | 31321.6 | 27944.8 | -4.320 | -4.740 | True |
| 25 | 31514.9 | 27292.4 | -4.170 | -4.599 | True |
| 30 | 31929.3 | 26861.1 | -4.055 | -4.538 | True |
| 40 | 32865.7 | 26106.1 | -3.843 | -4.375 | True |

BIC picks K=20, AIC picks K=40. **Selected K* = 20.**
Train - val log-likelihood gap at K*: 0.420.
Silhouette (hard assignments) = 0.2670, Davies-Bouldin = 1.4590.

## Figures
See `figures/`: k_distance_curves, eps_sweep, dbscan_map_*, gmm_bic_aic, gmm_map, dbscan_3d_ecef.
