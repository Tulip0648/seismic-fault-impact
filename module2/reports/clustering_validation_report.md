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
| 5 | 38580.2 | 37740.6 | -6.033 | -6.159 | True |
| 10 | 32463.5 | 30778.1 | -4.866 | -5.157 | True |
| 15 | 30643.1 | 28112.0 | -4.392 | -4.678 | True |
| 20 | 30469.0 | 27092.2 | -4.183 | -4.451 | True |
| 25 | 30762.4 | 26539.9 | -4.049 | -4.356 | True |
| 30 | 31172.6 | 26104.4 | -3.933 | -4.265 | True |
| 40 | 32265.6 | 25506.0 | -3.747 | -4.110 | True |

BIC picks K=20, AIC picks K=40. **Selected K* = 20.**
Train - val log-likelihood gap at K*: 0.268.
Silhouette (hard assignments) = 0.2798, Davies-Bouldin = 1.3791.

## Figures
See `figures/`: k_distance_curves, eps_sweep, dbscan_map_*, gmm_bic_aic, gmm_map, dbscan_3d_ecef.
