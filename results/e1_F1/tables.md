### Table 7 — DNS consistency

| Case    |   Re_τ (actual) |   Points (half channel, excl. wall) |   k-budget residual / ε, median |   k-budget residual / ε, max |   DNS trace residual, max (rel.) |
|:--------|----------------:|------------------------------------:|--------------------------------:|-----------------------------:|---------------------------------:|
| LM 180  |           182.1 |                                  95 |                          0.0015 |                       0.0081 |                            7e-16 |
| LM 550  |           543.5 |                                 191 |                          0.002  |                       0.011  |                            6e-16 |
| LM 1000 |          1000.5 |                                 255 |                          0.0015 |                       0.0089 |                            1e-15 |
| LM 2000 |          1994.8 |                                 383 |                          0.0017 |                       0.0091 |                            1e-15 |
| LM 5200 |          5185.9 |                                 767 |                          0.002  |                       0.02   |                            1e-15 |

### Table 8 — E_Π on each held-out Re_τ (mean ± SD over seeds)

| Model                             | 180           | 550           | 1000          | 2000          | 5200          |   Mean, interp. |   Mean, extrap. |
|:----------------------------------|:--------------|:--------------|:--------------|:--------------|:--------------|----------------:|----------------:|
| Rotta+SSG, published              | 1.073         | 0.405         | 0.284         | 0.217         | 0.182         |           0.302 |           0.627 |
| Rotta+SSG, refitted               | 0.396         | 0.159         | 0.143         | 0.146         | 0.158         |           0.149 |           0.277 |
| Rotta+LRR-IP, published           | 1.036         | 0.405         | 0.301         | 0.239         | 0.194         |           0.315 |           0.615 |
| Rotta+LRR-IP, refitted            | 0.636         | 0.238         | 0.208         | 0.204         | 0.210         |           0.216 |           0.423 |
| Rotta+LRR-IP+GL wall, published   | 0.889         | 0.329         | 0.258         | 0.240         | 0.260         |           0.275 |           0.574 |
| Rotta+LRR-IP+GL wall, refitted    | 0.639         | 0.228         | 0.206         | 0.206         | 0.211         |           0.214 |           0.425 |
| Ridge                             | 0.154         | 0.046         | 0.019         | 0.302         | 0.236         |           0.122 |           0.195 |
| Random forest                     | 0.211 ± 0.001 | 0.119 ± 0.004 | 0.055 ± 0.001 | 0.031 ± 0.000 | 0.045 ± 0.000 |           0.068 |           0.128 |
| Gradient-boosted trees (LightGBM) | 0.156 ± 0.005 | 0.133 ± 0.004 | 0.055 ± 0.000 | 0.028 ± 0.002 | 0.048 ± 0.000 |           0.072 |           0.102 |
| MLP                               | 0.143 ± 0.008 | 0.061 ± 0.013 | 0.021 ± 0.003 | 0.043 ± 0.009 | 0.167 ± 0.072 |           0.042 |           0.155 |

### Table 8b — 95 % block-bootstrap interval of E_Π (seed-averaged bounds)

| Model                             | 180            | 550            | 1000           | 2000           | 5200           |
|:----------------------------------|:---------------|:---------------|:---------------|:---------------|:---------------|
| Rotta+SSG, published              | [0.509, 1.813] | [0.131, 0.732] | [0.086, 0.521] | [0.104, 0.353] | [0.110, 0.221] |
| Rotta+SSG, refitted               | [0.252, 0.611] | [0.082, 0.230] | [0.074, 0.197] | [0.091, 0.187] | [0.104, 0.200] |
| Rotta+LRR-IP, published           | [0.615, 1.652] | [0.202, 0.688] | [0.168, 0.494] | [0.158, 0.349] | [0.132, 0.235] |
| Rotta+LRR-IP, refitted            | [0.326, 1.044] | [0.115, 0.381] | [0.103, 0.302] | [0.114, 0.266] | [0.127, 0.261] |
| Rotta+LRR-IP+GL wall, published   | [0.517, 1.442] | [0.150, 0.556] | [0.157, 0.406] | [0.180, 0.311] | [0.208, 0.299] |
| Rotta+LRR-IP+GL wall, refitted    | [0.331, 1.043] | [0.100, 0.367] | [0.089, 0.294] | [0.107, 0.274] | [0.132, 0.266] |
| Ridge                             | [0.116, 0.216] | [0.038, 0.050] | [0.014, 0.023] | [0.229, 0.328] | [0.192, 0.261] |
| Random forest                     | [0.171, 0.261] | [0.085, 0.138] | [0.040, 0.071] | [0.023, 0.040] | [0.034, 0.055] |
| Gradient-boosted trees (LightGBM) | [0.119, 0.201] | [0.092, 0.155] | [0.038, 0.074] | [0.023, 0.033] | [0.034, 0.065] |
| MLP                               | [0.096, 0.189] | [0.047, 0.069] | [0.014, 0.027] | [0.021, 0.059] | [0.117, 0.196] |

### Table 9 — Per-component errors (Π/ε units), averaged over E1 folds

| Model                             | Pi11 RMSE / MAE / R²   | Pi22 RMSE / MAE / R²   | Pi33 RMSE / MAE / R²   | Pi12 RMSE / MAE / R²   |   median r (τ=0.05) |   90th pct r (τ=0.05) |
|:----------------------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|--------------------:|----------------------:|
| Rotta+SSG, published              | 0.526 / 0.261 / -3.30  | 0.363 / 0.194 / -4.56  | 0.191 / 0.104 / -2.29  | 0.236 / 0.172 / 0.07   |               0.181 |                 0.844 |
| Rotta+SSG, refitted               | 0.156 / 0.114 / 0.65   | 0.126 / 0.090 / 0.34   | 0.082 / 0.057 / 0.54   | 0.204 / 0.171 / 0.45   |               0.135 |                 0.574 |
| Rotta+LRR-IP, published           | 0.557 / 0.346 / -3.64  | 0.413 / 0.287 / -6.03  | 0.168 / 0.086 / -1.56  | 0.179 / 0.155 / 0.45   |               0.255 |                 0.787 |
| Rotta+LRR-IP, refitted            | 0.405 / 0.295 / -1.29  | 0.234 / 0.161 / -1.31  | 0.181 / 0.146 / -1.28  | 0.131 / 0.111 / 0.75   |               0.172 |                 0.94  |
| Rotta+LRR-IP+GL wall, published   | 0.418 / 0.196 / -1.88  | 0.299 / 0.234 / -2.68  | 0.345 / 0.246 / -7.44  | 0.205 / 0.176 / 0.45   |               0.251 |                 0.753 |
| Rotta+LRR-IP+GL wall, refitted    | 0.389 / 0.258 / -1.15  | 0.193 / 0.125 / -0.64  | 0.205 / 0.151 / -1.90  | 0.163 / 0.140 / 0.68   |               0.156 |                 0.975 |
| Ridge                             | 0.118 / 0.106 / 0.63   | 0.094 / 0.081 / 0.26   | 0.115 / 0.102 / -1.66  | 0.139 / 0.121 / 0.71   |               0.14  |                 0.3   |
| Random forest                     | 0.051 / 0.044 / 0.96   | 0.032 / 0.028 / 0.96   | 0.030 / 0.025 / 0.92   | 0.102 / 0.089 / 0.82   |               0.081 |                 0.166 |
| Gradient-boosted trees (LightGBM) | 0.063 / 0.053 / 0.94   | 0.037 / 0.031 / 0.94   | 0.028 / 0.024 / 0.92   | 0.091 / 0.073 / 0.89   |               0.077 |                 0.138 |
| MLP                               | 0.051 / 0.043 / 0.94   | 0.037 / 0.030 / 0.92   | 0.030 / 0.024 / 0.88   | 0.102 / 0.089 / 0.85   |               0.081 |                 0.203 |

### Table 10 — E_Π by wall region

| Model                             | Regime   |   viscous |   buffer |   log |   outer |
|:----------------------------------|:---------|----------:|---------:|------:|--------:|
| Rotta+SSG, published              | interp   |     3.728 |    2.165 | 0.255 |   0.099 |
| Rotta+SSG, published              | extrap   |     4.465 |    2.624 | 0.642 |   0.262 |
| Rotta+SSG, refitted               | interp   |     1.964 |    0.356 | 0.077 |   0.148 |
| Rotta+SSG, refitted               | extrap   |     2.712 |    0.585 | 0.164 |   0.235 |
| Rotta+LRR-IP, published           | interp   |     4.126 |    1.894 | 0.328 |   0.164 |
| Rotta+LRR-IP, published           | extrap   |     5.288 |    2.29  | 0.702 |   0.324 |
| Rotta+LRR-IP, refitted            | interp   |     1.166 |    0.993 | 0.087 |   0.191 |
| Rotta+LRR-IP, refitted            | extrap   |     1.952 |    1.362 | 0.324 |   0.262 |
| Rotta+LRR-IP+GL wall, published   | interp   |     4.032 |    1.534 | 0.242 |   0.167 |
| Rotta+LRR-IP+GL wall, published   | extrap   |     5.213 |    1.916 | 0.643 |   0.318 |
| Rotta+LRR-IP+GL wall, refitted    | interp   |     0.998 |    0.935 | 0.087 |   0.193 |
| Rotta+LRR-IP+GL wall, refitted    | extrap   |     1.961 |    1.342 | 0.357 |   0.262 |
| Ridge                             | interp   |     0.915 |    0.178 | 0.102 |   0.125 |
| Ridge                             | extrap   |     0.788 |    0.22  | 0.148 |   0.185 |
| Random forest                     | interp   |     0.194 |    0.055 | 0.059 |   0.068 |
| Random forest                     | extrap   |     0.429 |    0.159 | 0.102 |   0.125 |
| Gradient-boosted trees (LightGBM) | interp   |     0.106 |    0.038 | 0.056 |   0.072 |
| Gradient-boosted trees (LightGBM) | extrap   |     0.279 |    0.113 | 0.107 |   0.093 |
| MLP                               | interp   |     0.341 |    0.067 | 0.019 |   0.044 |
| MLP                               | extrap   |     0.535 |    0.11  | 0.066 |   0.163 |

### Table 13 — Trace residual of predictions

| Model                             |   Mean t_n |   Max t_n |
|:----------------------------------|-----------:|----------:|
| Rotta+SSG, published              |   3.89e-16 |  1.83e-15 |
| Rotta+SSG, refitted               |   1.1e-16  |  1.38e-15 |
| Rotta+LRR-IP, published           |   2.67e-16 |  1.41e-15 |
| Rotta+LRR-IP, refitted            |   5.01e-17 |  1.2e-15  |
| Rotta+LRR-IP+GL wall, published   |   2.98e-16 |  1.59e-15 |
| Rotta+LRR-IP+GL wall, refitted    |   7.72e-17 |  2.28e-15 |
| Ridge                             |   1.33e-13 |  2.43e-12 |
| Random forest                     |   8.17e-16 |  7.92e-15 |
| Gradient-boosted trees (LightGBM) |   0.0159   |  0.48     |
| MLP                               |   0.005    |  0.171    |

### Table 14 — Computational cost (2-core cloud CPU)

| Model                             |   Tuning time per fold (s) |   Final fit (s) |   Inference per point (µs) |
|:----------------------------------|---------------------------:|----------------:|---------------------------:|
| Rotta+SSG, published              |                          0 |            0    |                       1.55 |
| Rotta+SSG, refitted               |                          0 |            0.01 |                       1.83 |
| Rotta+LRR-IP, published           |                          0 |            0    |                       0.99 |
| Rotta+LRR-IP, refitted            |                          0 |            0    |                       0.96 |
| Rotta+LRR-IP+GL wall, published   |                          0 |            0    |                       2.05 |
| Rotta+LRR-IP+GL wall, refitted    |                          0 |            0    |                       1.96 |
| Ridge                             |                          2 |            0.01 |                       5.94 |
| Random forest                     |                        298 |            1.54 |                     171.4  |
| Gradient-boosted trees (LightGBM) |                        202 |            0.37 |                     110.11 |
| MLP                               |                       1286 |            4.26 |                       6.8  |

### Table 11 — Selected hyperparameters / refitted coefficients per outer fold

| Model                             | 180                                                                                                                                                   | 550                                                                                                                                                | 1000                                                                                                                                                 | 2000                                                                                                                                                | 5200                                                                                                                                                     |
|:----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Rotta+SSG, published              | {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}                                                                      | {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}                                                                   | {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}                                                                     | {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}                                                                    | {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}                                                                         |
| Rotta+SSG, refitted               | {"C1": 3.005, "C1s": -2.756, "C2": 6.375, "C3": 1.048, "C3s": -0.477, "C4": 2.514, "C5": 2.047}                                                       | {"C1": 2.136, "C1s": -2.923, "C2": 5.559, "C3": 1.18, "C3s": -0.613, "C4": 2.71, "C5": 2.417}                                                      | {"C1": 2.002, "C1s": -2.887, "C2": 5.224, "C3": 1.178, "C3s": -0.621, "C4": 2.693, "C5": 2.415}                                                      | {"C1": 1.926, "C1s": -2.855, "C2": 5.111, "C3": 1.175, "C3s": -0.616, "C4": 2.68, "C5": 2.404}                                                      | {"C1": 1.934, "C1s": -2.861, "C2": 4.987, "C3": 1.171, "C3s": -0.598, "C4": 2.663, "C5": 2.381}                                                          |
| Rotta+LRR-IP, published           | {"CR": 1.8, "CP": 0.6}                                                                                                                                | {"CR": 1.8, "CP": 0.6}                                                                                                                             | {"CR": 1.8, "CP": 0.6}                                                                                                                               | {"CR": 1.8, "CP": 0.6}                                                                                                                              | {"CR": 1.8, "CP": 0.6}                                                                                                                                   |
| Rotta+LRR-IP, refitted            | {"CR": 0.604, "CP": 0.761}                                                                                                                            | {"CR": 0.145, "CP": 0.843}                                                                                                                         | {"CR": 0.054, "CP": 0.858}                                                                                                                           | {"CR": 0.005, "CP": 0.863}                                                                                                                          | {"CR": 0.006, "CP": 0.857}                                                                                                                               |
| Rotta+LRR-IP+GL wall, published   | {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.18}                                                                                              | {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.18}                                                                                           | {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.18}                                                                                             | {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.18}                                                                                            | {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.18}                                                                                                 |
| Rotta+LRR-IP+GL wall, refitted    | {"CR": 0.632, "CP": 0.785, "Cref1": 0.022, "D_Cref2xCP": 0.034}                                                                                       | {"CR": -0.132, "CP": 0.966, "Cref1": -0.159, "D_Cref2xCP": 0.044}                                                                                  | {"CR": -0.245, "CP": 0.989, "Cref1": -0.183, "D_Cref2xCP": 0.047}                                                                                    | {"CR": -0.291, "CP": 0.997, "Cref1": -0.191, "D_Cref2xCP": 0.051}                                                                                   | {"CR": -0.247, "CP": 0.984, "Cref1": -0.173, "D_Cref2xCP": 0.058}                                                                                        |
| Ridge                             | {"alpha": 0.0, "degree": 2}                                                                                                                           | {"alpha": 0.0, "degree": 2}                                                                                                                        | {"alpha": 0.0, "degree": 2}                                                                                                                          | {"alpha": 65.0255, "degree": 1}                                                                                                                     | {"alpha": 0.0, "degree": 2}                                                                                                                              |
| Random forest                     | {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 1, "max_features": 1.0}                                                                    | {"n_estimators": 300, "max_depth": 32, "min_samples_leaf": 1, "max_features": 1.0}                                                                 | {"n_estimators": 1000, "max_depth": 8, "min_samples_leaf": 1, "max_features": 0.5}                                                                   | {"n_estimators": 1000, "max_depth": 32, "min_samples_leaf": 1, "max_features": 1.0}                                                                 | {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 1, "max_features": 1.0}                                                                        |
| Gradient-boosted trees (LightGBM) | {"learning_rate": 0.1192, "num_leaves": 16, "min_child_samples": 10, "subsample": 0.9939, "reg_lambda": 0.0298, "n_estimators": [132, 176, 109, 138]} | {"learning_rate": 0.1492, "num_leaves": 8, "min_child_samples": 6, "subsample": 0.6376, "reg_lambda": 0.0223, "n_estimators": [63, 173, 136, 227]} | {"learning_rate": 0.031, "num_leaves": 11, "min_child_samples": 12, "subsample": 0.8451, "reg_lambda": 0.0136, "n_estimators": [362, 442, 558, 349]} | {"learning_rate": 0.1321, "num_leaves": 14, "min_child_samples": 11, "subsample": 0.7025, "reg_lambda": 0.0565, "n_estimators": [67, 88, 134, 215]} | {"learning_rate": 0.0105, "num_leaves": 8, "min_child_samples": 18, "subsample": 0.8531, "reg_lambda": 0.0042, "n_estimators": [1544, 1307, 2383, 2036]} |
| MLP                               | {"n_layers": 4, "width": 64, "activation": "silu", "lr": 0.0029, "weight_decay": 0.0, "epochs": 2067}                                                 | {"n_layers": 5, "width": 128, "activation": "tanh", "lr": 0.0018, "weight_decay": 0.0, "epochs": 267}                                              | {"n_layers": 4, "width": 64, "activation": "silu", "lr": 0.0011, "weight_decay": 0.0, "epochs": 1299}                                                | {"n_layers": 5, "width": 128, "activation": "tanh", "lr": 0.002, "weight_decay": 0.0056, "epochs": 464}                                             | {"n_layers": 4, "width": 64, "activation": "silu", "lr": 0.0009, "weight_decay": 0.0004, "epochs": 1311}                                                 |
