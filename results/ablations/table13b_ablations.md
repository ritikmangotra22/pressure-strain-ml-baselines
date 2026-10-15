| Model | Variant | 180 | 550 | 1000 | 2000 | 5200 | Mean, interp. | Mean, extrap. | Folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | Baseline (F1, grid weights) | 0.154 | 0.046 | 0.019 | 0.302 | 0.236 | 0.122 | 0.195 | 5 |
| Ridge | A1: F2 (+ invariants, P/ε) | 0.302 | 0.040 | 0.017 | 0.294 | 0.311 | 0.117 | 0.307 | 5 |
| Ridge | A1: F3 (+ wall distance Re_y) | 0.522 | 0.050 | 0.155 | 0.302 | 1.089 | 0.169 | 0.806 | 5 |
| Ridge | A3: uniform training weights | 0.352 | 0.044 | 0.022 | 0.318 | 0.527 | 0.128 | 0.439 | 5 |
| Random forest | Baseline (F1, grid weights) | 0.211 | 0.119 | 0.055 | 0.031 | 0.045 | 0.068 | 0.128 | 5 |
| Random forest | A1: F2 (+ invariants, P/ε) | 0.208 | 0.121 | 0.055 | 0.031 | 0.053 | 0.069 | 0.131 | 5 |
| Random forest | A1: F3 (+ wall distance Re_y) | 0.218 | 0.081 | 0.024 | 0.026 | 0.036 | 0.043 | 0.127 | 5 |
| Random forest | A3: uniform training weights | 0.234 | 0.150 | 0.059 | 0.033 | 0.080 | 0.081 | 0.157 | 5 |
| LightGBM | Baseline (F1, grid weights) | 0.156 | 0.133 | 0.055 | 0.028 | 0.048 | 0.072 | 0.102 | 5 |
| LightGBM | A1: F2 (+ invariants, P/ε) | 0.176 | 0.123 | 0.053 | 0.032 | 0.048 | 0.069 | 0.112 | 5 |
| LightGBM | A1: F3 (+ wall distance Re_y) | 0.204 | 0.082 | 0.027 | 0.024 | 0.043 | 0.044 | 0.124 | 5 |
| LightGBM | A3: uniform training weights | 0.229 | 0.141 | 0.055 | 0.024 | 0.070 | 0.074 | 0.149 | 5 |
