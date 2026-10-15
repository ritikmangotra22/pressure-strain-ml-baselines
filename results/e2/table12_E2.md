### Table 12 — E2: E_Π on Hoyas & Jiménez (trained on all five Lee & Moser cases; mean ± SD over seeds)

| Model                             | HJ 186        | HJ 547        | HJ 934        | HJ 2003       |
|:----------------------------------|:--------------|:--------------|:--------------|:--------------|
| Rotta+SSG, published              | 1.039         | 0.397         | 0.290         | 0.219         |
| Rotta+SSG, refitted               | 0.292         | 0.148         | 0.144         | 0.146         |
| Rotta+LRR-IP, published           | 1.005         | 0.398         | 0.303         | 0.238         |
| Rotta+LRR-IP, refitted            | 0.516         | 0.234         | 0.205         | 0.200         |
| Rotta+LRR-IP+GL wall, published   | 0.862         | 0.322         | 0.258         | 0.241         |
| Rotta+LRR-IP+GL wall, refitted    | 0.490         | 0.224         | 0.202         | 0.203         |
| Ridge                             | 0.045         | 0.027         | 0.019         | 0.019         |
| Random forest                     | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.030 ± 0.000 | 0.011 ± 0.000 |
| Gradient-boosted trees (LightGBM) | 0.019 ± 0.000 | 0.023 ± 0.000 | 0.035 ± 0.000 | 0.013 ± 0.000 |
| MLP                               | 0.020 ± 0.005 | 0.023 ± 0.004 | 0.016 ± 0.001 | 0.013 ± 0.001 |

### Table 12b — E2: 95 % block-bootstrap interval of E_Π (seed-averaged bounds)

| Model                             | HJ 186         | HJ 547         | HJ 934         | HJ 2003        |
|:----------------------------------|:---------------|:---------------|:---------------|:---------------|
| Rotta+SSG, published              | [0.508, 1.779] | [0.111, 0.738] | [0.086, 0.514] | [0.099, 0.300] |
| Rotta+SSG, refitted               | [0.176, 0.431] | [0.076, 0.209] | [0.083, 0.195] | [0.089, 0.187] |
| Rotta+LRR-IP, published           | [0.602, 1.614] | [0.190, 0.695] | [0.164, 0.488] | [0.147, 0.305] |
| Rotta+LRR-IP, refitted            | [0.273, 0.846] | [0.116, 0.380] | [0.103, 0.296] | [0.107, 0.250] |
| Rotta+LRR-IP+GL wall, published   | [0.504, 1.387] | [0.141, 0.566] | [0.153, 0.398] | [0.176, 0.282] |
| Rotta+LRR-IP+GL wall, refitted    | [0.257, 0.797] | [0.101, 0.363] | [0.092, 0.293] | [0.104, 0.259] |
| Ridge                             | [0.036, 0.051] | [0.022, 0.031] | [0.013, 0.022] | [0.010, 0.025] |
| Random forest                     | [0.011, 0.020] | [0.013, 0.019] | [0.023, 0.037] | [0.006, 0.015] |
| Gradient-boosted trees (LightGBM) | [0.016, 0.023] | [0.017, 0.026] | [0.025, 0.042] | [0.007, 0.018] |
| MLP                               | [0.014, 0.023] | [0.016, 0.029] | [0.012, 0.019] | [0.010, 0.014] |

### Table 12c — E2: E_Π by wall region and trace residual (averaged over the four HJ cases)

| Model                             |   viscous |   buffer |   log |   outer |   Mean t_n |
|:----------------------------------|----------:|---------:|------:|--------:|-----------:|
| Rotta+SSG, published              |     4.33  |    2.403 | 0.45  |   0.171 |    1.9e-16 |
| Rotta+SSG, refitted               |     2.154 |    0.356 | 0.098 |   0.174 |    7.8e-17 |
| Rotta+LRR-IP, published           |     4.887 |    2.103 | 0.525 |   0.246 |    1.3e-16 |
| Rotta+LRR-IP, refitted            |     1.359 |    1.138 | 0.148 |   0.208 |    3.1e-17 |
| Rotta+LRR-IP+GL wall, published   |     4.773 |    1.733 | 0.413 |   0.226 |    1.5e-16 |
| Rotta+LRR-IP+GL wall, refitted    |     1.068 |    1.068 | 0.145 |   0.21  |    4.5e-17 |
| Ridge                             |     0.192 |    0.045 | 0.019 |   0.027 |    3.1e-14 |
| Random forest                     |     0.121 |    0.031 | 0.015 |   0.018 |    3.5e-15 |
| Gradient-boosted trees (LightGBM) |     0.06  |    0.019 | 0.015 |   0.023 |    0.0041  |
| MLP                               |     0.151 |    0.025 | 0.012 |   0.018 |    0.00089 |
