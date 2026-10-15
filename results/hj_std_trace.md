| Re_τ | n | median trace/σ_tr | max trace/σ_tr | points with trace > σ_tr | median σ(Π)/‖Π‖ | max σ(Π)/‖Π‖ (y⁺) |
|---|---|---|---|---|---|---|
| 186 | 48 | 9.5e-05 | 1.4e-01 | 0 | 5.5e-03 | 1.9e-02 (4.87) |
| 547 | 128 | 1.9e-04 | 1.2e+00 | 0 | 3.1e-03 | 5.4e-02 (546.54) |
| 934 | 192 | 1.4e-03 | 6.4e+00 | 1 | 5.0e-03 | 4.0e-02 (880.51) |
| 2003 | 316 | 2.2e-06 | 4.3e-05 | 0 | 5.6e-03 | 3.0e-02 (1913.72) |

σ_tr = sqrt(σ11² + σ22² + σ33²) of the published p-strain standard deviations (independent errors); the column 'trace > σ_tr' uses the larger, fully correlated bound σ11 + σ22 + σ33.

| Re_τ | E_σ (E_Π formula applied to σ(Π)/ε vs Π/ε, grid weights) |
|---|---|
| 186 | 0.0050 |
| 547 | 0.0074 |
| 934 | 0.0078 |
| 2003 | 0.0110 |

Interpretation caveat: the README does not say whether σ is the sample standard deviation or the standard error of the time average. If it is the standard error, E_σ is a rough floor on the E2 error attributable to finite DNS sampling. If it is a per-sample standard deviation, the standard error is smaller by √N_eff (N_eff unknown).
