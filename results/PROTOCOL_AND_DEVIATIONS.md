# Registered protocol, as run, and deviations

The registered protocol is manuscript Sections 7–9 (draft of 2026-10-15).

## As run

| Item | Setting |
|---|---|
| Data, E1 | Lee & Moser (2015), Re_τ = 182, 544, 1001, 1995, 5186. 35 files, SHA-256 in `data/raw/lee_moser_2015/SHA256SUMS.txt` |
| Data, E2 | Hoyas & Jiménez (2008), Re_τ = 186, 547, 934, 2003. Archive SHA-256 in `data/raw/hoyas_jimenez_2008/SHA256_chandata.txt` |
| Points | Wall point removed (y⁺ ≤ 1e-8). LM: 95/191/255/383/767; HJ: 48/128/192/316 |
| Features | F1 = (b11, b22, b12, η = k\|dU/dy\|/ε, Re_t = k²/(νε)) |
| Target | (Π11, Π22, Π33, Π12)/ε |
| Weights | Trapezoidal grid spacing, normalised to 1 per case; per-case renormalisation when cases are pooled |
| E1 split | Leave one Re_τ out (5 outer folds). Interpolation: 550, 1000, 2000. Extrapolation: 180, 5200 |
| Tuning | Nested leave-one-Re_τ-out on the 4 training cases. Criterion: mean inner E_Π. Optuna TPE (sampler seed 2026), MedianPruner (10 start-up trials, pruning on the running mean inner error) |
| Trials | 100 for every tuned model (MLP first run with 40; retuned with 100, see D1) |
| Final fit | Refit on the 4 training cases with the selected hyperparameters. LightGBM rounds and MLP epochs = median of the inner early-stopping points |
| Seeds | 0–9 for RF, LightGBM and MLP; one deterministic fit for ridge and the classical closures |
| Metrics | E_Π (primary; weighted, Frobenius with off-diagonal counted twice); per-component MAE/RMSE/R²; pointwise relative error (τ = 0.05 and 0.01 of max); region E_Π; trace residual |
| Uncertainty | SD over seeds; moving-block bootstrap (2000 resamples, block = 10 % of profile) |
| E2 | Tune by leave-one-out over all 5 LM cases, refit on all 5, test on the 4 HJ cases. No HJ data used for any choice |

## Deviations and corrections

| ID | What | Why | Effect |
|---|---|---|---|
| D1 | MLP first tuned with 40 trials instead of 100. **Resolved 2026-10-15:** retuned with 100 trials (configs/e1_mlp100.json, configs/e2_mlp100.json) | Compute budget on a 2-core CPU | 100-trial results reported; first 40 trials reproduce the original logs exactly; originals in results/mlp40_original/; comparison in results/mlp_budget_comparison.md |
| D2 | Median pruning added to all tuners (not in the original plan) | Compute budget | Uses inner-fold data only; no test leakage |
| D3 | A2 (trace) evaluated post hoc on the saved E1 test predictions instead of by retraining | Compute budget | Exact for ridge and LightGBM (independent per-component fits). An approximation for RF and MLP (joint fits). The orthogonal trace-free projection is reported as well |
| D4 | E2 runs all four ML models instead of the "best two" | More complete. The planned subset is contained in it | None for the planned comparison |
| D5 | Ablations A1 (F2, F3) and A3 (uniform training weights) run for ridge, RF and LightGBM only; the MLP was omitted | Compute budget | Table 13b has no MLP rows. A3 changes training weights only; selection and test scoring keep the grid weights (`w_eval`) |
| D6 | Optional E3 (Couette flow) **not run** | Public budget files not confirmed | None |
| D7 | **Additional experiment (post hoc):** Rotta + LRR-IP with the Gibson & Launder (1978) wall-reflection term, published and refitted, in E1 and E2 | Requested during finalisation to test whether a standard wall correction changes the classical comparison | 10 additional E1 runs and 8 E2 runs. Earlier results unchanged (verified: max |ΔE_Π| = 0 over all 175 E1 and 140 E2 runs). Not part of the registered protocol |
| D8 | **Post hoc statistics** on saved predictions: unweighted E_Π, bootstrap block lengths 5/10/20 %, paired block bootstrap of model differences, tuning-convergence analysis, comparison with HJ published standard deviations | Registered in Section 8.4 but not run in the first pass | No retraining; `results/posthoc/`, `results/tuning_convergence.md`, `results/hj_std_trace.md` |
| C4 | Leakage unit test added (`test_tuning_never_touches_the_held_out_case`) | Section 9.4 claimed it existed; it did not | Test passes; the manuscript text now describes the test that exists |
| D9 | Replicate of the 100-trial MLP tuning on a MacBook Air M4 (macOS 26.2, Python 3.14.6) | Cross-machine check | Not used for reported results; selections differ at Re_τ = 550 and 5200; results/mac_mlp100/ |
| C3 | E2 script fix: tables lacked the `w_eval` column added to the shared tuner for A3, so E2 tuning crashed | Found when rerunning the MLP | Fixed; E2 ridge rerun reproduces recorded errors exactly (max |ΔE_Π| = 0) |
| C1 | **Bug fix:** the LM Re_τ = 180 wall point (stored at y/δ = 1.1e-16) was included by a `y⁺ > 0` filter | Found from Fig. 6 | All E1/E2 runs repeated with the fix. Superseded results kept in `results/superseded_wallpoint_bug/`; their effect is quantified in `results/bugfix_effect.md` |
| C2 | Diagnostics relabelled: "balance" checks are column-identity checks, not convergence checks | Found while investigating the zero-valued diagnostics | Documentation only |

## Runtime and environment

See `results/environment.txt` and `results/pip_freeze.txt`. Per-job wall-clock and tuning times are in the
`tune_time_s` and `fit_time_s` columns of `metrics_long.csv`, and in the job logs `results/log_*.txt`.
