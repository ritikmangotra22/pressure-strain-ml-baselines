# Paper 3 timing pilot — summary (20 Oct 2026)

Scope: 26 configurations (20 M4, 2 each of M1, M2, M3) drawn uniformly from the draft search space
(seed 3003). Each scored by inner leave-one-Re_tau-out over the five Lee & Moser cases, 2 training
seeds, no pruning. No Hoyas & Jimenez or Moser-Kim-Mansour data were loaded. Not used for any claim.

## Timing (Linux x86-64, 2 cores, one thread per job)

- 26/26 completed, 0 failed fits.
- One evaluation (5 inner folds x 2 seeds = 10 fits): mean 97 s, median 82 s, max 299 s (width 128, 4-5 layers).
- Mean 9.7 s per fit.

## Seed noise vs spread between configurations (M4, 20 configs)

| Objective | Median seed SD | Range across configs | SD / range |
|---|---|---|---|
| f_err | 0.0050 | 0.067 – 0.220 | 0.03 |
| f_gen | 0.0167 | 0.160 – 0.657 | 0.03 |
| return-to-isotropy violation | 0.0003 | 0 – 0.039 | 0.01 |
| coefficient norm | 0.10 | 3.2 – 9.7 | 0.02 |

Seed noise is small relative to differences between configurations, so one training seed per
candidate is enough to rank candidates; seed spread is still measured at the 10-seed refit.

## Objective relationships (M4, Spearman)

- f_err vs f_cost: -0.53 (larger networks more accurate): a genuine trade-off.
- f_err vs f_gen: 0.76: related but not redundant.
- Return-to-isotropy violation vs f_err: +0.50 (no conflict: better fits violate less).
  Violations are non-zero in 15 of 20 M4 configurations, all small (max 0.039).
- Coefficient norm is dominated by basis size: about 3.3-3.8 with 8 tensors, 7.1-9.7 with 3.
- Arm B separates clearly: rotated-frame excess error 0.85-1.21 for M1/M3, 0 for M2/M4;
  mean relative trace about 0.05 for M1/M2, 0 for M3/M4.

## Design issue found

The worst inner fold is Re_tau = 180 in 24 of 26 configurations (5200 in the other 2). Median
M4 inner errors: 180: 0.186, 550: 0.058, 1000: 0.035, 2000: 0.027, 5200: 0.080. As defined,
f_gen is therefore almost the same as "error at Re_tau = 180", a single low-Reynolds-number case.
Options for the protocol:

1. keep max (honest worst case, but effectively one case);
2. mean of the two extrapolation folds (180 and 5200);
3. max over folds of E_Pi(c) / E_ref(c), with E_ref the M0 reference error on that fold
   (worst relative degradation; no fold dominates by construction).

## Budget implied

With 1 seed per candidate (about 49 s per evaluation): primary setting 7 strategies x 5
optimizer seeds x 200 evaluations = 7000 evaluations, about 95 core-hours, about 2 days on 2 cores.
With 2 seeds: about 190 core-hours, about 4 days. The secondary comparability setting and the
ablations add roughly the same again.
