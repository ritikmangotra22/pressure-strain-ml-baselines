# Anomalies investigated during E1

## Ridge, held-out Re_τ = 2000: E_Π = 0.30, against 0.02 at Re_τ = 1000

- **Cause.** Nested tuning picked degree 1 (linear) with α ≈ 65. The inner leave-one-out
  validation runs over {180, 550, 1000, 5200}, and two of those inner folds (180 and 5200) are
  extrapolations. Quadratic features extrapolate badly, so they lose on the mean inner error
  (inner mean E_Π of the selected model = 0.376).
- **Check, after the fact (not used for selection).** With degree 2 and small α, the test error
  on Re_τ = 2000 would be about 0.02. Linear models give 0.16–0.31, depending on α.
- **Status.** Not a bug. This is a property of the registered selection rule: mean inner error,
  with inner folds that mix interpolation and extrapolation. It is reported in the Discussion and
  the protocol is left unchanged. Picking degree 2 from the test number would leak test
  information into model selection.
- **Reproduce.** `tune('ridge', ...)` with the E1 configuration; the scan is in the session log.
