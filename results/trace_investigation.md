# Pressure–strain trace in the two DNS databases

Relative trace = |Π11 + Π22 + Π33| / ‖Π‖_F, at each point excluding the wall. Computed by
`scripts/audit_data.py` (Lee & Moser) and `src/ps/io_hj.py::validate_hj` (Hoyas & Jiménez).

## Verified facts

| Database | Re_τ | Median relative trace | Maximum (location) | Bound from 8-digit ASCII output |
|---|---|---|---|---|
| Lee & Moser | 182 | ~1e-16 | 7e-16 | n/a (16-digit output) |
| Lee & Moser | 544, 1001, 1995, 5186 | ~1e-16 | ≤ 1.2e-15 | n/a |
| Hoyas & Jiménez | 186 | 3.2e-7 | 3.0e-5 (y⁺ ≈ 0.1) | 6.4e-8 (77 % of points exceed it) |
| Hoyas & Jiménez | 547 | 5.8e-7 | 2.5e-5 (first point off the wall) | 5.6e-8 (95 % exceed) |
| Hoyas & Jiménez | 934 | 2.9e-6 | 1.5e-4 (first point off the wall) | 5.4e-8 (99 % exceed) |
| Hoyas & Jiménez | 2003 | 1.5e-8 | 5.4e-8 | 5.4e-8 (0 % exceed) |

1. **Lee & Moser method.** Lee & Moser (2015) solve the Kim–Moin–Moser (1987) velocity–vorticity
   equations. In their words, "This formulation has the advantage of satisfying the continuity
   constraint exactly while eliminating the pressure" (arXiv:1410.7809, §2). Discretisation is
   Fourier–Galerkin in x and z, with seventh-order B-spline collocation in y. Their k-budget file
   states: "Analytic values of Pressure_Strain are zero everywhere."
2. **Lee & Moser data.** The published trace is at round-off (≤ 1.2e-15 relative) in all five cases.
3. **Hoyas & Jiménez method.** Their file headers cite the same Kim–Moin–Moser formulation.
   - Re_τ = 934 (del Álamo et al. 2004): Chebyshev spectral in y.
   - Re_τ = 2003 (Hoyas & Jiménez 2006): compact finite differences in y.
   - Fourier with 2/3 dealiasing in x and z for both (arXiv:1806.08678; APS DFD 2005 abstract LR.4).
4. **Hoyas & Jiménez files** are written with 8 significant digits.
   - At Re_τ = 2003 the trace is entirely within that rounding.
   - At 186, 547 and 934 it exceeds rounding at most points, and is largest at or next to the wall.
5. **Size relative to model errors.** In both databases the trace is at most 1.5e-4 of ‖Π‖. That is
   at least two orders of magnitude below every model error in this study (E_Π ≥ ~1e-2).

## Not verified (hypotheses)

- **Lee & Moser.** Exact discrete continuity in the solver is *consistent with* a round-off trace.
  The files alone cannot show whether Π33 was computed independently or set to −(Π11 + Π22) in
  post-processing; both give the same data.
- **Hoyas & Jiménez.** Because their solver uses the same continuity-exact formulation, solver
  continuity alone does **not** explain the non-zero trace at 186, 547 and 934. Possible causes, none
  verified:
  - statistics accumulated with derivative operators that differ from the solver's;
  - accumulation on a different (e.g. dealiased physical) grid;
  - averaging over different sample sets.
  The near-wall location of the maximum fits the first two, but this has not been confirmed with
  the authors.
- The y-discretisation does not explain the pattern either. The compact-finite-difference case
  (2003) has the smallest trace, and the Chebyshev case (934) the largest.

## Consequence for the study

- The training target (Lee & Moser) is exactly trace-free. Any model trace residual above about
  1e-15 is a property of the model.
- In E2, the external Hoyas & Jiménez targets carry a trace of up to 1.5e-4 relative. That is
  negligible against model errors, and it is reported in Table 7.

## Update (finalisation pass): comparison with the published standard deviations

Script `scripts/hj_std_analysis.py`, output `results/hj_std_trace.md`. Hoyas & Jiménez publish a
standard deviation σ for every budget term (Re*.std.<comp>). Away from the first off-wall point, the
trace is at most 0.14 σ_tr (σ_tr = √(σ11² + σ22² + σ33²)) at every point of every case. It exceeds
σ_tr only at the first point off the wall at Re_τ = 547 (1.2 σ_tr) and 934 (6.4 σ_tr). The trace is
therefore within the published statistical scatter almost everywhere; the first off-wall points at
547 and 934 remain unexplained. The README does not say whether σ is a sample standard deviation or a
standard error, so this comparison is indicative only.
