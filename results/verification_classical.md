# Verification of the classical closures

## SSG (Speziale, Sarkar & Gatski)

- **Primary source checked:** ICASE Report No. 90-5, NASA CR-181979 (January 1990), "Modeling the
  pressure–strain correlation of turbulence: an invariant dynamical systems approach". This is the
  report version of the JFM 227, 245–272 (1991) paper.
  <https://ntrs.nasa.gov/api/citations/19900008687/downloads/19900008687.pdf>
- **Constants (report Eqs. 67–69, read from the scan):** C1 = 3.4, C1* = 1.80, C2 = 4.2, C3* = 1.30,
  C4 = 1.25, C5 = 0.40.
  - C3 = 0.8 did not come through in the scan. OpenFOAM v12 `SSG.H`, which cites the 1991 JFM paper,
    gives C3 = 0.8, and so does the NASA TMR form written in a_ij = 2b_ij (C3 = 0.8 is the same in
    both scalings).
- **Model form (report Eq. 62), as implemented:**

  Π_ij = −(C1 ε + C1* P) b_ij + C2 ε (b_ik b_kj − ⅓ II δ_ij) + (C3 − C3* II^½) k S_ij
         + C4 k (b_ik S_jk + b_jk S_ik − ⅔ b_mn S_mn δ_ij) + C5 k (b_ik W_jk + b_jk W_ik)

  with II = b_ij b_ij, P = −R_ij ∂U_i/∂x_j and W_ij = ½(∂U_i/∂x_j − ∂U_j/∂x_i).
- **Rotation-term sign.** The scan is too garbled to show the W_ij convention explicitly.
  - OpenFOAM v12 `SSG.C` writes the term as `C5*twoSymm(b & skew(gradU))`, with OpenFOAM's
    gradU_ij = ∂U_j/∂x_i. That is exactly C5 k (b_ik W_jk + b_jk W_ik) with W as defined above.
  - `tests/test_core.py::test_ssg_matches_openfoam_convention_including_rotation_sign` reproduces the
    OpenFOAM expression in OpenFOAM's own index convention and matches our channel-flow reduction to
    1e-10. It fails if the sign of C5 is flipped.

## LRR-IP (isotropisation of production) — updated 2026-10-15 (finalisation pass)

- **Form:** Π_ij = −C_R ε (R_ij/k − ⅔ δ_ij) − C_P (P_ij − ⅓ P_kk δ_ij), with C_R = 1.8 and C_P = 0.6.
- **Bibliographic record of LRR (1975) confirmed** on Cambridge Core: J. Fluid Mech. 68(3), 537–566,
  doi:10.1017/S0022112075001814. The full text (and therefore the printed equations and constants) could
  NOT be opened from this environment. The constants are therefore verified against secondary sources only:
  - OpenFOAM `LRR.C` defaults: C1 = 1.8, C2 = 0.6 (header cites LRR 1975 and Gibson & Launder 1978).
  - ANSYS CFX-Solver Theory Guide (Release 2025 R2, Reynolds stress models), coefficient table for
    LRR-IP in its general form with a_ij = 2b_ij: Cs1 = 1.8, Cr2 = 0.8, Cr4 = Cr5 = 0.6 (Cr1 = Cr3 = 0).
  - Penn State ME 522 lecture notes (Cimbala, 2008), "Reynolds stress models": C1 = 1.8, C2 = 0.60 for LRR.
- **Independent algebraic check (new unit test).** With P_ij expanded in b, S and W, LRR-IP equals the SSG
  tensor form with (C1, C1*, C2, C3, C3*, C4, C5) = (2C_R, 0, 0, 4/3·C_P, 0, 2C_P, 2C_P) = (3.6, 0, 0, 0.8,
  0, 1.2, 1.2). In the a = 2b scaling this is exactly the ANSYS CFX row above. The test
  `test_lrr_ip_equals_ssg_form_with_mapped_coefficients` passes to 1e-10 and fails if the C5 sign is
  flipped. Because LRR-IP is built from the DNS production tensor directly, it has no separate
  rotation-sign choice; the test therefore also confirms, independently of OpenFOAM, the W_ij sign
  convention used in our SSG implementation.
- **Still open:** reading the constants in the printed LRR (1975) paper itself. Note that LRR (1975)
  presents more than one variant; the IP constants used here are those of the simplified (IP) model as
  reproduced by the three sources above.

## Wall-corrected LRR-IP (Gibson & Launder 1978) — additional experiment

- Gibson & Launder (1978): J. Fluid Mech. 86(3), 491–511, doi:10.1017/S0022112078001251 (record confirmed
  on Cambridge Core; full text not opened).
- Implemented form (as in OpenFOAM LRR.C, `wallReflection yes`; Cref1 = 0.5, Cref2 = 0.3, κ = 0.41,
  Cμ = 0.09):
  φ^w_ij = f (X_km n_k n_m δ_ij − 3/2 X_ik n_k n_j − 3/2 X_jk n_k n_i),
  X = Cref1 (ε/k) R + Cref2 φ_2, φ_2 = −C_P dev(P), f = Cμ^{3/4} k^{3/2}/(κ ε y).
  OpenFOAM adds 3 dev(symm((n·X) n)) · Cμ^{3/4} √k/(κ y) to the left-hand side, which is the same term.
- Unit test `test_gl_wall_reflection_matches_general_tensor_form_and_is_trace_free` checks the channel
  reduction against the general tensor form.
- Results: `results/e1_F1/tables.md` and `results/e2/table12_E2.md` (rows "Rotta+LRR-IP+GL wall").

## SSG with a wall correction — not run

SSG (1991) was formulated without a wall-echo term, and no single standard algebraic wall correction for it
was found. The near-wall SSG variants in use (e.g. elliptic-blending Reynolds-stress models) are different
models that need an additional elliptic equation and version-specific constants. They were not
implemented; this is a stated limitation.

## Other checks

- **Budget normalisation.** Budgets are for R_ij itself: P11 = −2⟨uv⟩dU/dy and P12 = −⟨vv⟩dU/dy hold to
  better than 1e-5 (Lee & Moser) and 1e-3 (Hoyas & Jiménez).
- **Wall units.** The total shear stress dU⁺/dy⁺ − ⟨uv⟩⁺ = 1 − y/δ holds to within 0.3 % in both
  databases.
- **Reynolds-number matching for E2** (actual Re_τ):

  | Hoyas & Jiménez | Lee & Moser (nearest) |
  |---|---|
  | 186 | 182 |
  | 547 | 544 |
  | 934 | none (interior; between 544 and 1001) |
  | 2003 | 1995 |

- **Classical closures on both databases (published SSG E_Π):** 0.398 (Hoyas & Jiménez, 547) against
  0.406 (Lee & Moser, 544). This is consistent with the two databases describing the same physics.
