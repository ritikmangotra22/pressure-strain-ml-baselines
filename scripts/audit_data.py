"""Independent audit of the Lee & Moser files (does not reuse io_lm column maps).

Reads every file with numpy.loadtxt, prints UNROUNDED diagnostics, and checks
physics that the pipeline relies on. Output: results/data_audit.txt
"""
from pathlib import Path

import numpy as np

D = Path("data/raw/lee_moser_2015")
LAB = ["0180", "0550", "1000", "2000", "5200"]
out = []
p = lambda *a: out.append(" ".join(str(x) for x in a))


def L(lab, tag):
    return np.loadtxt(D / f"LM_Channel_{lab}_{tag}.dat", comments="%")


for lab in LAB:
    mean, fl = L(lab, "mean_prof"), L(lab, "vel_fluc_prof")
    R = {c: L(lab, f"RSTE_{c}_prof") for c in ("uu", "vv", "ww", "uv", "k")}
    y, yp = mean[:, 0], mean[:, 1]
    U, dU = mean[:, 2], mean[:, 3]
    uu, vv, ww, uv, uw, vw, k = (fl[:, j] for j in range(2, 9))
    i = slice(1, None)  # exclude wall point
    p(f"\n===== Re_tau label {lab}: {len(y)} points, y/delta {y[0]:.3g}..{y[-1]:.6f}, Re_tau = y+/y = {yp[-1]/y[-1]:.2f}")

    # (a) k: provider-derived column -> identity expected (not an independent check)
    p("k - 0.5(uu+vv+ww): max abs", np.abs(k - 0.5 * (uu + vv + ww)).max())

    # (b) Balance column: provider-derived. Recompute and compare; then report balance magnitude.
    for c in ("uu", "vv", "ww", "uv", "k"):
        a = R[c]
        s = a[:, 2] + a[:, 3] + a[:, 4] + a[:, 5] + a[:, 6] - a[:, 7]
        eps = R["k"][:, 7]
        p(f"{c}: |recomputed - balance col| max {np.abs(s - a[:, 8]).max():.3e} ;"
          f" |balance|/eps median {np.median(np.abs(a[i, 8]) / eps[i]):.3e} max {np.max(np.abs(a[i, 8]) / eps[i]):.3e}")

    # (c) eps from k file vs half the sum of normal-stress dissipations: definitional identity
    eps = R["k"][:, 7]
    p("eps_k - 0.5(eps11+eps22+eps33): max abs", np.abs(eps - 0.5 * (R["uu"][:, 7] + R["vv"][:, 7] + R["ww"][:, 7])).max(),
      "| eps>0 everywhere:", bool((eps > 0).all()), "| eps_wall+ =", eps[0])
    # k-file production / pressure strain / pressure transport vs half-sums
    for j, nm in ((2, "production"), (5, "pressure_strain"), (6, "pressure_transport")):
        p(f"k-file {nm} - 0.5*sum(normal): max abs",
          np.abs(R["k"][:, j] - 0.5 * (R["uu"][:, j] + R["vv"][:, j] + R["ww"][:, j])).max())

    # (d) Pressure-strain trace, unrounded, absolute and relative
    tr = R["uu"][:, 5] + R["vv"][:, 5] + R["ww"][:, 5]
    nrm = np.sqrt(R["uu"][:, 5] ** 2 + R["vv"][:, 5] ** 2 + R["ww"][:, 5] ** 2 + 2 * R["uv"][:, 5] ** 2)
    p(f"trace Pi_ii: max abs {np.abs(tr).max():.3e}, max |tr|/||Pi|| {np.max(np.abs(tr[i]) / nrm[i]):.3e},"
      f" typical ||Pi|| {np.median(nrm):.3e}")
    # pressure transport trace: -2 d<p u_i>/dx_i = 0 by continuity as well
    trp = R["uu"][:, 6] + R["vv"][:, 6] + R["ww"][:, 6]
    p(f"pressure transport: uu max abs {np.abs(R['uu'][:, 6]).max():.3e} (zero by x-homogeneity);"
      f" ww max abs {np.abs(R['ww'][:, 6]).max():.3e} (zero by z-homogeneity); vv max abs {np.abs(R['vv'][:, 6]).max():.3e}")

    # (e) Budget normalisation (equation for R_ij, not R_ij/2)
    m = np.abs(uv * dU) > 0.05 * np.abs(uv * dU).max()
    p("P_uu / (-2 uv dU/dy) median", np.median(R["uu"][m, 2] / (-2 * uv[m] * dU[m])),
      "| P_uv / (-vv dU/dy) median", np.median(R["uv"][m, 2] / (-vv[m] * dU[m])),
      "| P_vv, P_ww max abs", np.abs(R["vv"][:, 2]).max(), np.abs(R["ww"][:, 2]).max(),
      "| P_k / (-uv dU) median", np.median(R["k"][m, 2] / (-uv[m] * dU[m])))
    # (f) Wall-unit normalisation: dU+/dy+ = 1 at wall, total shear = 1 - y/delta
    tot = dU - uv  # nu dU/dy - <uv> in wall units (Re_tau-based: = 1 - y/delta)
    p(f"dU/dy at wall {dU[0]:.6f} | max |(dU/dy - uv) - (1 - y/delta)| {np.max(np.abs(tot - (1 - y))):.3e}")
    # (g) Symmetry quality indicators
    p(f"max |uw| {np.abs(uw).max():.2e}  max |vw| {np.abs(vw).max():.2e}  max|W| {np.abs(mean[:,4]).max():.2e}")
    # (h) Physical signs of pressure strain (away from wall, y+ in 15..0.5Re)
    zone = (yp > 15) & (yp < 0.5 * yp[-1] / y[-1])
    p(f"signs for 15<y+<0.5Re_tau: Pi11<0 frac {np.mean(R['uu'][zone,5]<0):.3f}, Pi22>0 frac {np.mean(R['vv'][zone,5]>0):.3f},"
      f" Pi33>0 frac {np.mean(R['ww'][zone,5]>0):.3f}, Pi12>0 frac {np.mean(R['uv'][zone,5]>0):.3f}, uv<0 frac {np.mean(uv[zone]<0):.3f}")
    # near-wall splatting: Pi22 < 0 very close to the wall is a known DNS feature
    nw = (yp > 0) & (yp < 10)
    p(f"near wall y+<10: min Pi22 {R['vv'][nw,5].min():.4f} at y+ {yp[nw][np.argmin(R['vv'][nw,5])]:.2f}; Pi22 at wall {R['vv'][0,5]:.2e}")
    # (i) Anisotropy and realizability of the DNS input
    with np.errstate(divide="ignore", invalid="ignore"):
        b11, b22, b33, b12 = uu / (2 * k) - 1 / 3, vv / (2 * k) - 1 / 3, ww / (2 * k) - 1 / 3, uv / (2 * k)
    p(f"b trace max abs {np.nanmax(np.abs((b11 + b22 + b33)[i])):.2e}; b11 range [{np.nanmin(b11[i]):.3f},{np.nanmax(b11[i]):.3f}],"
      f" b22 min {np.nanmin(b22[i]):.4f} (>= -1/3 required), Cauchy-Schwarz uv^2<=uu*vv: {bool(np.all(uv[i]**2 <= uu[i]*vv[i]*(1+1e-9)))}")

Path("results").mkdir(exist_ok=True)
Path("results/data_audit.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
