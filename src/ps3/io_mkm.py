"""Reader for the Moser, Kim & Mansour (1999) channel statistics (Paper 3 confirmatory test set).

Source: https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz (downloaded 21 Oct 2026),
unpacked into data/raw/moser_kim_mansour_1999 (balances and profiles only; SHA256SUMS.txt).

Layout (from the file headers; checked numerically by ps.io_hj.validate_hj):
  chanNNN/profiles/chanNNN.means     : y/h, y+, Umean, dUmean/dy, Wmean, dWmean/dy, Pmean
  chanNNN/profiles/chanNNN.reystress : y/h, y+, R_uu, R_vv, R_ww, R_uv, R_uw, R_vw
  chanNNN/balances/chanNNN.{uu,vv,ww,uv,k}bal :
                                       y/h, y+, dissip, produc, p-strain, p-diff, t-diff, v-diff, bal
Normalisation: u_tau, nu/u_tau (wall units). Lower half channel, 0 <= y/h <= 1, ny = 129 points
per half. Budget columns are SIGNED and ADDED, as in Hoyas & Jimenez, so the returned Case uses
the io_lm conventions (eps > 0, budgets[c]['pressure_strain']) and ps.features.build_table works
unchanged.

This module reads data only. It is used to build and hash the confirmatory table BEFORE any
Paper 3 model is evaluated on it; models are evaluated on it only by the confirmatory-test script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ps.io_hj import BAL
from ps.io_lm import Case, sha256

EXT = {"uu": "uubal", "vv": "vvbal", "ww": "wwbal", "uv": "uvbal", "k": "kbal"}
NOMINAL = {180: "MKM_Channel_0180", 395: "MKM_Channel_0395", 590: "MKM_Channel_0590"}


def load_mkm_case(root: Path, re: int) -> Case:
    root = Path(root) / f"chan{re}"
    mf, rf = root / "profiles" / f"chan{re}.means", root / "profiles" / f"chan{re}.reystress"
    means, rs = np.loadtxt(mf, comments="#"), np.loadtxt(rf, comments="#")
    sums = {mf.name: sha256(mf), rf.name: sha256(rf)}
    if means.shape[0] != rs.shape[0] or not np.allclose(means[:, 0], rs[:, 0], atol=1e-8):
        raise ValueError(f"chan{re}: means and reystress grids differ")
    budgets = {}
    for c, ext in EXT.items():
        f = root / "balances" / f"chan{re}.{ext}"
        a = np.loadtxt(f, comments="#")
        sums[f.name] = sha256(f)
        if a.shape[0] != means.shape[0] or not np.allclose(a[:, 0], means[:, 0], atol=1e-8):
            raise ValueError(f"{f.name}: grid differs from profile grid")
        b = {t: a[:, j].copy() for t, j in BAL.items()}
        b["visc_dissipation"] = -b["dissip"]
        budgets[c] = b
    uu, vv, ww, uv = rs[:, 2], rs[:, 3], rs[:, 4], rs[:, 5]
    # The means file gives dU+/d(y/h) (it equals Re_tau at the wall), not dU+/dy+: divide by the
    # actual Re_tau = y+/(y/h). Found by validate_hj (production ratio was 1/Re_tau before this).
    re_tau = means[-1, 1] / means[-1, 0]
    case = Case(name=NOMINAL[re], re_tau_nominal=int(re), y_delta=means[:, 0], y_plus=means[:, 1],
                U=means[:, 2], dUdy=means[:, 3] / re_tau, uu=uu, vv=vv, ww=ww, uv=uv, k=0.5 * (uu + vv + ww),
                eps=-budgets["k"]["dissip"], checksums=sums)
    case.budgets = budgets
    return case
