"""Reader for the Hoyas & Jimenez (2008) channel statistics (E2 external test set).

Layout (from the file headers, verified numerically in validate_hj):
  ReNNN/profiles/ReNNN.prof : y/h, y+, U+, u'+, v'+, w'+ (rms), -Om_z+ (= dU+/dy+), om_x', om_y', om_z',
                              uv'+, uw'+, vw'+, pr'+, ps'+, psto'+, p'
  ReNNN/balances/ReNNN.bal.{uu,vv,ww,uv,kbal}:
                              y/h, y+, dissip, produc, p-strain, p-diff, t-diff, v-diff, bal, flux
  Budget terms are SIGNED and ADDED (bal = sum of columns 2..7); dissip is negative for normal stresses.
Normalisation: u_tau, nu/u_tau (wall units), as in Lee & Moser.

The returned Case uses the same conventions as io_lm (eps > 0; budgets[c]['pressure_strain']),
so features.build_table works unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_lm import Case, sha256

BAL = {"y_delta": 0, "y_plus": 1, "dissip": 2, "production": 3, "pressure_strain": 4,
       "pressure_transport": 5, "turb_transport": 6, "visc_transport": 7, "balance": 8}
PROF = {"y_delta": 0, "y_plus": 1, "U": 2, "urms": 3, "vrms": 4, "wrms": 5, "dUdy": 6, "uv": 10}
EXT = {"uu": "uu", "vv": "vv", "ww": "ww", "uv": "uv", "k": "kbal"}


def load_hj_case(root: Path, re: int) -> Case:
    root = Path(root)
    pf = root / f"Re{re}" / "profiles" / f"Re{re}.prof"
    prof = np.loadtxt(pf, comments="%")
    sums = {pf.name: sha256(pf)}
    budgets = {}
    for c, ext in EXT.items():
        f = root / f"Re{re}" / "balances" / f"Re{re}.bal.{ext}"
        a = np.loadtxt(f, comments="%")
        sums[f.name] = sha256(f)
        if a.shape[0] != prof.shape[0] or not np.allclose(a[:, 0], prof[:, 0], atol=1e-6):
            raise ValueError(f"{f.name}: grid differs from profile grid")
        b = {t: a[:, j].copy() for t, j in BAL.items()}
        b["visc_dissipation"] = -b["dissip"]          # positive-dissipation convention of io_lm
        budgets[c] = b
    uu, vv, ww = prof[:, 3] ** 2, prof[:, 4] ** 2, prof[:, 5] ** 2
    k = 0.5 * (uu + vv + ww)
    y = prof[:, 0]
    keep = y <= 1.0 + 1e-9                             # half channel (files hold the lower half)
    case = Case(name=f"HJ_Channel_{re:04d}", re_tau_nominal=int(re),
                y_delta=y[keep], y_plus=budgets["k"]["y_plus"][keep], U=prof[keep, 2], dUdy=prof[keep, 6],
                uu=uu[keep], vv=vv[keep], ww=ww[keep], uv=prof[keep, 10], k=k[keep],
                eps=-budgets["k"]["dissip"][keep], checksums=sums)
    case.budgets = {c: {t: v[keep] for t, v in b.items()} for c, b in budgets.items()}
    return case


def validate_hj(case: Case) -> dict:
    d, B = {}, case.budgets
    for c, b in B.items():
        s = sum(b[t] for t in ("dissip", "production", "pressure_strain", "pressure_transport",
                               "turb_transport", "visc_transport"))
        scale = max(np.abs(b[t]).max() for t in ("dissip", "production", "pressure_strain")) + 1e-30
        d[f"balance_column_identity_{c}"] = float(np.max(np.abs(s - b["balance"])) / scale)
        d[f"budget_residual_over_eps_median_{c}"] = float(np.median(np.abs(b["balance"][1:]) / case.eps[1:]))
    dU, uv, vv = case.dUdy, case.uv, case.vv
    m = np.abs(uv * dU) > 0.05 * np.abs(uv * dU).max()
    d["P_uu_over_-2uvdU"] = float(np.median(B["uu"]["production"][m] / (-2 * uv[m] * dU[m])))
    d["P_uv_over_-vvdU"] = float(np.median(B["uv"]["production"][m] / (-vv[m] * dU[m])))
    d["P_k_over_-uvdU"] = float(np.median(B["k"]["production"][m] / (-uv[m] * dU[m])))
    d["eps_k_over_half_sum"] = float(np.median(case.eps[1:] / (-0.5 * sum(B[c]["dissip"][1:] for c in ("uu", "vv", "ww")))))
    tr = sum(B[c]["pressure_strain"] for c in ("uu", "vv", "ww"))
    nrm = np.sqrt(sum(B[c]["pressure_strain"] ** 2 for c in ("uu", "vv", "ww")) + 2 * B["uv"]["pressure_strain"] ** 2)
    d["dns_trace_rel_median"] = float(np.median(np.abs(tr[1:]) / nrm[1:]))
    d["dns_trace_rel_max"] = float(np.max(np.abs(tr[1:]) / nrm[1:]))
    d["total_stress_max_dev"] = float(np.max(np.abs(case.dUdy - case.uv - (1 - case.y_delta))))
    d["Re_tau_from_grid"] = float(case.y_plus[-1] / case.y_delta[-1])
    for name, cond in (("P_uu", abs(d["P_uu_over_-2uvdU"] - 1) < 0.02), ("P_uv", abs(d["P_uv_over_-vvdU"] - 1) < 0.02),
                       ("eps", abs(d["eps_k_over_half_sum"] - 1) < 0.02), ("total_stress", d["total_stress_max_dev"] < 0.02)):
        if not cond:
            raise ValueError(f"{case.name}: consistency check {name} failed: {d}")
    return d
