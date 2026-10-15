"""Feature, target and weight construction (Sections 7.1-7.4 of the manuscript).

All quantities are in wall units, so nu = 1 and Re_t = k^2 / eps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io_lm import Case

FEATURE_SETS = {
    "F1": ["b11", "b22", "b12", "eta", "Re_t"],
    "F2": ["b11", "b22", "b12", "eta", "Re_t", "II_b", "III_b", "P_over_eps"],
    "F3": ["b11", "b22", "b12", "eta", "Re_t", "Re_y"],
}
TARGETS = ["Pi11", "Pi22", "Pi33", "Pi12"]
# Frobenius multiplicity of each target component (off-diagonal counted twice).
FROB_M = np.array([1.0, 1.0, 1.0, 2.0])

# Columns that must never be used as features (leakage / not RSM-available).
FORBIDDEN_FEATURES = {"y_plus", "y_delta", "re_tau", "production", "turb_transport",
                      "visc_transport", "pressure_transport", "visc_dissipation", "balance"}


def region_of(y_plus: np.ndarray, re_tau: float) -> np.ndarray:
    r = np.full(y_plus.shape, "outer", dtype=object)
    r[y_plus < 0.2 * re_tau] = "log"
    r[y_plus < 30] = "buffer"
    r[y_plus < 5] = "viscous"
    return r


def grid_weights(y: np.ndarray) -> np.ndarray:
    """Trapezoidal grid-spacing weights, normalised to sum to one."""
    w = np.empty_like(y)
    w[1:-1] = 0.5 * (y[2:] - y[:-2])
    w[0] = 0.5 * (y[1] - y[0])
    w[-1] = 0.5 * (y[-1] - y[-2])
    w = np.clip(w, 0, None)
    return w / w.sum()


def build_table(case: Case, drop_wall: bool = True) -> pd.DataFrame:
    """One row per wall-normal point with features, targets, weights and metadata."""
    k, eps, dU = case.k, case.eps, case.dUdy
    keep = np.ones_like(k, dtype=bool)
    if drop_wall:
        # Wall point: k = 0, b_ij undefined. NB the LM Re_tau=180 file stores its wall point at
        # y/delta = 1.1e-16 (y+ = 2e-14, k = 2e-34), so a `> 0` test is NOT enough (bug fixed
        # 2026-10-15). The first real grid point in any case has y+ > 1e-3.
        keep &= case.y_plus > 1e-8
    keep &= k > 0
    keep &= eps > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        b11 = case.uu / (2 * k) - 1 / 3
        b22 = case.vv / (2 * k) - 1 / 3
        b33 = case.ww / (2 * k) - 1 / 3
        b12 = case.uv / (2 * k)
        eta = k * np.abs(dU) / eps
        re_t = k ** 2 / eps
        re_y = case.y_plus * np.sqrt(k)
        ii = b11 ** 2 + b22 ** 2 + b33 ** 2 + 2 * b12 ** 2
        # III = b_ij b_jk b_ki for the channel-flow structure (b13 = b23 = 0)
        iii = b11 ** 3 + b22 ** 3 + b33 ** 3 + 3 * b12 ** 2 * (b11 + b22)
        p_eps = -case.uv * dU / eps
        pi = {c: case.budgets[c]["pressure_strain"] / eps for c in ("uu", "vv", "ww", "uv")}

    df = pd.DataFrame({
        "case": case.name, "re_tau": case.re_tau_nominal,
        "y_plus": case.y_plus, "y_delta": case.y_delta,
        "b11": b11, "b22": b22, "b33": b33, "b12": b12,
        "eta": eta, "Re_t": re_t, "Re_y": re_y,
        "II_b": ii, "III_b": iii, "P_over_eps": p_eps,
        "Pi11": pi["uu"], "Pi22": pi["vv"], "Pi33": pi["ww"], "Pi12": pi["uv"],
        "eps": eps, "k": k, "dUdy": dU,
    })[keep].reset_index(drop=True)
    df["w"] = grid_weights(df["y_delta"].to_numpy())
    df["region"] = region_of(df["y_plus"].to_numpy(), case.re_tau_nominal)
    if not np.isfinite(df[FEATURE_SETS["F2"] + ["Re_y"] + TARGETS].to_numpy()).all():
        raise ValueError(f"{case.name}: non-finite feature/target values after filtering")
    return df


def check_no_leakage(feature_cols: list[str]) -> None:
    bad = set(feature_cols) & FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"Forbidden feature(s) requested: {sorted(bad)}")
