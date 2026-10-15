"""Evaluation metrics (manuscript Section 8.3-8.4). Inputs are (n, 4) arrays in
component order (11, 22, 33, 12) of Pi/eps, plus point weights w (any scale)."""
from __future__ import annotations

import numpy as np

M = np.array([1.0, 1.0, 1.0, 2.0])
COMP = ["Pi11", "Pi22", "Pi33", "Pi12"]


def _w(w):
    w = np.asarray(w, float)
    return w / w.sum()


def frob2(a: np.ndarray) -> np.ndarray:
    return (a ** 2 * M).sum(axis=1)


def e_pi(y_hat, y, w) -> float:
    w = _w(w)
    return float(np.sqrt((w * frob2(y_hat - y)).sum() / (w * frob2(y)).sum()))


def per_component(y_hat, y, w) -> dict:
    w = _w(w)
    out = {}
    for j, c in enumerate(COMP):
        e = y_hat[:, j] - y[:, j]
        ybar = (w * y[:, j]).sum()
        ss_tot = (w * (y[:, j] - ybar) ** 2).sum()
        out[f"{c}_MAE"] = float((w * np.abs(e)).sum())
        out[f"{c}_RMSE"] = float(np.sqrt((w * e ** 2).sum()))
        out[f"{c}_R2"] = float(1 - (w * e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    return out


def weighted_quantile(x, w, q):
    idx = np.argsort(x)
    x, w = x[idx], _w(w)[idx]
    cw = np.cumsum(w)
    return float(np.interp(q, cw - 0.5 * w, x))


def pointwise_relative(y_hat, y, w, floor_frac=0.05) -> dict:
    nrm = np.sqrt(frob2(y))
    tau = floor_frac * nrm.max()
    r = np.sqrt(frob2(y_hat - y)) / np.maximum(nrm, tau)
    return {f"r_median_f{floor_frac}": weighted_quantile(r, w, 0.5),
            f"r_p90_f{floor_frac}": weighted_quantile(r, w, 0.9)}


def trace_residual(y_hat, w) -> dict:
    t = np.abs(y_hat[:, 0] + y_hat[:, 1] + y_hat[:, 2]) / np.maximum(np.sqrt(frob2(y_hat)), 1e-30)
    return {"trace_mean": float((_w(w) * t).sum()), "trace_max": float(t.max())}


def block_bootstrap_e_pi(y_hat, y, w, block_frac=0.10, n_boot=2000, seed=0) -> tuple[float, float]:
    """95% CI of E_Pi using a moving-block bootstrap along the profile."""
    rng = np.random.default_rng(seed)
    n = len(y)
    L = max(2, int(round(block_frac * n)))
    starts_max = n - L
    nblocks = int(np.ceil(n / L))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=nblocks)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        vals[b] = e_pi(y_hat[idx], y[idx], w[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def all_metrics(y_hat, y, w, regions=None) -> dict:
    out = {"E_Pi": e_pi(y_hat, y, w)}
    out.update(per_component(y_hat, y, w))
    out.update(pointwise_relative(y_hat, y, w, 0.05))
    out.update(pointwise_relative(y_hat, y, w, 0.01))
    out.update(trace_residual(y_hat, w))
    if regions is not None:
        for reg in ("viscous", "buffer", "log", "outer"):
            m = regions == reg
            out[f"E_Pi_{reg}"] = e_pi(y_hat[m], y[m], w[m]) if m.sum() >= 2 else np.nan
    return out
