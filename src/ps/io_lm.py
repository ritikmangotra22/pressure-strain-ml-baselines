"""Readers for the Lee & Moser (2015) channel-flow statistics files.

Files expected per case (Re_tau label NNNN = 0180, 0550, 1000, 2000, 5200):
    LM_Channel_NNNN_mean_prof.dat       mean velocity profile
    LM_Channel_NNNN_vel_fluc_prof.dat   velocity covariances and k
    LM_Channel_NNNN_RSTE_{uu,vv,ww,uv,k}_prof.dat   Reynolds-stress budgets

Column order is NOT trusted blindly. Every reader is followed by a numerical
consistency check (see `validate_case`) that fails loudly if the assumed
columns do not reproduce known identities (k = trace/2, budget balance,
dU/dy against finite differences of U).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Assumed column layouts (0-based). Verified at load time by validate_case().
RSTE_COLS = {
    "y_delta": 0, "y_plus": 1, "production": 2, "turb_transport": 3,
    "visc_transport": 4, "pressure_strain": 5, "pressure_transport": 6,
    "visc_dissipation": 7, "balance": 8,
}
MEAN_COLS = {"y_delta": 0, "y_plus": 1, "U": 2, "dUdy": 3}
FLUC_COLS = {"y_delta": 0, "y_plus": 1, "uu": 2, "vv": 3, "ww": 4, "uv": 5, "k": 8}

COMPONENTS = ("uu", "vv", "ww", "uv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dat(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a whitespace-separated numeric file; return (array, header_lines).

    Lines that do not parse as all-numeric rows are treated as header/comments.
    """
    rows, header = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                header.append(s)
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    ncol = {len(r) for r in rows}
    if len(ncol) != 1:
        raise ValueError(f"Inconsistent column counts {ncol} in {path}")
    return np.asarray(rows, dtype=float), header


@dataclass
class Case:
    """All profiles of one DNS case, in wall units (u_tau = nu = 1)."""
    name: str
    re_tau_nominal: int
    y_delta: np.ndarray
    y_plus: np.ndarray
    U: np.ndarray
    dUdy: np.ndarray          # in wall units
    uu: np.ndarray
    vv: np.ndarray
    ww: np.ndarray
    uv: np.ndarray
    k: np.ndarray
    eps: np.ndarray           # dissipation of k (from RSTE_k file)
    budgets: dict = field(default_factory=dict)   # comp -> {term: array}
    checksums: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)


def load_lm_case(data_dir: Path, re_label: str, prefix: str = "LM_Channel") -> Case:
    data_dir = Path(data_dir)
    f = lambda tag: data_dir / f"{prefix}_{re_label}_{tag}.dat"
    paths = {"mean": f("mean_prof"), "fluc": f("vel_fluc_prof")}
    for c in COMPONENTS + ("k",):
        paths[f"rste_{c}"] = f(f"RSTE_{c}_prof")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))

    arrays, headers, sums = {}, {}, {}
    for key, p in paths.items():
        arrays[key], headers[key] = read_dat(p)
        sums[p.name] = sha256(p)

    mean, fluc = arrays["mean"], arrays["fluc"]
    n = mean.shape[0]
    for key, a in arrays.items():
        if a.shape[0] != n:
            raise ValueError(f"{key}: {a.shape[0]} rows, expected {n}")
        if not np.allclose(a[:, 1], mean[:, 1], rtol=1e-8, atol=1e-10):
            raise ValueError(f"{key}: y+ grid differs from mean profile grid")

    budgets = {}
    for c in COMPONENTS + ("k",):
        a = arrays[f"rste_{c}"]
        if a.shape[1] < len(RSTE_COLS):
            raise ValueError(f"RSTE_{c}: {a.shape[1]} columns, expected >= {len(RSTE_COLS)}")
        budgets[c] = {t: a[:, j].copy() for t, j in RSTE_COLS.items()}

    case = Case(
        name=f"{prefix}_{re_label}",
        re_tau_nominal=int(re_label),
        y_delta=mean[:, MEAN_COLS["y_delta"]],
        y_plus=mean[:, MEAN_COLS["y_plus"]],
        U=mean[:, MEAN_COLS["U"]],
        dUdy=mean[:, MEAN_COLS["dUdy"]],
        uu=fluc[:, FLUC_COLS["uu"]], vv=fluc[:, FLUC_COLS["vv"]],
        ww=fluc[:, FLUC_COLS["ww"]], uv=fluc[:, FLUC_COLS["uv"]],
        k=fluc[:, FLUC_COLS["k"]],
        eps=np.abs(budgets["k"]["visc_dissipation"]),
        budgets=budgets, checksums=sums, headers=headers,
    )
    return case


def validate_case(case: Case, tol: float = 1e-3) -> dict:
    """Numerical checks that the assumed column layout is right.

    Returns a dict of diagnostics; raises ValueError on a hard failure.
    """
    d = {}
    # 1. k equals half the trace of the normal stresses.
    k_rec = 0.5 * (case.uu + case.vv + case.ww)
    rel = np.max(np.abs(k_rec - case.k)) / np.max(np.abs(case.k))
    d["k_trace_rel_err"] = float(rel)
    if rel > 1e-2:
        raise ValueError(f"{case.name}: k != (uu+vv+ww)/2 (rel err {rel:.2e}); "
                         "check vel_fluc column order")
    # 2. dU/dy matches a finite-difference derivative of U (away from the centreline).
    dU_fd = np.gradient(case.U, case.y_plus)
    inner = case.y_plus < 0.8 * case.y_plus.max()
    rel = np.median(np.abs(dU_fd[inner] - case.dUdy[inner]) / np.maximum(np.abs(case.dUdy[inner]), 1e-12))
    d["dUdy_fd_median_rel_err"] = float(rel)
    if rel > 5e-2:
        raise ValueError(f"{case.name}: dU/dy column disagrees with d(U)/dy+ (median rel {rel:.2e})")
    # 3. Budget balance: identify the sign with which dissipation enters.
    for c, b in case.budgets.items():
        s = b["production"] + b["turb_transport"] + b["visc_transport"] + b["pressure_strain"] + b["pressure_transport"]
        scale = max(np.max(np.abs(b[t])) for t in ("production", "turb_transport", "visc_transport",
                                                     "pressure_strain", "visc_dissipation")) + 1e-30
        r_minus = np.max(np.abs(s - b["visc_dissipation"] - b["balance"])) / scale
        r_plus = np.max(np.abs(s + b["visc_dissipation"] - b["balance"])) / scale
        # NB: the published balance column is DERIVED from the other columns, so this is a
        # column-order check (identity, ~1e-16), not a measure of statistical convergence.
        d[f"balance_column_identity_{c}"] = float(min(r_minus, r_plus))
        d[f"dissipation_sign_{c}"] = "subtracted" if r_minus <= r_plus else "added"
        # Statistical convergence indicator: magnitude of the budget residual itself.
        eps_k = np.abs(case.budgets["k"]["visc_dissipation"])
        res = np.abs(b["balance"][1:]) / eps_k[1:]
        d[f"budget_residual_over_eps_median_{c}"] = float(np.median(res))
        d[f"budget_residual_over_eps_max_{c}"] = float(np.max(res))
        if min(r_minus, r_plus) > tol * 10:
            raise ValueError(f"{case.name}: RSTE_{c} columns do not reproduce the balance "
                             f"(rel {min(r_minus, r_plus):.2e}); check column order")
    # 4. epsilon from the k budget vs half the normal-stress dissipations.
    eps_n = 0.5 * sum(np.abs(case.budgets[c]["visc_dissipation"]) for c in ("uu", "vv", "ww"))
    rel = np.max(np.abs(eps_n - case.eps)) / np.max(case.eps)
    d["eps_consistency_rel_err"] = float(rel)
    # 5. DNS trace of pressure strain. Found to be ~1e-15 relative in all cases (exact discrete
    #    continuity of the spectral code); there is no statistical noise floor for the trace.
    tr = sum(case.budgets[c]["pressure_strain"] for c in ("uu", "vv", "ww"))
    norm = np.sqrt(sum(case.budgets[c]["pressure_strain"] ** 2 for c in ("uu", "vv", "ww"))
                   + 2 * case.budgets["uv"]["pressure_strain"] ** 2)
    d["dns_trace_rel_max"] = float(np.max(np.abs(tr[1:]) / np.maximum(norm[1:], 1e-30)))
    # 6. Budget normalisation: the uu equation must be for R_11 (not R_11/2),
    #    i.e. production_uu = -2 * uv * dU/dy. Our classical closures assume this.
    p_expected = -2.0 * case.uv * case.dUdy
    m = np.abs(p_expected) > 0.05 * np.max(np.abs(p_expected))
    ratio = np.median(case.budgets["uu"]["production"][m] / p_expected[m])
    d["uu_production_ratio"] = float(ratio)
    if abs(ratio - 1.0) > 0.05:
        raise ValueError(f"{case.name}: production_uu / (-2 uv dU/dy) = {ratio:.3f}; "
                         "budget normalisation differs from R_ij equations")
    # 7. Pressure transport in the uu equation should vanish.
    d["uu_pressure_transport_max"] = float(np.max(np.abs(case.budgets["uu"]["pressure_transport"])))
    return d
