"""Design decision 1: should tr(b W*^2) be added to the invariant set of M2/M4?

Pre-training verification only (no model is trained). Parts:

  A  definitions: channel-flow closed forms of every invariant, checked on all DNS states, and the
     general identity tr(b W^2) = (1/4) w.b.w (w = vorticity vector of W) on random 3-D states
  B  dimensional consistency: invariants unchanged when the dimensional inputs (k, eps, dU/dy, nu)
     are rescaled as a change of velocity and length units would rescale them
  C  transformation properties: invariance under random rotations and reflections, on DNS states
     and on random general 3-D states
  D  uniqueness: exact reconstruction of (b11, b22, b12) from the four S*/W*-dependent invariants
     on every DNS state; twin search on states placed exactly on the E4 degenerate set
  E  conditioning and redundancy on the DNS states: correlations, condition numbers, and the size
     of each invariant near the wall under both time scales

    python scripts/paper2/check4_invariant_set.py --config configs/paper2/check4.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, diagnostics as D  # noqa: E402
from ps2 import tensors as T  # noqa: E402

ALL = ["tr_b2", "tr_b3", "tr_S2", "tr_bS", "tr_bSW", "tr_bW2"]


def inv(df, scheme, c_t, names=ALL):
    b, G = T.channel_tensors(df, scheme, c_t)
    S, W = T.sym_skew(G)
    return T.invariants(b, S, W, names).numpy(), G[:, 0, 1].numpy()


def part_a(dfs, c_t):
    out = {}
    for db, df in dfs.items():
        I, g = inv(df, "durbin", c_t)
        b11, b22, b33, b12 = (df[c].to_numpy() for c in ("b11", "b22", "b33", "b12"))
        closed = {"tr_S2": g ** 2 / 2, "tr_bS": g * b12, "tr_bSW": g ** 2 / 4 * (b22 - b11),
                  "tr_bW2": g ** 2 / 4 * b33}
        out[db] = {k: float(np.abs(I[:, ALL.index(k)] - v).max()) for k, v in closed.items()}
    # general 3-D identity tr(bW^2) = (1/4) w.b.w, W_ij = -(1/2) eps_ijk w_k
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(1000):
        A = rng.normal(size=(3, 3)); b = 0.5 * (A + A.T); b -= np.trace(b) / 3 * np.eye(3)
        w = rng.normal(size=3)
        W = -0.5 * np.array([[0, w[2], -w[1]], [-w[2], 0, w[0]], [w[1], -w[0], 0]])
        err = max(err, abs(np.trace(b @ W @ W) - 0.25 * w @ b @ w))
    out["general_identity_max_abs_err"] = err
    return out


def part_b(df, c_t, rng):
    """Change of units: velocity scale U, length scale L (nu = 1 in wall units originally)."""
    errs = []
    base = inv(df, "durbin", c_t)[0]
    for _ in range(5):
        U, L = 10 ** rng.uniform(-2, 2), 10 ** rng.uniform(-3, 1)
        k = df.k.to_numpy() * U ** 2
        eps = df.eps.to_numpy() * U ** 3 / L
        dU = df.dUdy.to_numpy() * U / L
        nu = 1.0 * U * L
        d2 = df.copy()
        d2["eta"] = k * np.abs(dU) / eps
        d2["Re_t"] = k ** 2 / (nu * eps)
        d2["dUdy"] = dU
        I2 = inv(d2, "durbin", c_t)[0]
        errs.append(float((np.abs(I2 - base) / np.maximum(np.abs(base), 1e-12)).max()))
    return {"max_rel_change_under_unit_rescaling": max(errs)}


def part_c(df, c_t, rng):
    b, G = T.channel_tensors(df, "durbin", c_t)
    S, W = T.sym_skew(G)
    I0 = T.invariants(b, S, W, ALL)
    worst = 0.0
    Qs = list(T.random_rotations(100, 11)) + [T.REFLECT_Y, -T.random_rotations(1, 5)[0]]
    for Q in Qs:
        Qn = Q.expand(len(b), 3, 3)
        Sq, Wq = T.sym_skew(T.rotate(Qn, G))
        Iq = T.invariants(T.rotate(Qn, b), Sq, Wq, ALL)
        worst = max(worst, float(((Iq - I0).abs() / I0.abs().clamp(min=1e-12)).max()))
    # general 3-D states
    A = torch.tensor(rng.normal(size=(500, 3, 3)), dtype=T.DT)
    bg = T.dev(0.5 * (A + A.transpose(1, 2))) * 0.2
    Gg = torch.tensor(rng.normal(size=(500, 3, 3)), dtype=T.DT)
    Sg, Wg = T.sym_skew(Gg)
    Ig = T.invariants(bg, Sg, Wg, ALL)
    worst_g = 0.0
    for Q in Qs:
        Qn = Q.expand(500, 3, 3)
        Sq, Wq = T.sym_skew(T.rotate(Qn, Gg))
        Iq = T.invariants(T.rotate(Qn, bg), Sq, Wq, ALL)
        worst_g = max(worst_g, float(((Iq - Ig).abs() / Ig.abs().clamp(min=1e-12)).max()))
    return {"dns_max_rel_change": worst, "general_states_max_rel_change": worst_g, "n_transforms": len(Qs)}


def reconstruct(I, g):
    """(b11, b22, b12) from tr_bS, tr_bSW, tr_bW2 for g != 0 (closed forms of Part A)."""
    b12 = I[:, ALL.index("tr_bS")] / g
    d = 4 * I[:, ALL.index("tr_bSW")] / g ** 2          # b22 - b11
    b33 = 4 * I[:, ALL.index("tr_bW2")] / g ** 2
    s = -b33                                             # b11 + b22
    return (s - d) / 2, (s + d) / 2, b12


def part_d(dfs, c_t, cfg):
    out = {}
    for db, df in dfs.items():
        for sc in ("durbin", "k/eps"):
            I, g = inv(df, sc, c_t)
            m = g > 0
            b11, b22, b12 = reconstruct(I[m], g[m])
            err = np.max(np.abs(np.c_[b11, b22, b12] - df.loc[m, ["b11", "b22", "b12"]].to_numpy()), axis=1)
            yp = df.loc[m, "y_plus"].to_numpy()
            out[f"{db}_{sc}"] = {"n_points_g_gt_0": int(m.sum()), "n_points_g_eq_0": int((~m).sum()),
                                 "max_abs_reconstruction_error": float(err.max()),
                                 "max_err_y_plus_lt_1": float(err[yp < 1].max()) if (yp < 1).any() else None,
                                 "min_g": float(g[m].min()), "g_at_first_point_median": float(
                                     np.median([g[(df.case == c).to_numpy()][0] for c in df.case.unique()]))}
    # twin search on states exactly on the E4 degenerate set: d^2 + 4 b12^2 = s^2
    rng = np.random.default_rng(cfg["twin_seed"])
    states = []
    while len(states) < cfg["n_twin_states"]:
        d, b12 = rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.0)
        s = np.sqrt(d ** 2 + 4 * b12 ** 2) * rng.choice([-1, 1])
        x = np.array([(s - d) / 2, (s + d) / 2, b12, rng.uniform(0.5, 20)])
        if D.realizable(x) and D.realizable(np.array([(-s - d) / 2, (-s + d) / 2, b12, x[3]])):
            states.append(x)
    res = {}
    for nm in ("E4", "E4+bW2"):
        r2 = np.random.default_rng(cfg["twin_seed"] + 1)
        found = sum(D.find_twin(x, T.INVARIANT_SETS[nm], r2, n_starts=cfg["twin_starts"]) is not None for x in states)
        res[nm] = found
        print(f"D: degenerate-set states with a twin, {nm}: {found}/{len(states)}", flush=True)
    out["twins_on_degenerate_set"] = {"n_states": len(states), **res}
    return out


def part_e(dfs, c_t):
    out = {}
    for db, df in dfs.items():
        for sc in ("durbin", "k/eps"):
            I, g = inv(df, sc, c_t)
            X = np.c_[I, df.Re_t.to_numpy()]
            names = ALL + ["Re_t"]
            Z = (X - X.mean(0)) / X.std(0)
            C = np.corrcoef(Z, rowvar=False)
            def cond(cols):
                return float(np.linalg.cond(Z[:, cols]))
            idx5 = [0, 1, 2, 3, 4, 6]
            idx6 = list(range(7))
            vif = {}
            for j in range(len(names)):
                others = [i for i in idx6 if i != j]
                beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z[:, others]], Z[:, j], rcond=None)
                r2 = 1 - np.var(Z[:, j] - np.c_[np.ones(len(Z)), Z[:, others]] @ beta) / np.var(Z[:, j])
                vif[names[j]] = float(1 / max(1 - r2, 1e-16))
            v = (df.region == "viscous").to_numpy()
            out[f"{db}_{sc}"] = {
                "corr_with_tr_bW2": {n: round(float(C[ALL.index("tr_bW2"), i]), 4) for i, n in enumerate(names)},
                "cond_E4_plus_Re_t": cond(idx5), "cond_E4bW2_plus_Re_t": cond(idx6),
                "VIF_all_inputs_E4bW2": {k: round(v_, 2) for k, v_ in vif.items()},
                "range_tr_bW2": [float(I[:, 5].min()), float(I[:, 5].max())],
                "median_abs_viscous": {n: float(np.median(np.abs(I[v, i]))) for i, n in enumerate(ALL)},
                "median_abs_all": {n: float(np.median(np.abs(I[:, i]))) for i, n in enumerate(ALL)}}
    return out


def main(cfg):
    t0 = time.time()
    rng = np.random.default_rng(cfg["seed"])
    dfs = {"LM": data.load("LM"), "HJ": data.load("HJ")}
    res = {"A_definitions": part_a(dfs, cfg["C_T"])}
    print("A done", flush=True)
    res["B_dimensions"] = {db: part_b(df, cfg["C_T"], rng) for db, df in dfs.items()}
    res["C_transformations"] = {db: part_c(df, cfg["C_T"], rng) for db, df in dfs.items()}
    print("B, C done", flush=True)
    res["D_uniqueness"] = part_d(dfs, cfg["C_T"], cfg)
    res["E_conditioning"] = part_e(dfs, cfg["C_T"])
    res["runtime_s"] = round(time.time() - t0, 1)
    res["config"] = cfg
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "check4_invariant_set.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:9000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(json.loads(Path(ap.parse_args().config).read_text()))
