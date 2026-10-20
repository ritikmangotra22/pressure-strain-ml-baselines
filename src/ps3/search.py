"""Paper 3 search space, candidate training and the four inner-fold objectives.

DRAFT for the timing pilot; not yet frozen. Nothing here reads held-out test data: every function
takes TRAINING cases only and scores candidates by inner leave-one-Re_tau-out validation.

Changes relative to src/ps2 (Paper 2 code is imported, never modified):
  * dropout between hidden layers (proposal 6.3.1); the model is put in eval mode for every
    validation and prediction pass, so dropout acts only on training steps;
  * tensor-basis size {3, 8} and invariant set {four, E4, P1} as searched variables for M2/M4;
  * C_T in {3, 6, 10} as a searched variable (inputs rebuilt per candidate);
  * objectives: f_err (mean inner E_Pi), f_gen (max inner E_Pi), f_phys (Arm A or Arm B form),
    f_cost (matrix-multiply FLOPs per point, counted by torch.utils.flop_counter).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from ps2 import models as M2mod, pipeline as PL, tensors as T

INV = {"four": "four", "five": "E4", "six": "P1"}
MAX_EPOCHS, PATIENCE = PL.MAX_EPOCHS, PL.PATIENCE


# ------------------------------------------------------------------ search space
def sample_space(rng: np.random.Generator, arch: str | None = None) -> dict:
    """Uniform draw from the draft Paper 3 space (random search and the pilot)."""
    a = arch or rng.choice(["M1", "M2", "M3", "M4"])
    c = {"arch": str(a),
         "n_layers": int(rng.integers(2, 6)),
         "width": int(rng.choice([32, 64, 128])),
         "activation": str(rng.choice(["tanh", "silu"])),
         "lr": float(10 ** rng.uniform(np.log10(1e-4), np.log10(3e-3))),
         "weight_decay": float(10 ** rng.uniform(-6, -2)),
         "dropout": float(rng.uniform(0.0, 0.2)),
         "c_t": float(rng.choice([3.0, 6.0, 10.0]))}
    if c["arch"] in ("M2", "M4"):
        c["n_basis"] = int(rng.choice([3, 8]))
        c["inv_set"] = str(rng.choice(["four", "five", "six"]))
    return c


# ------------------------------------------------------------------ models
def _mlp(n_in, n_out, n_layers, width, activation, dropout):
    act = nn.Tanh if activation == "tanh" else nn.SiLU
    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, width, dtype=torch.float32), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = width
    layers.append(nn.Linear(d, n_out, dtype=torch.float32))
    return nn.Sequential(*layers)


def build(c: dict, train: PL.Split, seed: int) -> nn.Module:
    """Paper 2 architectures with the Paper 3 extensions, scalers fitted on `train` only."""
    torch.manual_seed(seed)
    kw = dict(n_layers=c["n_layers"], width=c["width"], activation=c["activation"])
    if c["arch"] in ("M1", "M3"):
        sc = M2mod.fit_scaler(M2mod.cartesian_inputs(train.b, train.G, train.re))
        m = (M2mod.M1 if c["arch"] == "M1" else M2mod.M3)(sc, init="float32_cast", **kw)
        torch.manual_seed(seed)
        m.net = _mlp(16, 6, dropout=c["dropout"], **kw)
        return m.to(torch.float32)
    inv = INV[c["inv_set"]]
    nb = c["n_basis"]
    base = M2mod.M4 if c["arch"] == "M4" else M2mod.M2
    full = T.basis_dev if c["arch"] == "M4" else T.basis_m2
    # M2 keeps its free identity coefficient: subset 3 -> identity + first three raw products
    n_out = nb if c["arch"] == "M4" else nb + 1
    basis_fn = (lambda b, S, W: full(b, S, W)[:n_out])
    cls = type(f"{c['arch']}_b{nb}_{inv}", (base,), {"inv_set": inv, "basis": staticmethod(basis_fn)})
    S, W = T.sym_skew(train.G)
    feats = torch.cat([T.invariants(train.b, S, W, T.INVARIANT_SETS[inv]), train.re[:, None]], 1)
    m = cls.__new__(cls)
    nn.Module.__init__(m)
    m.scaler = M2mod.fit_scaler(feats)
    torch.manual_seed(seed)
    m.net = _mlp(len(T.INVARIANT_SETS[inv]) + 1, n_out, dropout=c["dropout"], **kw)
    return m.to(torch.float32)


# ------------------------------------------------------------------ training
def fit(c: dict, train: PL.Split, seed: int, val: PL.Split | None = None, epochs: int | None = None):
    """Full-batch AdamW, float32, early stopping on `val` (Paper 2 settings)."""
    torch.set_num_threads(1)
    model = build(c, train, seed)
    s = PL.target_scale(train)
    run = PL._runner(model)
    xt, Pt = run.inputs(train), (train.P / s).float()
    wt = torch.tensor(train.w / train.w.sum(), dtype=torch.float32)
    if val is not None:
        xv, Pv = run.inputs(val), (val.P / s).float()
        wv = torch.tensor(val.w / val.w.sum(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    loss_fn = lambda Ph, Pr, ww: (ww * ((Ph - Pr) ** 2).sum(dim=(1, 2))).sum()
    best, best_ep, best_state, since, status = math.inf, 0, None, 0, "ok"
    t0 = time.time()
    n_ep = epochs or MAX_EPOCHS
    for ep in range(1, n_ep + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(run(xt), Pt, wt)
        if not torch.isfinite(loss):
            status = "non_finite"
            break
        loss.backward()
        opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(run(xv), Pv, wv))
            if not math.isfinite(vl):
                status = "non_finite"
                break
            if vl < best - 1e-9:
                best, best_ep, since = vl, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= PATIENCE:
                    break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return PL.Fitted(model, s), {"status": status, "epochs_run": ep, "best_epoch": best_ep if val is not None else n_ep,
                                 "fit_seconds": time.time() - t0,
                                 "n_params": sum(p.numel() for p in model.parameters())}


# ------------------------------------------------------------------ objectives
def _lumley_states(n=2000, seed=2026, re_range=(1.0, 1e4)):
    """Realizable diagonal anisotropy states sampled uniformly over the Lumley triangle
    (rejection sampling on eigenvalues), with log-uniform Re_t. S = W = 0."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        l1, l2 = rng.uniform(-1 / 3, 2 / 3, 2)
        l3 = -l1 - l2
        if -1 / 3 <= l3 <= 2 / 3 and abs(l1) + abs(l2) + abs(l3) > 1e-3:
            out.append((l1, l2, l3))
    lam = np.array(out)
    b = torch.zeros((n, 3, 3), dtype=T.DT)
    for i in range(3):
        b[:, i, i] = torch.tensor(lam[:, i])
    re = torch.tensor(10 ** rng.uniform(*np.log10(re_range), n), dtype=T.DT)
    return b, torch.zeros_like(b), re


LUMLEY = None


def return_to_isotropy(fitted: PL.Fitted) -> dict:
    """Pi:b with S = W = 0 must not be positive (slow part must not increase anisotropy)."""
    global LUMLEY
    if LUMLEY is None:
        LUMLEY = _lumley_states()
    b, G, re = LUMLEY
    with torch.no_grad():
        P = fitted.m64()(b, G, re) * fitted.s
    pb = (P * b).sum(dim=(1, 2)).numpy()
    return {"rti_viol_mean": float(np.maximum(pb, 0).mean()), "rti_viol_frac": float((pb > 0).mean())}


def coef_norm(fitted: PL.Fitted, sp: PL.Split) -> float:
    """Grid-weighted mean ||g||_2 of the basis coefficients, in Pi/eps units (basis models)."""
    m = fitted.m64()
    if not hasattr(m, "prepare"):
        return float("nan")
    feats, _ = m.prepare(sp.b, sp.G, sp.re, T.DT)
    with torch.no_grad():
        g = m.net(m.scaler(feats)).numpy() * fitted.s
    w = sp.w / sp.w.sum()
    return float((w * np.linalg.norm(g, axis=1)).sum())


def frame_and_trace(fitted: PL.Fitted, sp: PL.Split, n_rot=20) -> dict:
    """Arm B physics terms: mean relative trace and excess E_Pi under random frame rotations."""
    m = fitted.m64()
    with torch.no_grad():
        P = m(sp.b, sp.G, sp.re)
        tr_rel = (T.tr(P).abs() / T.frob(P).clamp(min=1e-300)).numpy()
        base = PL.e_pi(P.numpy() * fitted.s, sp.P.numpy(), sp.w)
        errs = []
        for Q in T.random_rotations(n_rot, 777):
            Qn = Q.expand(len(sp.b), 3, 3)
            Pq = m(T.rotate(Qn, sp.b), T.rotate(Qn, sp.G), sp.re)
            back = T.rotate(Qn.transpose(1, 2), Pq).numpy() * fitted.s
            errs.append(PL.e_pi(back, sp.P.numpy(), sp.w))
    return {"trace_rel_mean": float(np.average(tr_rel, weights=sp.w)),
            "rot_excess": float(np.mean(errs) - base)}


def flops_per_point(fitted: PL.Fitted, sp: PL.Split) -> int:
    """Matrix-multiply FLOPs for one closure evaluation, counted by torch FlopCounterMode
    (features, network and basis assembly; element-wise operations are not counted)."""
    m = fitted.m64()
    x = (sp.b[:1], sp.G[:1], sp.re[:1])
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(*x)
    return int(fc.get_total_flops())


# ------------------------------------------------------------------ one candidate
def evaluate_candidate(c: dict, train_cases: dict[str, PL.Split], seeds=(0, 1)) -> dict:
    """Inner leave-one-Re_tau-out over the given training cases, `seeds` per inner fold.
    Returns objectives (means over seeds), their seed spread and timing. Any non-finite fit makes
    the candidate infeasible (recorded, never dropped)."""
    t0 = time.time()
    names = list(train_cases)
    per_seed = []
    status, n_fits, fit_s, epochs = "ok", 0, 0.0, []
    sp_tr = {k: PL.prepare(train_cases[k].df, "durbin", c["c_t"]) for k in names}
    for s in seeds:
        errs, rti, cn, ft = [], [], [], []
        for vc in names:
            tr = PL.concat([sp_tr[k] for k in names if k != vc])
            va = sp_tr[vc]
            fitted, info = fit(c, tr, seed=s, val=va)
            n_fits += 1
            fit_s += info["fit_seconds"]
            epochs.append(info["best_epoch"])
            if info["status"] != "ok":
                status = info["status"]
                break
            errs.append(PL.e_pi(fitted.predict(va), va.P.numpy(), va.w))
            r = return_to_isotropy(fitted)
            rti.append(r["rti_viol_mean"])
            cn.append(coef_norm(fitted, tr))
            ft.append(frame_and_trace(fitted, va))
        if status != "ok":
            break
        per_seed.append({"f_err": float(np.mean(errs)), "f_gen": float(np.max(errs)),
                         "inner_errors": errs, "rti": float(np.mean(rti)), "coef_norm": float(np.mean(cn)),
                         "trace_rel_mean": float(np.mean([f["trace_rel_mean"] for f in ft])),
                         "rot_excess": float(np.mean([f["rot_excess"] for f in ft]))})
    out = {**c, "status": status, "n_fits": n_fits, "fit_seconds_total": round(fit_s, 2),
           "wall_seconds": round(time.time() - t0, 2), "median_best_epoch": float(np.median(epochs)) if epochs else None}
    if status != "ok":
        return out
    for k in ("f_err", "f_gen", "rti", "coef_norm", "trace_rel_mean", "rot_excess"):
        v = [p[k] for p in per_seed]
        out[k] = float(np.mean(v))
        out[k + "_seed_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    out["inner_errors_seed0"] = per_seed[0]["inner_errors"]
    any_case = next(iter(train_cases.values()))
    fitted, _ = fit(c, PL.prepare(any_case.df, "durbin", c["c_t"]), seed=0, epochs=1)
    out["f_cost_flops"] = flops_per_point(fitted, PL.prepare(any_case.df, "durbin", c["c_t"]))
    return out


# ------------------------------------------------------------------ final objective vector
CASES_LM = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"]
RTI_FLOOR = 1e-4   # used if the M0 reference has (near-)zero return-to-isotropy violation; CONFIRM


def objective_vector(r: dict, ref: dict, arm: str) -> dict:
    """Map a candidate result (evaluate_candidate) to the four minimised objectives.

    f_err  mean inner E_Pi (unchanged, Paper 2's tuning objective)
    f_gen  max over inner folds of E_Pi(fold) / E_Pi_ref(fold)   [decision of 20 Oct 2026:
           relative to the M0 reference, because the absolute max is the Re_tau = 180 fold in
           24/26 pilot configurations]
    f_phys arm A: 0.5 rti / max(rti_ref, floor) + 0.5 coef_norm / coef_norm_ref
           arm B: trace_rel_mean + rot_excess
    f_cost matrix-multiply FLOPs per point
    Infeasible (failed fit) -> None.
    """
    if r.get("status") != "ok":
        return None
    errs = r["inner_errors_seed0"]          # with one seed per candidate this is the candidate's error
    ref_e = [ref["fold_E_Pi_mean"][c] for c in CASES_LM]
    f_gen = float(max(e / q for e, q in zip(errs, ref_e)))
    if arm == "A":
        f_phys = 0.5 * r["rti"] / max(ref["rti_mean"], RTI_FLOOR) + 0.5 * r["coef_norm"] / ref["coef_norm_mean"]
    else:
        f_phys = r["trace_rel_mean"] + r["rot_excess"]
    return {"f_err": float(np.mean(errs)), "f_gen": f_gen, "f_phys": float(f_phys), "f_cost": float(r["f_cost_flops"])}
