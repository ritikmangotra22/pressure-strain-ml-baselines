"""Protocol v1 training, tuning, refit and evaluation pipeline for M1-M4 and the ablations.

Everything here follows configs/paper2/protocol_v1.json. Implementation details that the protocol
did not spell out are listed in IMPLEMENTATION_NOTES and were fixed before any training run.

Leakage controls (tested in tests/paper2/test_pipeline.py):
  * every scaler, the target scale and the model inputs are built from the training cases of the
    current (inner or outer) split only;
  * the held-out outer case is never passed to tune(); its targets are used only in evaluate();
  * the selection criterion is the inner leave-one-out E_Pi; test data never enter selection.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import models, tensors as T
from ps.metrics import all_metrics

IMPLEMENTATION_NOTES = {
    "target_scale": "targets divided by s = sqrt(sum w ||Pi/eps||_F^2 / 4) of the training cases (Paper 1/R0 convention); predictions multiplied back",
    "loss": "sum_n w_n ||f_n - Pi_n/s||_F^2 over the full 3x3 tensor, w normalised to sum 1; A-soft adds lambda * sum_n w_n (tr f_n)^2",
    "early_stopping_loss": "data term on the inner validation case, float32, patience 200, max 5000 epochs",
    "selection_metric": "inner-validation E_Pi of the full tensor, float64 with weights cast",
    "primary_metric": "E_Pi of the full symmetric tensor (13/23 components included: DNS value 0); Paper 1's 4-component metrics also stored",
    "non_finite_policy": "a fit whose loss becomes non-finite stops; during tuning the trial is recorded as FAIL and scored +inf for every model alike; a final refit that fails is retried once with identical settings and then recorded as failed",
    "A_aug": "8 random rotations of every training sample (inputs and targets rotated), original frame not included; scalers fitted on the augmented inputs; rotation draws seeded by the training seed",
    "A_soft": "M3 without dev(.); isotropic-limit subtraction kept",
}

MAX_EPOCHS, PATIENCE = 5000, 200


@dataclass
class Split:
    """Float64 tensors of one set of DNS points."""
    b: torch.Tensor
    G: torch.Tensor
    re: torch.Tensor
    P: torch.Tensor
    w: np.ndarray
    df: pd.DataFrame


def prepare(df, scheme="durbin", c_t=6.0) -> Split:
    b, G = T.channel_tensors(df, scheme, c_t)
    return Split(b, G, torch.tensor(df.Re_t.to_numpy(float), dtype=T.DT), T.target_tensor(df),
                 df.w.to_numpy(float), df.reset_index(drop=True))


def concat(splits):
    df = pd.concat([s.df for s in splits], ignore_index=True)
    df["w"] = df.groupby("case")["w"].transform(lambda x: x / x.sum())
    return Split(torch.cat([s.b for s in splits]), torch.cat([s.G for s in splits]),
                 torch.cat([s.re for s in splits]), torch.cat([s.P for s in splits]), df.w.to_numpy(float), df)


def augment_rotations(sp: Split, n_rot: int, seed: int) -> Split:
    """A-aug: n_rot random rotations of every sample (inputs and targets)."""
    Qs = T.random_rotations(n_rot, 10_000 + seed)
    bs, Gs, Ps = [], [], []
    for Q in Qs:
        Qn = Q.expand(len(sp.b), 3, 3)
        bs.append(T.rotate(Qn, sp.b)); Gs.append(T.rotate(Qn, sp.G)); Ps.append(T.rotate(Qn, sp.P))
    df = pd.concat([sp.df] * n_rot, ignore_index=True)
    w = np.tile(sp.w, n_rot) / n_rot
    return Split(torch.cat(bs), torch.cat(Gs), sp.re.repeat(n_rot), torch.cat(Ps), w, df)


def target_scale(sp: Split) -> float:
    w = sp.w / sp.w.sum()
    return float(np.sqrt((w * (sp.P.numpy() ** 2).sum(axis=(1, 2))).sum() / 4.0)) + 1e-12


def e_pi(P_hat: np.ndarray, P: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float(np.sqrt((w * ((P_hat - P) ** 2).sum(axis=(1, 2))).sum() / (w * (P ** 2).sum(axis=(1, 2))).sum()))


class Fitted:
    """A trained model plus its target scale; predictions always in float64."""

    def __init__(self, model, s):
        self.model, self.s = model, s

    def m64(self):
        import copy
        return copy.deepcopy(self.model).double().eval()

    def predict(self, sp: Split, model64=None) -> np.ndarray:
        m = model64 or self.m64()
        with torch.no_grad():
            return m(sp.b, sp.G, sp.re).numpy() * self.s


def fit(name, train: Split, params: dict, seed: int, val: Split | None = None, epochs: int | None = None,
        inv_set="P1", lam_tr=0.0, history_every=10):
    """Full-batch AdamW in float32. Returns (Fitted, info)."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    arch = {k: params[k] for k in ("n_layers", "width", "activation")}
    model = models.build_any(name, train.b, train.G, train.re, seed=seed, inv_set=inv_set,
                             init="float32_cast", dtype=torch.float32, **arch)
    s = target_scale(train)
    f32 = lambda x: x.float()
    run = _runner(model)
    xt, Pt = run.inputs(train), f32(train.P / s)
    wt = torch.tensor(train.w / train.w.sum(), dtype=torch.float32)
    if val is not None:
        xv, Pv = run.inputs(val), f32(val.P / s)
        wv = torch.tensor(val.w / val.w.sum(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    data_loss = lambda Ph, Pr, ww: (ww * ((Ph - Pr) ** 2).sum(dim=(1, 2))).sum()
    n_ep = epochs or MAX_EPOCHS
    best, best_ep, best_state, since, hist, status = math.inf, 0, None, 0, [], "ok"
    t0 = time.time()
    for ep in range(1, n_ep + 1):
        opt.zero_grad()
        out = run(xt)
        loss = data_loss(out, Pt, wt)
        if lam_tr:
            loss = loss + lam_tr * (wt * T.tr(out) ** 2).sum()
        if not torch.isfinite(loss):
            status = "non_finite"
            break
        loss.backward()
        opt.step()
        vl = None
        if val is not None:
            with torch.no_grad():
                vl = float(data_loss(run(xv), Pv, wv))
            if vl < best - 1e-9:
                best, best_ep, since = vl, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                since += 1
        if ep % history_every == 0 or ep == 1:
            hist.append({"epoch": ep, "train_loss": float(loss.detach()), "val_loss": vl})
        if val is not None and since >= PATIENCE:
            break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    info = {"epochs_run": ep, "best_epoch": best_ep if val is not None else n_ep, "status": status,
            "fit_seconds": round(time.time() - t0, 3), "n_params": sum(p.numel() for p in model.parameters())}
    return Fitted(model, s), info, hist


class _runner:
    """Feeds a float32 model its training inputs. Basis models get invariants and basis tensors
    computed once in float64 and cast (protocol v1 precision policy); Cartesian models get the
    float64 inputs cast to float32 (an exact copy apart from rounding)."""

    def __init__(self, model):
        self.model = model
        self.cached = hasattr(model, "prepare")

    def inputs(self, sp):
        if self.cached:
            return self.model.prepare(sp.b, sp.G, sp.re, torch.float32)
        return (sp.b.float(), sp.G.float(), sp.re.float())

    def __call__(self, x):
        return self.model.forward_from(*x) if self.cached else self.model(*x)


def suggest(trial):
    """Protocol v1 search space (identical for every model)."""
    return {"n_layers": trial.suggest_int("n_layers", 2, 5),
            "width": trial.suggest_categorical("width", [32, 64, 128]),
            "activation": trial.suggest_categorical("activation", ["tanh", "silu"]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}


def tune(name, train_cases: dict, sampler_seed: int, n_trials=100, inv_set="P1", log=print):
    """Inner leave-one-case-out tuning on the given TRAINING cases only. Returns (params, records)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names = list(train_cases)
    records = []
    t_start = time.time()

    def objective(trial):
        params = suggest(trial)
        errs, eps = [], []
        for step, vc in enumerate(names):
            tr = concat([train_cases[c] for c in names if c != vc])
            va = train_cases[vc]
            fitted, info, _ = fit(name, tr, params, seed=0, val=va, inv_set=inv_set)
            if info["status"] != "ok":
                trial.set_user_attr("failed", info["status"])
                return math.inf
            errs.append(e_pi(fitted.predict(va), va.P.numpy(), va.w))
            eps.append(info["best_epoch"])
            trial.report(float(np.mean(errs)), step)
            if trial.should_prune():
                trial.set_user_attr("epochs", eps)
                raise optuna.TrialPruned()
        trial.set_user_attr("epochs", eps)
        trial.set_user_attr("inner_errors", errs)
        return float(np.mean(errs))

    def cb(study, t):
        rec = {"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
               "epochs": t.user_attrs.get("epochs"), "inner_errors": t.user_attrs.get("inner_errors"),
               "failed": t.user_attrs.get("failed"), "elapsed_s": round(time.time() - t_start, 1)}
        records.append(rec)
        log(f"  trial {t.number + 1}/{n_trials} {t.state.name} value={t.value if t.value is not None else float('nan'):.4f} "
            f"elapsed={rec['elapsed_s']:.0f}s")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials, callbacks=[cb])
    best = study.best_trial
    params = dict(best.params)
    params["epochs"] = int(np.median(best.user_attrs["epochs"]))
    params["inner_val_E_Pi"] = float(best.value)
    params["tune_seconds"] = round(time.time() - t_start, 1)
    return params, records


# ---------------------------------------------------------------------------------------------
# Evaluation and structural diagnostics (float64)
# ---------------------------------------------------------------------------------------------
TEST_ROTATIONS = None


def test_transforms():
    global TEST_ROTATIONS
    if TEST_ROTATIONS is None:
        Q = {f"rand{i:03d}": q for i, q in enumerate(T.random_rotations(200, 12345))}
        for d in (15, 45, 90):
            Q[f"rot_z_{d}"] = T.rot_about_z(d)
        Q["reflect_y"] = T.REFLECT_Y
        TEST_ROTATIONS = Q
    return TEST_ROTATIONS


def evaluate(fitted: Fitted, te: Split, full_diagnostics=True) -> tuple[dict, np.ndarray]:
    m64 = fitted.m64()
    P_hat = fitted.predict(te, m64)
    P = te.P.numpy()
    out = {"E_Pi": e_pi(P_hat, P, te.w)}
    four = lambda A: np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1]], 1)
    p1 = all_metrics(four(P_hat), four(P), te.w, te.df.region.to_numpy())
    out.update({f"p1_{k}": v for k, v in p1.items()})
    for reg in ("viscous", "buffer", "log", "outer"):
        m = (te.df.region == reg).to_numpy()
        out[f"E_Pi_{reg}"] = e_pi(P_hat[m], P[m], te.w[m]) if m.sum() >= 2 else float("nan")
    yd = te.df.y_delta.to_numpy()
    m = yd >= 0.9
    out["E_Pi_centreline_ydelta_ge_0.9"] = e_pi(P_hat[m], P[m], te.w[m]) if m.sum() >= 2 else float("nan")
    out["offchannel_rel_max"] = float((np.sqrt(2 * (P_hat[:, 0, 2] ** 2 + P_hat[:, 1, 2] ** 2)) /
                                       np.maximum(np.linalg.norm(P_hat, axis=(1, 2)), 1e-300)).max())
    if full_diagnostics:
        out.update(structural(fitted, m64, te, P_hat))
    return out, P_hat


def structural(fitted: Fitted, m64, te: Split, P_hat: np.ndarray) -> dict:
    d = {}
    Ph = torch.tensor(P_hat / fitted.s, dtype=T.DT)
    nP = T.frob(Ph).clamp(min=1e-300)
    eq_max, eq_sum, eq_n, rot_errs = 0.0, 0.0, 0, []
    with torch.no_grad():
        for k, Q in test_transforms().items():
            Qn = Q.expand(len(te.b), 3, 3)
            Pq = m64(T.rotate(Qn, te.b), T.rotate(Qn, te.G), te.re)
            err = T.frob(Pq - T.rotate(Qn, Ph)) / nP
            eq_max, eq_sum, eq_n = max(eq_max, float(err.max())), eq_sum + float(err.sum()), eq_n + len(err)
            if k.startswith("rand") and int(k[4:]) < 20:   # T3: E_Pi on rotated test inputs (rotated back)
                back = T.rotate(Qn.transpose(1, 2), Pq).numpy() * fitted.s
                rot_errs.append(e_pi(back, te.P.numpy(), te.w))
            if k == "reflect_y":
                d["eq_reflect_y_max"] = float(err.max())
        d["eq_max"], d["eq_mean"] = eq_max, eq_sum / eq_n
        d["T3_E_Pi_rotated_mean"] = float(np.mean(rot_errs))
        d["trace_rel_max"] = float((T.tr(Ph).abs() / nP).max())
        d["trace_rel_mean"] = float(np.average((T.tr(Ph).abs() / nP).numpy(), weights=te.w))
        d["asym_max"] = float((Ph - Ph.transpose(1, 2)).abs().max())
        rt = torch.logspace(0, 4, 60, dtype=T.DT)
        z = torch.zeros((60, 3, 3), dtype=T.DT)
        d["iso_limit_max"] = float(T.frob(m64(z, z, rt)).max()) * fitted.s
        m32 = fitted.model.eval()
        run = _runner(m32)
        P32 = run(run.inputs(te)).double()
        d["f32_f64_pred_rel_max"] = float((T.frob(P32 - Ph) / nP).max())
    # D3 realizability tendency: signed error along the smallest-eigenvalue direction of b
    b = te.b.numpy()
    ev, evec = np.linalg.eigh(b)
    lam = ev[:, 0] + 1 / 3                      # smallest eigenvalue of R / (2k)
    e = evec[:, :, 0]
    sel = lam < 0.05
    sn = np.einsum("ni,nij,nj->n", e, P_hat - te.P.numpy(), e)
    d["D3_n_points"] = int(sel.sum())
    d["D3_frac_negative"] = float((sn[sel] < 0).mean()) if sel.any() else float("nan")
    d["D3_wmean"] = float(np.average(sn[sel], weights=te.w[sel])) if sel.any() else float("nan")
    return d


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
