"""Final audit, supplementary quantities computed from saved checkpoints and metrics (no training).

  1. A-soft: mean relative trace residual of every soft-penalty model on its own TRAINING cases
     (needed to apply the selection rule of main-tab Section E.7: smallest inner E_Pi among weights
     whose mean trace residual is <= 1e-3).
  2. Inference time per point and parameter count of every main-experiment model (float64,
     single thread, forward pass only, median of 20 repeats on the job's test cases).
  3. A-scale: viscous-sublayer E_Pi with k/eps versus the Durbin-bounded time scale (design item 12).

    python3 scripts/paper2/audit_supplementary.py --root <archive root> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT, OUT = Path(a.root), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
from ps2 import models, pipeline as PL, tensors as T  # noqa: E402
import run_protocol as RP  # noqa: E402

torch.set_num_threads(1)
RES = ROOT / "results/paper2"
S = RP.splits("durbin")
E1 = RP.E1


def load_model(d, s, name, inv_set, scheme_splits):
    c = torch.load(d / "checkpoints" / f"seed{s}.pt", weights_only=False)
    p = c["params"]
    any_case = next(iter(scheme_splits.values()))
    m = models.build_any(name, any_case.b, any_case.G, any_case.re, seed=int(s), inv_set=inv_set,
                         init="float32_cast", dtype=torch.float32, **{k: p[k] for k in ("n_layers", "width", "activation")})
    m.load_state_dict(c["state_dict"])
    return PL.Fitted(m, c["s"]), p


# ---- 1. A-soft trace on the training cases ---------------------------------------------------
rows = []
for d in sorted((RES / "ablations/A-soft").glob("*/*")):
    if not (d / "status.json").exists():
        continue
    lam = float(d.parts[-2].split("lam")[1])
    outer = d.parts[-1]
    train = PL.concat([S[c] for c in E1 if c != outer])
    w = train.w / train.w.sum()
    for s in range(10):
        f, _ = load_model(d, s, "M3_soft", "P1", S)
        H = f.predict(train)
        tr = np.abs(np.trace(H, axis1=1, axis2=2)) / np.linalg.norm(H.reshape(len(H), -1), axis=1)
        rows.append({"lambda": lam, "outer": outer, "seed": s, "train_trace_rel_mean": float(tr.mean()),
                     "train_trace_rel_wmean": float((w * tr).sum()), "train_trace_rel_max": float(tr.max())})
soft = pd.DataFrame(rows)
soft.to_csv(OUT / "asoft_training_trace.csv", index=False)
soft_sum = soft.groupby("lambda")[["train_trace_rel_mean", "train_trace_rel_max"]].agg(["min", "mean", "max"])
soft_sum.to_csv(OUT / "asoft_training_trace_summary.csv")

# ---- 2. inference time and parameter count ----------------------------------------------------
rows = []
for st in sorted((RES / "main").glob("*/run*/*/status.json")):
    d = st.parent
    m, run, outer = d.parts[-3], d.parts[-2], d.parts[-1]
    tests = [S[c] for c in S if c.startswith("HJ")] if outer == "E2" else [S[outer]]
    te = PL.concat(tests)
    f, p = load_model(d, 0, m, "P1", S)
    m64 = f.m64()
    with torch.no_grad():
        m64(te.b, te.G, te.re)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            m64(te.b, te.G, te.re)
            ts.append(time.perf_counter() - t0)
    met = pd.read_csv(d / "metrics.csv")
    rows.append({"model": m, "run": run, "outer": outer, "n_params": int(sum(x.numel() for x in f.model.parameters())),
                 "n_params_logged": int(met.n_params.iloc[0]), "n_layers": p["n_layers"], "width": p["width"],
                 "refit_epochs": p["epochs"], "refit_seconds_mean": float(met.drop_duplicates("seed").fit_seconds.mean()),
                 "infer_us_per_point_f64": 1e6 * float(np.median(ts)) / len(te.b), "n_test_points": len(te.b)})
cost = pd.DataFrame(rows)
cost.to_csv(OUT / "model_cost.csv", index=False)
cost_sum = cost.groupby("model")[["n_params", "refit_epochs", "refit_seconds_mean", "infer_us_per_point_f64"]].median()
cost_sum.to_csv(OUT / "model_cost_summary.csv")

# ---- 3. A-scale sublayer --------------------------------------------------------------------
rows = []
for d in sorted((RES / "ablations/A-scale").glob("*/*")):
    st = json.loads((d / "status.json").read_text())
    pm, _, outer = st["parent"].split("__")
    a_m = pd.read_csv(d / "metrics.csv")
    p_m = pd.read_csv(RES / "main" / pm / "run1" / outer / "metrics.csv")
    for col in ("E_Pi_viscous", "E_Pi_buffer", "E_Pi"):
        rows.append({"model": pm, "outer": outer, "metric": col, "k_over_eps": a_m[col].mean(), "durbin": p_m[col].mean(),
                     "diff": a_m[col].mean() - p_m[col].mean()})
asc = pd.DataFrame(rows)
asc.to_csv(OUT / "ascale_regions.csv", index=False)
summary = {"asoft_min_train_trace_rel_mean_any_lambda": float(soft.train_trace_rel_mean.min()),
           "asoft_any_lambda_meets_1e-3": bool((soft.groupby(["lambda", "outer"]).train_trace_rel_mean.mean() <= 1e-3).any()),
           "params_match_logged": bool((cost.n_params == cost.n_params_logged).all())}
(OUT / "supplementary_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(soft_sum.round(5).to_string())
print(cost_sum.round(3).to_string())
print(asc[asc.metric == "E_Pi_viscous"].round(4).to_string())
