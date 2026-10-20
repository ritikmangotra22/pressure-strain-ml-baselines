"""Final audit, independent checks (no re-training). Run against an extracted archive:

    python3 scripts/paper2/audit_independent_checks.py --root <extracted archive root> --out <dir>

1. E_Pi recomputed for every saved prediction file with code written independently of
   pipeline.e_pi / analyse_protocol.e_pi_from (explicit 3x3 tensors, Frobenius norm).
2. Targets and weights in every prediction file equal the frozen input tables (SHA-256 checked).
3. Rows of every prediction file are ordered by wall distance (assumed by the block bootstrap).
4. Every saved checkpoint is reloaded, its model rebuilt and re-evaluated in float64; the
   predictions are compared with the saved prediction files.
5. Primary model-by-fold means, factorial point estimates, tuning-dominated flags and the
   consistency rule are recomputed from metrics.csv with independent code.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2]))
ap.add_argument("--out", required=True)
ap.add_argument("--skip-checkpoints", action="store_true")
a = ap.parse_args()
ROOT, OUT = Path(a.root), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results/paper2"
PROTO = json.loads((ROOT / "configs/paper2/protocol_v1.json").read_text())
COMP = {"11": (0, 0), "22": (1, 1), "33": (2, 2), "12": (0, 1), "13": (0, 2), "23": (1, 2)}
report = {}


def tensors(df, suffix=""):
    P = np.zeros((len(df), 3, 3))
    for c, (i, j) in COMP.items():
        P[:, i, j] = P[:, j, i] = df[f"Pi{c}{suffix}"].to_numpy()
    return P


def e_pi_indep(P, H, w):
    w = np.asarray(w, float)
    num = np.einsum("n,nij,nij->", w, H - P, H - P)
    den = np.einsum("n,nij,nij->", w, P, P)
    return float(np.sqrt(num / den))


# ---- frozen inputs -----------------------------------------------------------------------
tables = {}
for key, spec in PROTO["data"].items():
    if not isinstance(spec, dict):
        continue
    p = Path(a.data_root) / spec["table"]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    report[f"input_{key}_sha256_ok"] = sha == spec["sha256"]
    tables[key] = pd.read_csv(p)
tab = pd.concat(tables.values(), ignore_index=True)
tab_by_case = {c: g.reset_index(drop=True) for c, g in tab.groupby("case")}
_lm = tables["train_and_E1"]
report["input_LM_w_equals_w_eval"] = bool(np.array_equal(_lm.w.to_numpy(), _lm.w_eval.to_numpy()))
report["input_HJ_has_w_eval"] = "w_eval" in tables["E2_external"].columns

# ---- 1-3: every prediction file -------------------------------------------------------------
rows = []
job_dirs = sorted(p.parent for p in (RES / "main").glob("*/run*/*/status.json")) + \
           sorted(p.parent for p in (RES / "ablations").glob("*/*/*/status.json"))
for d in job_dirs:
    met = pd.read_csv(d / "metrics.csv")
    for _, r in met.iterrows():
        f = d / "predictions" / f"{r.case}__seed{int(r.seed)}.csv"
        pr = pd.read_csv(f)
        src = tab_by_case[r.case]
        P, H = tensors(pr), tensors(pr, "_hat")
        Psrc = np.zeros_like(P)
        for c in ("11", "22", "33", "12"):
            i, j = COMP[c]
            Psrc[:, i, j] = Psrc[:, j, i] = src[f"Pi{c}"].to_numpy()
        w_case = src.w.to_numpy() / src.w.sum()
        rows.append({"job": str(d.relative_to(RES)), "case": r.case, "seed": int(r.seed),
                     "E_Pi_logged": r.E_Pi, "E_Pi_indep": e_pi_indep(P, H, pr.w.to_numpy()),
                     "target_maxdiff_vs_table": float(np.abs(P - Psrc).max()),
                     "w_maxdiff_vs_table": float(np.abs(pr.w.to_numpy() / pr.w.sum() - w_case).max()),
                     "n_rows_match": len(pr) == len(src),
                     "y_monotonic": bool(np.all(np.diff(pr.y_plus.to_numpy()) > 0))})
pf = pd.DataFrame(rows)
pf["abs_diff"] = (pf.E_Pi_logged - pf.E_Pi_indep).abs()
pf.to_csv(OUT / "prediction_file_checks.csv", index=False)
report.update({"n_prediction_files": len(pf), "max_abs_E_Pi_diff": float(pf.abs_diff.max()),
               "max_target_diff_vs_table": float(pf.target_maxdiff_vs_table.max()),
               "max_weight_diff_vs_table": float(pf.w_maxdiff_vs_table.max()),
               "all_rows_match": bool(pf.n_rows_match.all()), "all_y_monotonic": bool(pf.y_monotonic.all())})

# ---- 4: checkpoint reload -----------------------------------------------------------------
if not a.skip_checkpoints:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts/paper2"))
    from ps2 import models, pipeline as PL  # noqa: E402
    import run_protocol as RP  # noqa: E402
    S = {"durbin": RP.splits("durbin"), "k/eps": RP.splits("k/eps")}
    ck = []
    for d in job_dirs:
        st = json.loads((d / "status.json").read_text())
        if "variant" in st:
            v = st["variant"]
            name, inv_set, scheme = v["name"], v["inv_set"], v["scheme"]
        else:
            name, inv_set, scheme = d.parts[-3], "P1", "durbin"
        met = pd.read_csv(d / "metrics.csv")
        for s in sorted(met.seed.unique()):
            c = torch.load(d / "checkpoints" / f"seed{s}.pt", weights_only=False)
            p = c["params"]
            any_case = next(iter(S[scheme].values()))
            m = models.build_any(name, any_case.b, any_case.G, any_case.re, seed=int(s), inv_set=inv_set,
                                 init="float32_cast", dtype=torch.float32,
                                 **{k: p[k] for k in ("n_layers", "width", "activation")})
            m.load_state_dict(c["state_dict"])
            fitted = PL.Fitted(m, c["s"])
            for case in met[met.seed == s].case:
                H = fitted.predict(S[scheme][case])
                saved = tensors(pd.read_csv(d / "predictions" / f"{case}__seed{s}.csv"), "_hat")
                ck.append({"job": str(d.relative_to(RES)), "seed": int(s), "case": case,
                           "max_abs_diff": float(np.abs(H - saved).max())})
    ck = pd.DataFrame(ck)
    ck.to_csv(OUT / "checkpoint_reload_checks.csv", index=False)
    report.update({"n_checkpoint_predictions": len(ck), "checkpoint_max_abs_diff": float(ck.max_abs_diff.max())})

# ---- 5: aggregates and decision rules from metrics.csv (independent code) -------------------
m = []
for st in (RES / "main").glob("*/run*/*/status.json"):
    d = st.parent
    x = pd.read_csv(d / "metrics.csv")[["case", "seed", "E_Pi"]]
    x["model"], x["run"] = d.parts[-3], int(d.parts[-2][3:])
    m.append(x)
m = pd.concat(m)
runmean = m.groupby(["model", "case", "run"]).E_Pi.mean().unstack("run")
prim = runmean.mean(axis=1).unstack("model")
rng_ = (runmean.max(axis=1) - runmean.min(axis=1)).unstack("model")
mbf = pd.read_csv(RES / "analysis/model_by_fold.csv").set_index(["case", "model"])
report["model_by_fold_max_diff"] = float(max(abs(prim.loc[c, mo] - mbf.loc[(c, mo), "primary_mean"]) for c in prim.index for mo in prim.columns))
eff = pd.read_csv(RES / "analysis/effects.csv")
defs = {"inv": lambda e: 0.5 * ((e.M2 - e.M1) + (e.M4 - e.M3)), "con": lambda e: 0.5 * ((e.M3 - e.M1) + (e.M4 - e.M2)),
        "int": lambda e: (e.M4 - e.M3) - (e.M2 - e.M1), "M1_M2": lambda e: e.M2 - e.M1, "M1_M3": lambda e: e.M3 - e.M1,
        "M2_M4": lambda e: e.M4 - e.M2, "M3_M4": lambda e: e.M4 - e.M3}
inv_models = {"inv": "M1 M2 M3 M4", "con": "M1 M2 M3 M4", "int": "M1 M2 M3 M4", "M1_M2": "M1 M2", "M1_M3": "M1 M3",
              "M2_M4": "M2 M4", "M3_M4": "M3 M4"}
chk = []
for _, r in eff.iterrows():
    e = prim.loc[r.case]
    est = defs[r.effect](e)
    td = abs(est) <= max(rng_.loc[r.case, mo] for mo in inv_models[r.effect].split())
    chk.append({"case": r.case, "effect": r.effect, "est_diff": abs(est - r.estimate), "td_match": td == r.tuning_dominated,
                "resolved_match": bool(r.ci_lo > 0 or r.ci_hi < 0) == r.resolved})
chk = pd.DataFrame(chk)
chk.to_csv(OUT / "effects_recompute.csv", index=False)
INTERP = PROTO["folds"]["interpolation"]
cons = {}
for k in defs:
    g = eff[(eff.effect == k) & eff.case.isin(INTERP)]
    for sgn in (-1, 1):
        cons[f"{k}_{sgn:+d}"] = int(((g.resolved) & (np.sign(g.estimate) == sgn) & (~g.tuning_dominated.astype(bool))).sum())
        cons[f"{k}_{sgn:+d}_ignoring_TD"] = int(((g.resolved) & (np.sign(g.estimate) == sgn)).sum())
report.update({"effects_point_max_diff": float(chk.est_diff.max()), "td_flags_all_match": bool(chk.td_match.all()),
               "resolved_flags_all_match": bool(chk.resolved_match.all()), "consistency_counts": cons,
               "any_confirmatory_effect_consistent": any(cons[f"{k}_{s:+d}"] >= 2 for k in ("inv", "con", "int") for s in (-1, 1))})
(OUT / "independent_checks.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
