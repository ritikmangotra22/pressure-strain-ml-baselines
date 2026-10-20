"""Section H check 3 (reproduction R0): Paper 1's MLP configuration run through Paper 2 code.

Registered criterion (Paper 2 doc, Section H item 3 / ablation R0): "M1 with Paper 1's inputs
reproduces Paper 1's MLP fold errors within about 0.02." Operationalised before running as:
for every E1 fold, |mean_seeds E_Pi(R0) - mean_seeds E_Pi(Paper 1 MLP)| <= 0.02.

Stages (each is a separate invocation so the long one can run in the background):

  env       Paper 1's own code (ps.models.MLPModel), recorded hyperparameters and seeds.
            Separates environment/data effects from code effects. Expected: ~1e-16.
  p2_f32    Paper 2 tensor-form trainer, Paper 1 inputs, recorded hyperparameters, float32.
  p2_f64    Same in float64 (the precision Paper 2 will use for M1-M4), layers created in float64.
  p2_f64c   float64, layers created in float32 and cast (same initial weights as Paper 1).
  retune    Paper 2 trainer, an independent 100-trial nested re-tuning per fold (Optuna TPE,
            sampler seed 2026, median pruner), then 10 seeds. Run per fold. Default precision
            float32 (Paper 1's), so that the comparison isolates the code path; float64 tuning
            took ~70 s per trial here (about 5 h in total) and the precision effect is measured
            separately by stage p2_f64.

Data: the released Paper 1 Lee & Moser processed table (raw files unavailable here).

    python scripts/paper2/check3_r0.py --stage env
    python scripts/paper2/check3_r0.py --stage p2_f32
    python scripts/paper2/check3_r0.py --stage p2_f64
    python scripts/paper2/check3_r0.py --stage retune --fold LM_Channel_0180   # one per fold
    python scripts/paper2/check3_r0.py --stage summarise
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps.features import TARGETS  # noqa: E402
from ps.metrics import e_pi  # noqa: E402
from ps.models import make_model  # noqa: E402
from ps2 import data  # noqa: E402
from ps2.train import F1, R0Model, renorm, suggest  # noqa: E402

OUT = ROOT / "results/paper2/check3"
P1 = ROOT / "results/e1_F1"
CASES = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"]
SEEDS = list(range(10))
TOL = 0.02


def tables():
    df = data.load("LM")
    return {c: df[df.case == c].reset_index(drop=True) for c in CASES}


def recorded_params(held):
    p = dict(json.loads((P1 / "best_params.json").read_text())[f"mlp__{held}"])
    p.pop("fit_info", None)
    return p


def score(model, te):
    return e_pi(model.predict(te), te[TARGETS].to_numpy(), te["w_eval"].to_numpy())


def run_recorded(stage):
    tabs = tables()
    rec = pd.read_csv(P1 / "metrics_long.csv")
    rec = rec[rec.model == "mlp"]
    rows = []
    for held in CASES:
        p = recorded_params(held)
        tr = renorm(pd.concat([tabs[c] for c in CASES if c != held]))
        te = tabs[held]
        for s in SEEDS:
            t0 = time.time()
            if stage == "env":
                m = make_model("mlp", F1, s, p)
            else:
                m = R0Model(seed=s, dtype="float32" if stage == "p2_f32" else "float64",
                            init="float32_cast" if stage == "p2_f64c" else "native", **p)
            m.fit(tr)
            e = score(m, te)
            r = float(rec[(rec.held_out == held) & (rec.seed == s)].E_Pi.iloc[0])
            rows.append({"stage": stage, "held_out": held, "seed": s, "E_Pi": e, "E_Pi_paper1": r,
                         "abs_diff": abs(e - r), "fit_s": round(time.time() - t0, 2)})
            print(f"{stage} {held} seed {s}: {e:.6f} vs {r:.6f} (|d| = {abs(e - r):.2e})", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"r0_{stage}.csv", index=False)


def run_retune(held, n_trials=100, dtype="float32"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tabs = tables()
    train_cases = [c for c in CASES if c != held]
    t_start = time.time()

    def objective(trial):
        params = suggest(trial)
        errs, extras = [], []
        for step, vc in enumerate(train_cases):
            tr = renorm(pd.concat([tabs[c] for c in train_cases if c != vc]))
            va = tabs[vc]
            m = R0Model(seed=0, dtype=dtype, **params)
            extras.append(m.fit(tr, va))
            errs.append(score(m, va))
            trial.report(float(np.mean(errs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("extras", extras)
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda st, t: print(
        f"  [{held}] trial {t.number + 1}/{n_trials} {t.state.name} "
        f"value={t.value if t.value is not None else float('nan'):.4f} elapsed={time.time() - t_start:.0f}s", flush=True)])
    best = study.best_trial
    params = dict(best.params)
    params["epochs"] = int(np.median([e["epochs"] for e in best.user_attrs["extras"]]))
    tr = renorm(pd.concat([tabs[c] for c in train_cases]))
    rows = []
    for s in SEEDS:
        m = R0Model(seed=s, dtype=dtype, **params)
        m.fit(tr)
        rows.append({"stage": "retune", "dtype": dtype, "held_out": held, "seed": s, "E_Pi": score(m, tabs[held]),
                     "inner_val_E_Pi": best.value})
    pd.DataFrame(rows).to_csv(OUT / f"r0_retune_{held}.csv", index=False)
    (OUT / f"r0_retune_{held}_params.json").write_text(json.dumps(
        {"params": params, "dtype": dtype, "inner_val_E_Pi": best.value, "n_trials": len(study.trials),
         "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
         "tune_time_s": round(time.time() - t_start, 1)}, indent=2))
    print(pd.DataFrame(rows).E_Pi.describe(), flush=True)


def summarise():
    rec = pd.read_csv(P1 / "metrics_long.csv")
    p1 = rec[rec.model == "mlp"].groupby("held_out").E_Pi.agg(["mean", "std"])
    out, verdict = {}, {}
    frames = []
    for st in ("env", "p2_f32", "p2_f64", "p2_f64c"):
        f = OUT / f"r0_{st}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    rt = sorted(OUT.glob("r0_retune_LM_*.csv"))
    if rt:
        frames.append(pd.concat([pd.read_csv(f) for f in rt]))
    allr = pd.concat(frames)
    tab = allr.groupby(["stage", "held_out"]).E_Pi.agg(["mean", "std"]).unstack("stage")
    for st in allr.stage.unique():
        m = allr[allr.stage == st].groupby("held_out").E_Pi.mean()
        d = (m - p1["mean"].reindex(m.index)).abs()
        out[st] = {"folds": int(len(m)), "max_fold_mean_abs_diff": float(d.max()),
                   "fold_mean_abs_diff": d.round(5).to_dict()}
        if "abs_diff" in allr and st != "retune":
            out[st]["max_per_run_abs_diff"] = float(allr[allr.stage == st].abs_diff.max())
        verdict[st] = bool(len(m) == 5 and d.max() <= TOL) if len(m) == 5 else f"incomplete ({len(m)}/5 folds)"
        # Welch t-test and permutation test of the seed-mean difference, per fold (descriptive)
        from scipy import stats
        tests = {}
        rng = np.random.default_rng(0)
        for h, g in allr[allr.stage == st].groupby("held_out"):
            a = g.E_Pi.to_numpy()
            b = rec[(rec.model == "mlp") & (rec.held_out == h)].sort_values("seed").E_Pi.to_numpy()
            pool, obs = np.r_[a, b], a.mean() - b.mean()
            perm = np.array([(lambda x: x[:len(a)].mean() - x[len(a):].mean())(rng.permutation(pool)) for _ in range(20000)])
            tests[h] = {"diff": round(float(obs), 5), "se": round(float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))), 5),
                        "welch_p": round(float(stats.ttest_ind(a, b, equal_var=False).pvalue), 4),
                        "perm_p": round(float(np.mean(np.abs(perm) >= abs(obs) - 1e-15)), 4)}
        out[st]["seed_mean_tests"] = tests
    # selected configurations of the independent re-tuning vs Paper 1
    sel = {}
    p1_params = json.loads((P1 / "best_params.json").read_text())
    for f in sorted(OUT.glob("r0_retune_LM_*_params.json")):
        h = f.name.replace("r0_retune_", "").replace("_params.json", "")
        r = json.loads(f.read_text())
        q = {k: v for k, v in p1_params[f"mlp__{h}"].items() if k != "fit_info"}
        sel[h] = {"retune": r["params"], "paper1": q, "same_architecture": all(r["params"][k] == q[k] for k in ("n_layers", "width", "activation")),
                  "inner_val_retune": r["inner_val_E_Pi"], "tune_time_s": r["tune_time_s"], "n_pruned": r["n_pruned"]}
    res = {"criterion": f"|fold mean E_Pi - Paper 1 fold mean| <= {TOL} on all 5 folds",
           "paper1_mlp_fold_mean": p1["mean"].round(5).to_dict(), "paper1_mlp_fold_sd": p1["std"].round(5).to_dict(),
           "stages": out, "pass": verdict, "retune_selected_configurations": sel}
    (OUT / "check3_summary.json").write_text(json.dumps(res, indent=2))
    tab.to_csv(OUT / "check3_fold_means.csv")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["env", "p2_f32", "p2_f64", "p2_f64c", "retune", "summarise"])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "retune":
        run_retune(a.fold, a.n_trials, a.dtype)
    elif a.stage == "summarise":
        summarise()
    else:
        run_recorded(a.stage)
