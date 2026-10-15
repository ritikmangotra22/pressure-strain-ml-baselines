"""Experiment E1: nested leave-one-Re_tau-out evaluation (manuscript Sections 7.5, 9.1).

Usage:
    python scripts/run_e1.py --config configs/e1_full.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ps.features import FEATURE_SETS, TARGETS, build_table, check_no_leakage  # noqa: E402
from ps.io_lm import load_lm_case, validate_case  # noqa: E402
from ps.metrics import all_metrics, block_bootstrap_e_pi, e_pi  # noqa: E402
from ps.models import SPACES, STOCHASTIC, make_model  # noqa: E402

INTERP = {550, 1000, 2000}
EXTRAP = {180, 5200}


def renorm(df):
    """Re-normalise weights so each case sums to one (equal weight per Re_tau)."""
    df = df.copy()
    df["w"] = df.groupby("case")["w"].transform(lambda s: s / s.sum())
    if "uniform_weights" in df.attrs:
        pass
    return df


def tune(name, train_cases, tables, features, n_trials, seed):
    """Inner leave-one-case-out tuning. Returns (best_params, final_fit_extras)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = SPACES[name](trial)
        errs, extras = [], []
        for step, val_case in enumerate(train_cases):
            tr = renorm(pd.concat([tables[c] for c in train_cases if c != val_case]))
            va = tables[val_case]
            m = make_model(name, features, seed, params)
            info = m.fit(tr, va)
            errs.append(e_pi(m.predict(va), va[TARGETS].to_numpy(), va["w_eval"].to_numpy()))
            extras.append(info)
            # Median pruning on the running mean of inner-fold errors (inner data only).
            trial.report(float(np.mean(errs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("extras", extras)
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    t_start = time.time()

    def log_trial(study, trial):
        print(f"    [tune {name}] trial {trial.number + 1}/{n_trials} {trial.state.name}"
              f" value={trial.value if trial.value is not None else float('nan'):.4f}"
              f" elapsed={time.time() - t_start:.0f}s", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])
    n_pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    print(f"    [tune {name}] done: {len(study.trials)} trials, {n_pruned} pruned, best inner E_Pi={study.best_value:.4f}", flush=True)
    best = study.best_trial
    params = SPACES[name](optuna.trial.FixedTrial(best.params))
    extras = best.user_attrs["extras"]
    # Early-stopping lengths learned in the inner loop set the final refit length.
    if name == "lgbm":
        params["n_estimators"] = [int(np.median([e["n_estimators"][j] for e in extras])) for j in range(4)]
    if name == "mlp":
        params["epochs"] = int(np.median([e["epochs"] for e in extras]))
    return params, float(best.value)


def main(cfg_path, models=None, out_dir=None, folds=None):
    cfg = json.loads(Path(cfg_path).read_text())
    if folds:
        cfg["folds"] = folds.split(",")
    if models:
        cfg["models"] = models.split(",")
    if out_dir:
        cfg["out_dir"] = out_dir
    out = Path(cfg["out_dir"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    features = FEATURE_SETS[cfg.get("feature_set", "F1")]
    check_no_leakage(features)

    # ---- data ----
    tables, validation, checksums = {}, {}, {}
    for lab in cfg["labels"]:
        case = load_lm_case(Path(cfg["data_dir"]), lab)
        validation[case.name] = validate_case(case)
        checksums.update(case.checksums)
        t = build_table(case)
        # w_eval: grid-spacing weights used for ALL scoring (selection and test), never changed.
        # w: training weights; ablation A3 replaces them with uniform weights (training only).
        t["w_eval"] = t["w"]
        if cfg.get("uniform_weights"):
            t["w"] = 1.0 / len(t)
        tables[case.name] = t
    allt = pd.concat(tables.values())
    allt.to_csv(out / "processed_table.csv", index=False)
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2))
    if cfg.get("synthetic"):
        (out / "WARNING_SYNTHETIC_DATA.txt").write_text(
            "These results come from SYNTHETIC test data. They are not DNS results and must not be reported.\n")

    # ---- outer loop ----
    rows, best_params = [], {}
    names = list(tables)
    folds = cfg.get("folds") or names
    for held in [n for n in names if n in folds or n.split("_")[-1] in folds]:
        train_cases = [c for c in names if c != held]
        te = tables[held]
        re = int(te["re_tau"].iloc[0])
        y_te, w_te = te[TARGETS].to_numpy(), te["w_eval"].to_numpy()
        for name in cfg["models"]:
            t0 = time.time()
            if name in SPACES:
                n_trials = cfg.get("n_trials_per_model", {}).get(name, cfg["n_trials"])
                params, inner = tune(name, train_cases, tables, features, n_trials, seed=0)
            else:
                params, inner = {}, np.nan
            tune_time = time.time() - t0
            seeds = cfg["seeds"] if name in STOCHASTIC else [0]
            for s in seeds:
                t1 = time.time()
                m = make_model(name, features, s, params)
                info = m.fit(renorm(pd.concat([tables[c] for c in train_cases])))
                fit_time = time.time() - t1
                t2 = time.time()
                y_hat = m.predict(te)
                inf_time = (time.time() - t2) / len(te)
                met = all_metrics(y_hat, y_te, w_te, te["region"].to_numpy())
                lo, hi = block_bootstrap_e_pi(y_hat, y_te, w_te, n_boot=cfg.get("n_boot", 2000), seed=s)
                rows.append({"model": name, "held_out": held, "re_tau": re,
                             "regime": "interp" if re in INTERP else "extrap", "seed": s,
                             "inner_val_E_Pi": inner, "E_Pi_ci_lo": lo, "E_Pi_ci_hi": hi,
                             "tune_time_s": tune_time, "fit_time_s": fit_time,
                             "infer_time_s_per_pt": inf_time, **met})
                pred = te[["case", "y_plus", "region"] + TARGETS].copy()
                for j, c in enumerate(TARGETS):
                    pred[c + "_hat"] = y_hat[:, j]
                pred.to_csv(out / "predictions" / f"{name}__{held}__seed{s}.csv", index=False)
                if s == seeds[0]:
                    best_params[f"{name}__{held}"] = {**{k: (v if not isinstance(v, np.generic) else v.item())
                                                         for k, v in params.items()}, **({"fit_info": info} if info else {})}
            print(f"[{held}] {name}: E_Pi = {np.mean([r['E_Pi'] for r in rows if r['model']==name and r['held_out']==held]):.4f}"
                  f"  (tune {tune_time:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_long.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # ---- summary table (Table 8 layout) ----
    g = df.groupby(["model", "re_tau"])["E_Pi"].agg(["mean", "std"]).reset_index()
    tab = g.pivot(index="model", columns="re_tau", values="mean")
    sd = g.pivot(index="model", columns="re_tau", values="std").fillna(0)
    fmt = tab.copy().astype(object)
    for c in tab.columns:
        fmt[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(tab[c], sd[c])]
    reg = df.groupby(["model", "regime"])["E_Pi"].mean().unstack()
    for col in ("interp", "extrap"):   # a single-fold part may contain only one regime
        fmt[f"mean_{col}"] = reg[col].map(lambda v: f"{v:.4f}") if col in reg else "—"
    md = fmt.to_markdown()
    if cfg.get("synthetic"):
        md = "**SYNTHETIC TEST DATA — NOT DNS — DO NOT REPORT**\n\n" + md
    (out / "table8_E_Pi.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=None, help="comma-separated override of cfg['models']")
    ap.add_argument("--out", default=None, help="override of cfg['out_dir']")
    ap.add_argument("--folds", default=None, help="comma-separated held-out labels, e.g. 0180,5200")
    a = ap.parse_args()
    main(a.config, a.models, a.out, a.folds)
