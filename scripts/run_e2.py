"""Experiment E2: train on all five Lee & Moser cases, test on Hoyas & Jimenez (2008).

Tuning: leave-one-Re_tau-out over the five LM cases (no HJ data used for any choice).
Model selection for E2 (which ML models enter) uses E1 *inner-validation* errors only.

    python scripts/run_e2.py --config configs/e2.json --models ridge,lgbm
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps.features import FEATURE_SETS, TARGETS, build_table, check_no_leakage  # noqa: E402
from ps.io_hj import load_hj_case, validate_hj  # noqa: E402
from ps.io_lm import load_lm_case, validate_case  # noqa: E402
from ps.metrics import all_metrics, block_bootstrap_e_pi  # noqa: E402
from ps.models import SPACES, STOCHASTIC, make_model  # noqa: E402
from run_e1 import renorm, tune  # noqa: E402


def main(cfg_path, models, out_dir):
    cfg = json.loads(Path(cfg_path).read_text())
    models = models.split(",") if models else cfg["models"]
    out = Path(out_dir or cfg["out_dir"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    features = FEATURE_SETS[cfg.get("feature_set", "F1")]
    check_no_leakage(features)
    train = {}
    for lab in cfg["labels"]:
        c = load_lm_case(Path(cfg["data_dir"]), lab)
        validate_case(c)
        train[c.name] = build_table(c)
        # run_e1.tune scores inner folds with w_eval (introduced for ablation A3); in E2 it equals w.
        train[c.name]["w_eval"] = train[c.name]["w"]
    test, val = {}, {}
    for re in cfg["hj_cases"]:
        c = load_hj_case(Path(cfg["hj_dir"]), re)
        val[c.name] = validate_hj(c)
        test[c.name] = build_table(c)
    (out / "validation_hj.json").write_text(json.dumps(val, indent=2))
    pd.concat(test.values()).to_csv(out / "processed_table_hj.csv", index=False)
    names = list(train)
    rows, params_out = [], {}
    for name in models:
        t0 = time.time()
        if name in SPACES:
            n_trials = cfg.get("n_trials_per_model", {}).get(name, cfg["n_trials"])
            params, inner = tune(name, names, train, features, n_trials, seed=0)
        else:
            params, inner = {}, np.nan
        tune_time = time.time() - t0
        params_out[name] = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}
        for s in (cfg["seeds"] if name in STOCHASTIC else [0]):
            m = make_model(name, features, s, params)
            info = m.fit(renorm(pd.concat(train.values())))
            if "coef" in info:
                params_out[name] = info
            for case, te in test.items():
                y, w = te[TARGETS].to_numpy(), te["w"].to_numpy()
                yh = m.predict(te)
                met = all_metrics(yh, y, w, te["region"].to_numpy())
                lo, hi = block_bootstrap_e_pi(yh, y, w, n_boot=cfg.get("n_boot", 2000), seed=s)
                rows.append({"model": name, "test_case": case, "re_tau": int(te["re_tau"].iloc[0]), "seed": s,
                             "inner_val_E_Pi": inner, "tune_time_s": tune_time, "E_Pi_ci_lo": lo, "E_Pi_ci_hi": hi, **met})
                p = te[["case", "y_plus", "region"] + TARGETS].copy()
                for j, c in enumerate(TARGETS):
                    p[c + "_hat"] = yh[:, j]
                p.to_csv(out / "predictions" / f"{name}__{case}__seed{s}.csv", index=False)
        print(f"E2 {name}: " + ", ".join(f"{c}={np.mean([r['E_Pi'] for r in rows if r['model']==name and r['test_case']==c]):.4f}"
                                        for c in test), flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics_long.csv", index=False)
    (out / "params.json").write_text(json.dumps(params_out, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.config, a.models, a.out)
