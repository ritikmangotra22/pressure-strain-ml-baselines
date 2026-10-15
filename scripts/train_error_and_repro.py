"""Training error + exact-reproducibility check for every E1 final model.

For each (model, held-out fold, seed) recorded in results/e1_F1/metrics_long.csv:
  1. rebuild the training set (the four other Lee & Moser cases, per-case weight renormalisation);
  2. refit the model with the recorded hyperparameters and seed;
  3. compute the TRAINING E_Pi (on the four training cases, pooled with the same weights);
  4. recompute the TEST E_Pi and check it matches the recorded value (reproducibility).

    python scripts/train_error_and_repro.py --results results/e1_F1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps.features import FEATURE_SETS, TARGETS, build_table  # noqa: E402
from ps.io_lm import load_lm_case  # noqa: E402
from ps.metrics import e_pi  # noqa: E402
from ps.models import make_model  # noqa: E402
from run_e1 import renorm  # noqa: E402


def main(res, data_dir):
    res = Path(res)
    met = pd.read_csv(res / "metrics_long.csv")
    params = json.loads((res / "best_params.json").read_text())
    F = FEATURE_SETS["F1"]
    tables = {}
    for lab in ["0180", "0550", "1000", "2000", "5200"]:
        c = load_lm_case(Path(data_dir), lab)
        tables[c.name] = build_table(c)
    rows = []
    for _, r in met.iterrows():
        p = dict(params.get(f"{r.model}__{r.held_out}", {}))
        p.pop("fit_info", None)
        if r.model == "rf" and p.get("max_depth") == "none":
            p["max_depth"] = None
        tr = renorm(pd.concat([t for n, t in tables.items() if n != r.held_out]))
        te = tables[r.held_out]
        m = make_model(r.model, F, int(r.seed), p)
        m.fit(tr)
        e_tr = e_pi(m.predict(tr), tr[TARGETS].to_numpy(), tr["w"].to_numpy())
        e_te = e_pi(m.predict(te), te[TARGETS].to_numpy(), te["w"].to_numpy())
        rows.append({"model": r.model, "held_out": r.held_out, "re_tau": r.re_tau, "seed": r.seed,
                     "E_Pi_train": e_tr, "E_Pi_inner_val": r.inner_val_E_Pi, "E_Pi_test": r.E_Pi,
                     "E_Pi_test_refit": e_te, "repro_abs_diff": abs(e_te - r.E_Pi)})
    df = pd.DataFrame(rows)
    df.to_csv(res / "train_val_test.csv", index=False)
    g = df.groupby("model")[["E_Pi_train", "E_Pi_inner_val", "E_Pi_test", "repro_abs_diff"]].agg(["mean", "max"])
    print(g.to_string())
    order = ["ssg_published", "ssg_refit", "lrr_published", "lrr_refit", "lrr_gl_published", "lrr_gl_refit", "ridge", "rf", "lgbm", "mlp"]
    s = df.groupby("model")[["E_Pi_train", "E_Pi_inner_val", "E_Pi_test"]].mean().reindex([o for o in order if o in set(df.model)])
    s["max |test refit − recorded|"] = df.groupby("model")["repro_abs_diff"].max()
    md = ("### Table 8c — Training, inner-validation and held-out test E_Π (means over folds and seeds)\n\n"
          "Training: the four training cases of each fold. Inner validation: the mean leave-one-out error of the "
          "selected configuration during tuning; it is used for model selection and is NaN for untuned models. "
          "Test: the held-out case. The last column is the exact-reproducibility check from refitting.\n\n"
          + s.to_markdown(floatfmt=".4f"))
    (res / "table8c_train_val_test.md").write_text(md + "\n")
    print(md)
    bad = df[df.repro_abs_diff > 1e-6]
    if len(bad):
        print(f"WARNING: {len(bad)} runs did not reproduce to 1e-6:\n", bad.head(20).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default="data/raw/lee_moser_2015")
    a = ap.parse_args()
    main(a.results, a.data)
