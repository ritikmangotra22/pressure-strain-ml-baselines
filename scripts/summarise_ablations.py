"""Table 13b: ablations A1 (feature sets F2, F3) and A3 (uniform training weights) vs baseline E1 (F1, grid weights).

All variants are scored with the same grid-spacing weights on the same held-out folds.
    python scripts/summarise_ablations.py
"""
from pathlib import Path

import pandas as pd

NICE = {"ridge": "Ridge", "rf": "Random forest", "lgbm": "LightGBM"}
VARIANTS = [("Baseline (F1, grid weights)", "results/e1_F1/metrics_long.csv"),
            ("A1: F2 (+ invariants, P/ε)", "results/ablations/a1_F2/metrics_long.csv"),
            ("A1: F3 (+ wall distance Re_y)", "results/ablations/a1_F3/metrics_long.csv"),
            ("A3: uniform training weights", "results/ablations/a3_uniform/metrics_long.csv")]
rows, missing = [], []
for name, f in VARIANTS:
    if not Path(f).exists():
        missing.append(name)
        continue
    d = pd.read_csv(f)
    d = d[d.model.isin(NICE)]
    for m in NICE:
        x = d[d.model == m]
        if x.empty:
            continue
        n_folds = x.re_tau.nunique()
        rows.append({"Model": NICE[m], "Variant": name,
                     **{str(r): f"{x[x.re_tau == r].E_Pi.mean():.3f}" for r in (180, 550, 1000, 2000, 5200)},
                     "Mean, interp.": f"{x[x.regime == 'interp'].E_Pi.mean():.3f}",
                     "Mean, extrap.": f"{x[x.regime == 'extrap'].E_Pi.mean():.3f}",
                     "Folds": n_folds})
df = pd.DataFrame(rows)
df["_o"] = df.Model.map({v: i for i, v in enumerate(NICE.values())})
df = df.sort_values(["_o"], kind="stable").drop(columns="_o")
cols = list(df.columns)
md = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
out = "\n".join(md)
if missing:
    out += "\n\nNot yet available: " + ", ".join(missing)
Path("results/ablations").mkdir(parents=True, exist_ok=True)
Path("results/ablations/table13b_ablations.md").write_text(out + "\n")
print(out)
