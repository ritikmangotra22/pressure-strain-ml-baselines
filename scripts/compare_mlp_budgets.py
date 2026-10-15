"""Table 18: MLP with 40 trials (original), 100 trials (cloud, reported) and 100 trials (MacBook Air M4 replicate).
    python scripts/compare_mlp_budgets.py
"""
import json
from pathlib import Path

import pandas as pd

RUNS = {"40 trials (cloud)": "results/mlp40_original/e1_F1_parts/mlp_{f}",
        "100 trials (cloud, reported)": "results/e1_F1_parts/mlp_{f}",
        "100 trials (Mac replicate)": "results/mac_mlp100/results/e1_F1_parts_mlp100/mlp_{f}"}
FOLDS = ["0180", "0550", "1000", "2000", "5200"]
rows = []
for name, pat in RUNS.items():
    for f in FOLDS:
        d = Path(pat.format(f=f))
        m = pd.read_csv(d / "metrics_long.csv")
        p = next(iter(json.loads((d / "best_params.json").read_text()).values()))
        rows.append({"run": name, "re_tau": int(f), "test_mean": m.E_Pi.mean(), "test_sd": m.E_Pi.std(),
                     "inner": m.inner_val_E_Pi.iloc[0],
                     "config": f"{p['n_layers']}×{p['width']} {p['activation']}, {p['epochs']} ep"})
df = pd.DataFrame(rows)
df.to_csv("results/mlp_budget_comparison.csv", index=False)
md = ["| Held-out Re_τ | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
for f in FOLDS:
    cells = []
    for name in RUNS:
        r = df[(df.run == name) & (df.re_tau == int(f))].iloc[0]
        cells.append(f"{r.test_mean:.3f} ± {r.test_sd:.3f} (inner {r.inner:.3f}; {r.config})")
    md.append(f"| {int(f)} | " + " | ".join(cells) + " |")
for reg, fs in (("Mean, interp.", [550, 1000, 2000]), ("Mean, extrap.", [180, 5200])):
    md.append(f"| {reg} | " + " | ".join(f"{df[(df.run == n) & df.re_tau.isin(fs)].test_mean.mean():.3f}" for n in RUNS) + " |")
Path("results/mlp_budget_comparison.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
