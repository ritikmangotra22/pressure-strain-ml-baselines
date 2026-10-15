"""Tuning-budget sensitivity from the E1 job logs: best inner E_Pi after 40 trials vs after all trials.
Used to judge how much the MLP's reduced budget (40 trials, deviation D1) might cost.
    python scripts/tuning_convergence.py
"""
import re
from pathlib import Path

import numpy as np

LOGS = ["results/log_classical_ridge_rf.txt", "results/log_lgbm.txt"] + \
       [f"results/log_mlp100_{f}.txt" for f in ("0180", "0550", "1000", "2000", "5200")]
rows = []
for log in LOGS:
    vals, model = [], None
    for line in Path(log).read_text().splitlines():
        m = re.match(r"\s*\[tune (\w+)\] trial (\d+)/(\d+) (\w+) value=(\S+)", line)
        if m:
            model = m.group(1)
            v = float(m.group(5)) if m.group(4) == "COMPLETE" else np.nan
            vals.append(v)
            continue
        m = re.match(r"\[LM_Channel_(\d+)\] (\w+): E_Pi", line)
        if m and vals and m.group(2) == model:
            v = np.array(vals)
            best = np.fmin.accumulate(np.where(np.isnan(v), np.inf, v))
            rows.append((model, int(m.group(1)), len(v), best[min(39, len(v) - 1)], best[-1],
                         int(np.nanargmin(v)) + 1, int(np.isnan(v).sum())))
            vals = []
out = ["| Model | Held-out Re_τ | Trials | Best inner E_Π after 40 | Best inner E_Π, final | Relative gain after trial 40 | Trial of best | Pruned |",
       "|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r[0], r[1])):
    out.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {100 * (r[3] - r[4]) / r[3]:.1f} % | {r[5]} | {r[6]} |")
t = "\n".join(out)
Path("results/tuning_convergence.md").write_text(t + "\n")
print(t)
