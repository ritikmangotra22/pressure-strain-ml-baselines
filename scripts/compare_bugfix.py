"""Quantify the effect of the Re_tau=180 wall-point bug: superseded vs corrected E1 results.

    python scripts/compare_bugfix.py
"""
import glob
from pathlib import Path

import pandas as pd

OLD = "results/superseded_wallpoint_bug/e1_F1_parts/*/metrics_long.csv"
NEW = "results/e1_F1_parts/*/metrics_long.csv"


def load(p):
    return pd.concat(pd.read_csv(f) for f in glob.glob(p))


old, new = load(OLD), load(NEW)
k = ["model", "re_tau", "seed"]
m = old[k + ["E_Pi"]].merge(new[k + ["E_Pi"]], on=k, suffixes=("_old", "_new"))
m["abs_diff"] = (m["E_Pi_new"] - m["E_Pi_old"]).abs()
m["rel_diff"] = m["abs_diff"] / m["E_Pi_new"]
s = m.groupby(["model", "re_tau"])[["E_Pi_old", "E_Pi_new", "abs_diff", "rel_diff"]].mean()
md = ("### Effect of the wall-point bug on E1 test E_Π (superseded − corrected, seed-averaged)\n\n"
      + s.to_markdown(floatfmt=".2e")
      + f"\n\nMax relative change over all runs: {m['rel_diff'].max():.2e}; median {m['rel_diff'].median():.2e}; n = {len(m)} matched runs.\n")
Path("results/bugfix_effect.md").write_text(md)
print(md)
