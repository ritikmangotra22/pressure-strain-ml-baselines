"""RQ2: Spearman correlation between objectives along each multi-objective job's non-dominated set
(inner objectives only). A trade-off exists only if two objectives are negatively correlated on
the front (first deliverable, Section G).

    python scripts/paper3/front_correlations.py  ->  results/paper3/analysis/front_correlations.csv
"""
import itertools

import pandas as pd
from scipy.stats import spearmanr

import analyse_search as A

MULTI = ("nsga2", "qnehvi", "random", "nsga2_cond", "nsga2_nogen", "nsga2_nophys")


def main():
    rows = []
    for d in sorted(A.S.iterdir()):
        if not (d / "status.json").exists():
            continue
        s, arm, seed = d.name.split("__")
        if s not in MULTI:
            continue
        df = A.load(d.name)
        front = df[A.nondominated(df[A.OBJ].to_numpy())]
        for a, b in itertools.combinations(A.OBJ, 2):
            rows.append({"strategy": s, "arm": arm[3:], "seed": int(seed[4:]), "pair": f"{a}~{b}",
                         "rho": spearmanr(front[a], front[b]).statistic, "n_front": len(front)})
    r = pd.DataFrame(rows)
    r.to_csv(A.OUT / "front_correlations.csv", index=False)
    print(r.groupby("pair").rho.agg(["median", "min", "max"]).round(2))


if __name__ == "__main__":
    main()
