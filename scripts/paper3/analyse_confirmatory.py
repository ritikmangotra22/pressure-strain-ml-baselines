"""Paper 3 statistics on the confirmatory-test outputs (reads only results/paper3/confirmatory/*,
never the test tables themselves). protocol_v1 statistics:

  bootstrap  Paper 2 paired hierarchical bootstrap, 4000 replicates, resampling optimizer seeds,
             refit seeds and 10 % moving blocks of the profile; the SAME resample indices are used
             for every strategy (optimizer seeds are matched by label 3001-3005, refit seeds 0-9)
  consistent a difference is consistent if its 95 % interval excludes 0, it is larger than the
             optimizer-seed range of both strategies, and it has the same sign on both MKM cases

A "cell" is one strategy x arm: 5 selected models (one per optimizer seed) x 10 refit seeds.
Cell error on a case = mean over its models and refit seeds of E_Pi. M0 and the classical
references have no optimizer seeds (M0: refit seeds only; classical: deterministic).

The comparisons below were fixed with the RQ interpretation rules (first deliverable, Section G)
before the test was run; a negative difference means the first-named cell is more accurate.

    python scripts/paper3/analyse_confirmatory.py [--dir results/paper3/confirmatory]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
Y = ["Pi11", "Pi22", "Pi33", "Pi12"]
MULT = np.array([1, 1, 1, 2.0])
B = 4000
OPT_SEEDS = [3001, 3002, 3003, 3004, 3005]
REFIT_SEEDS = list(range(10))
MKM = ["MKM_Channel_0395", "MKM_Channel_0590"]

# (label, cell a, cell b, research question): difference = E(a) - E(b)
COMPARISONS = [
    ("random_A - M0", "random/A", "M0", "RQ1"),
    ("tpe_A - M0", "tpe/A", "M0", "RQ1"),
    ("nsga2_A - tpe_A", "nsga2/A", "tpe/A", "RQ3"),
    ("qnehvi_A - tpe_A", "qnehvi/A", "tpe/A", "RQ3"),
    ("nsga2_A - random_A", "nsga2/A", "random/A", "RQ3"),
    ("qnehvi_A - random_A", "qnehvi/A", "random/A", "RQ3"),
    ("nsga2_B - random_B", "nsga2/B", "random/B", "RQ3"),
    ("qnehvi_B - random_B", "qnehvi/B", "random/B", "RQ3"),
    ("nsga2_B - nsga2_A", "nsga2/B", "nsga2/A", "RQ3"),
    ("A2 nsga2_nogen - nsga2_A", "nsga2_nogen/A", "nsga2/A", "RQ2/RQ3"),
    ("A4 nsga2_nophys - nsga2_A", "nsga2_nophys/A", "nsga2/A", "RQ2/RQ3"),
    ("A3 wsum - tpe_A", "wsum/A", "tpe/A", "RQ2/RQ3"),
    ("nsga2_cond - nsga2_A", "nsga2_cond/A", "nsga2/A", "RQ5"),
]


def cell_of(model: str) -> str:
    if model == "M0" or model.startswith("classical:") or model == "anchored_oracle":
        return model
    strategy, arm, _ = model.split("__")
    return f"{strategy}/{arm[3:]}"


def opt_seed_of(model: str):
    return int(model.split("__")[2][4:]) if "__seed" in model else None


def block_indices(n, rng, frac=0.10):
    L = max(2, int(round(frac * n)))
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


def point_arrays(pred: pd.DataFrame, case: str):
    """{model: {refit seed: per-point weighted-norm squared error}}, plus den and w (shared grid)."""
    d = pred[pred.case == case]
    out, den, w = {}, None, None
    for (m, s), g in d.groupby(["model", "seed"], sort=False):
        P, H = g[Y].to_numpy(), g[[c + "_hat" for c in Y]].to_numpy()
        out.setdefault(m, {})[int(s)] = (((H - P) ** 2) * MULT).sum(1)
        if den is None:
            den, w = ((P ** 2) * MULT).sum(1), g.w.to_numpy()
    return out, den, w


class CaseData:
    def __init__(self, pred, case):
        self.err, self.den, self.w = point_arrays(pred, case)
        self.n = len(self.w)
        self.cells = {}
        for m in self.err:
            self.cells.setdefault(cell_of(m), []).append(m)

    def cell_error(self, cell, opt_idx, seed_idx, pts):
        """Mean E_Pi over the cell's models (optimizer seeds opt_idx) and refit seeds seed_idx."""
        ww, denom = self.w[pts], (self.w[pts] * self.den[pts]).sum()
        models = self.cells[cell]
        by_seed = {opt_seed_of(m): m for m in models}
        if None in by_seed:                                  # M0 / classical: no optimizer seed
            use = [by_seed[None]] * len(opt_idx)
        else:
            present = [by_seed[o] for o in OPT_SEEDS if o in by_seed]   # a job with no feasible model is absent
            use = [present[i % len(present)] for i in opt_idx]
        vals = []
        for m in use:
            seeds = sorted(self.err[m])
            ss = [seeds[i % len(seeds)] for i in seed_idx] if seeds != [-1] else [-1]
            vals += [np.sqrt((self.err[m][s][pts] * ww).sum() / denom) for s in ss]
        return float(np.mean(vals))

    def job_means(self, cell):
        """Per-optimizer-seed mean E_Pi (over refit seeds): the optimizer-seed range uses these."""
        allp = np.arange(self.n)
        return {opt_seed_of(m): self.cell_error_single(m, allp) for m in self.cells[cell]}

    def cell_error_single(self, m, pts):
        ww, denom = self.w[pts], (self.w[pts] * self.den[pts]).sum()
        return float(np.mean([np.sqrt((e[pts] * ww).sum() / denom) for e in self.err[m].values()]))


def analyse_case(cd: CaseData, comps, rng, n_boot=B):
    allp = np.arange(cd.n)
    allo, alls = list(range(len(OPT_SEEDS))), list(range(len(REFIT_SEEDS)))
    point = {c: cd.cell_error(c, allo, alls, allp) for c in cd.cells}
    boot = {lab: [] for lab, *_ in comps}
    for _ in range(n_boot):
        oi = list(rng.integers(0, len(OPT_SEEDS), len(OPT_SEEDS)))
        si = list(rng.integers(0, len(REFIT_SEEDS), len(REFIT_SEEDS)))
        pts = block_indices(cd.n, rng)
        cache = {}
        for lab, a, b, _ in comps:
            for c in (a, b):
                if c not in cache:
                    cache[c] = cd.cell_error(c, oi, si, pts)
            boot[lab].append(cache[a] - cache[b])
    rows = []
    for lab, a, b, rq in comps:
        bt = np.array(boot[lab])
        ra = np.ptp(list(cd.job_means(a).values())) if a not in ("M0",) and not a.startswith("classical:") else 0.0
        rb = np.ptp(list(cd.job_means(b).values())) if b not in ("M0",) and not b.startswith("classical:") else 0.0
        rows.append({"comparison": lab, "rq": rq, "a": a, "b": b, "E_a": point[a], "E_b": point[b],
                     "diff": point[a] - point[b], "ci_lo": float(np.quantile(bt, 0.025)),
                     "ci_hi": float(np.quantile(bt, 0.975)), "opt_range_a": ra, "opt_range_b": rb})
    return point, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results/paper3/confirmatory")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-boot", type=int, default=B)
    a = ap.parse_args()
    pred = pd.read_csv(a.dir / "predictions.csv.gz")
    out = a.dir / "analysis"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    all_rows, cells = [], []
    for case in sorted(pred.case.unique()):
        cd = CaseData(pred, case)
        comps = [c for c in COMPARISONS if c[1] in cd.cells and c[2] in cd.cells]
        point, rows = analyse_case(cd, comps, rng, a.n_boot)
        all_rows += [{"case": case, **r} for r in rows]
        cells += [{"case": case, "cell": c, "E_Pi": e} for c, e in point.items()]
    res = pd.DataFrame(all_rows)
    if len(res):
        res["resolved"] = (res.ci_lo > 0) | (res.ci_hi < 0)
        res["beyond_opt_range"] = res["diff"].abs() > res[["opt_range_a", "opt_range_b"]].max(axis=1)
        mk = res[res.case.isin(MKM)]
        same = mk.groupby("comparison")["diff"].apply(lambda d: len(d) == len(MKM) and (np.all(d > 0) or np.all(d < 0)))
        res["same_sign_on_mkm"] = res.comparison.map(same).fillna(False)
        res["consistent"] = res.case.isin(MKM) & res.resolved & res.beyond_opt_range & res.same_sign_on_mkm
    res.to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(cells).to_csv(out / "cell_errors.csv", index=False)
    ap_f = a.dir / "a_posteriori.jsonl"
    if ap_f.exists():
        apr = pd.DataFrame([json.loads(l) for l in ap_f.read_text().splitlines() if l.strip()])
        apr["cell"] = apr.model.map(cell_of)
        oracle_ok = apr[apr.model == "anchored_oracle"].set_index("case").solve_status.eq("converged").to_dict()
        apr["oracle_converged_on_case"] = apr.case.map(oracle_ok)
        summ = apr.groupby(["case", "cell"]).agg(
            median_amplification=("amplification", "median"), n=("amplification", "size"),
            n_solves=("solve_status", lambda s: int(s.notna().sum())),
            frac_converged=("solve_status", lambda s: float((s == "converged").sum() / max(s.notna().sum(), 1))),
            oracle_converged_on_case=("oracle_converged_on_case", "first")).reset_index()
        summ.to_csv(out / "a_posteriori_summary.csv", index=False)
    pd.set_option("display.width", 250)
    if len(res):
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
