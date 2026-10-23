"""Paper 3 search analysis on INNER objectives only (no test data are read).

For every completed job: hypervolume history, final non-dominated set, and the configuration chosen
by the protocol_v1 selection rule. Hypervolume definition (protocol_v1 deviation 2, a clarification
fixed on 22 Oct 2026 before any comparison between strategies was made):

  each objective is divided by M0's value (f_err 0.06307, f_gen 1.166, f_cost 103278 FLOPs;
  f_phys: Arm A 1.0, Arm B not divided because M0's Arm-B value is 0), and the hypervolume is
  computed against the reference point 2 in every normalised dimension (Arm B f_phys: 1.0 raw).
  Points outside the reference box contribute nothing. TPE optimises f_err only, but its
  trials are scored on the same four objectives so that its hypervolume is comparable.

  Tie tolerance in the selection rule: the frozen text gives 0.0073, M0's seed SD of f_err, written
  when f_gen was still an absolute error. With f_gen now a ratio, the tolerance is M0's seed SD of
  f_gen itself, 0.0721 (same deviation 2).

    python scripts/paper3/analyse_search.py [--seed 3001]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from optuna._hypervolume import compute_hypervolume

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "results/paper3/search"
OUT = ROOT / "results/paper3/analysis"
PROTO = json.loads((ROOT / "configs/paper3/protocol_v1.json").read_text())
M0 = json.loads((ROOT / "results/paper3/reference_m0/m0_inner.json").read_text())
OBJ = ["f_err", "f_gen", "f_phys", "f_cost"]
CASES = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"]
F_ERR0 = float(np.mean([np.mean(list(v.values())) for v in M0["fold_E_Pi_per_seed"].values()]))
_ref = np.array([M0["fold_E_Pi_mean"][c] for c in CASES])
F_GEN0 = float(np.mean([max(np.array([v[c] for c in CASES]) / _ref) for v in M0["fold_E_Pi_per_seed"].values()]))
SCALE = {"A": np.array([F_ERR0, F_GEN0, 1.0, M0["f_cost_flops"]]),
         "B": np.array([F_ERR0, F_GEN0, 1.0, M0["f_cost_flops"]])}
REFPT = {"A": np.array([2.0, 2.0, 2.0, 2.0]), "B": np.array([2.0, 2.0, 1.0, 2.0])}
TIE = 0.0721   # seed SD of M0's f_gen over seeds 0-4 (deviation 2: the 0.0073 in the frozen text is f_err's SD)


def load(job):
    rows = [json.loads(l) for l in (S / job / "evals.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("objectives")]
    df = pd.DataFrame([{**{k: r[k] for k in r if k in ("trial", "arch", "n_layers", "width", "activation", "lr",
                                                       "weight_decay", "dropout", "c_t", "n_basis", "inv_set",
                                                       "median_best_epoch", "propose_to_done_s", "trace_rel_mean",
                                                       "rot_excess", "rti", "coef_norm", "f_cond")},
                         **r["objectives"]} for r in rows]).sort_values("trial").reset_index(drop=True)
    return df


def nondominated(P):
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if keep[i]:
            dom = np.all(P <= P[i], 1) & np.any(P < P[i], 1)
            if dom.any():
                keep[i] = False
    return keep


def hv(P, arm):
    Z = P / SCALE[arm]
    Z = Z[np.all(Z < REFPT[arm], 1)]
    if not len(Z):
        return 0.0
    Z = Z[nondominated(Z)]
    return float(compute_hypervolume(Z, REFPT[arm]))


def select(df, arm, strategy):
    """protocol_v1 selection rule (TPE: lowest f_err, the Paper 2 procedure; nsga2_cond: deviation 5)."""
    if strategy == "tpe":
        return df.loc[df.f_err.idxmin()]
    if strategy == "wsum":                                  # A3 (deviation 4): its own scalar
        w = (df.f_err / F_ERR0 + df.f_gen / F_GEN0 + df.f_phys + df.f_cost / M0["f_cost_flops"]) / 4
        return df.loc[w.idxmin()]
    objs = OBJ + (["f_cond"] if strategy == "nsga2_cond" else [])
    P = df[objs].to_numpy()
    front = df[nondominated(P)]
    if strategy == "nsga2_cond":
        ok = front[front.f_cond <= 1.0]
        front = ok if len(ok) else front.loc[[front.f_cond.idxmin()]]
    feas = front[(front.f_phys <= (1.0 if arm == "A" else np.inf)) & (front.f_cost <= 2 * M0["f_cost_flops"])]
    if arm == "B":
        feas = feas[(feas.trace_rel_mean <= 1e-3) & (feas.rot_excess <= 0.01)]
    if not len(feas):
        return None
    tied = feas[feas.f_gen <= feas.f_gen.min() + TIE]      # within M0's seed SD of f_gen
    return tied.loc[tied.f_cost.idxmin()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hist = [], []
    for d in sorted(S.iterdir()):
        if not (d / "status.json").exists():
            continue
        strategy, arm, seed = d.name.split("__")
        arm, seed = arm[3:], int(seed[4:])
        if a.seed and seed != a.seed:
            continue
        df = load(d.name)
        P = df[OBJ].to_numpy()
        for n in (20, 50, 100, 150, 200):
            hist.append({"strategy": strategy, "arm": arm, "seed": seed, "n_eval": n, "hv": hv(P[:n], arm)})
        sel = select(df, arm, strategy)
        st = json.loads((d / "status.json").read_text())
        rows.append({"strategy": strategy, "arm": arm, "seed": seed, "hv_200": hv(P, arm),
                     "n_front": int(nondominated(P).sum()), "best_f_err": df.f_err.min(),
                     "n_failed": st["n_failed_fits"], "n_interrupted": st["n_interrupted"],
                     "median_train_s": df.propose_to_done_s.median(),
                     **({f"sel_{k}": sel[k] for k in OBJ + ["arch", "n_layers", "width", "activation", "dropout", "c_t",
                                                              "n_basis", "inv_set", "trial"] if k in sel} if sel is not None else {"sel_none": True})})
    pd.DataFrame(rows).to_csv(OUT / "jobs_summary.csv", index=False)
    pd.DataFrame(hist).to_csv(OUT / "hv_history.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(hist).pivot_table(index=["arm", "strategy", "seed"], columns="n_eval", values="hv").round(4))


if __name__ == "__main__":
    main()
