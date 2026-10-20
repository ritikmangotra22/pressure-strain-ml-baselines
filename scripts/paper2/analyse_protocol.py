"""Stages D and E of protocol v1: verification and statistics, computed only from saved artifacts.

  * recomputes E_Pi from every saved prediction file and compares it with metrics.csv;
  * checks split integrity (test cases never among the training cases of their job);
  * aggregates model x fold results over 3 tuning runs x 10 seeds (primary estimate);
  * factorial effects Delta_inv, Delta_con, Delta_int with the paired hierarchical bootstrap,
    the tuning-dominated rule and the consistency rule fixed in protocol v1;
  * ablations against their parent (run 1, same hyperparameters), paired bootstrap over seeds
    and point blocks;
  * structural diagnostics, cost and the run manifest.

    python scripts/paper2/analyse_protocol.py [--root results/paper2/main] [--out results/paper2/analysis]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTO = json.loads((ROOT / "configs/paper2/protocol_v1.json").read_text())
E1 = PROTO["folds"]["E1"]
INTERP, EXTRAP = PROTO["folds"]["interpolation"], PROTO["folds"]["extrapolation"]
MODELS = ["M1", "M2", "M3", "M4"]
COMP = ["11", "22", "33", "12", "13", "23"]
MULT = np.array([1, 1, 1, 2, 2, 2.0])
B = 4000


def load_pred(f):
    d = pd.read_csv(f)
    P = d[[f"Pi{c}" for c in COMP]].to_numpy()
    H = d[[f"Pi{c}_hat" for c in COMP]].to_numpy()
    return d, P, H


def e_pi_from(P, H, w):
    w = w / w.sum()
    return float(np.sqrt((w * (((H - P) ** 2) * MULT).sum(1)).sum() / (w * ((P ** 2) * MULT).sum(1)).sum()))


def verify_job(d: Path):
    """Recompute E_Pi from predictions; return (max abs diff, n files, integrity ok)."""
    met = pd.read_csv(d / "metrics.csv")
    diffs, n = [], 0
    for _, r in met.iterrows():
        dd, P, H = load_pred(d / "predictions" / f"{r.case}__seed{int(r.seed)}.csv")
        diffs.append(abs(e_pi_from(P, H, dd.w.to_numpy()) - r.E_Pi))
        n += 1
        if set(dd.case) != {r.case}:
            return None, n, False
    ok = True
    sel = d / "selected.json"
    if sel.exists():
        s = json.loads(sel.read_text())
        ok = not (set(s["test_cases"]) & set(s["train_cases"]))
    return max(diffs) if diffs else None, n, ok


def collect(root: Path):
    rows = []
    for st in sorted(root.glob("*/run*/*/status.json")):
        d = st.parent
        m, run, outer = d.parts[-3], d.parts[-2], d.parts[-1]
        met = pd.read_csv(d / "metrics.csv")
        met["model"], met["run"], met["outer"] = m, int(run[3:]), outer
        rows.append(met)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def point_errors(root, model, run, case, seeds):
    """Per-point weighted squared error and squared norm for each seed: arrays (n_seeds, n_points)."""
    num, den, w = [], None, None
    for s in seeds:
        f = root / model / f"run{run}" / (case if case in E1 else "E2") / "predictions" / f"{case}__seed{s}.csv"
        d, P, H = load_pred(f)
        num.append((((H - P) ** 2) * MULT).sum(1))
        den, w = ((P ** 2) * MULT).sum(1), d.w.to_numpy()
    return np.array(num), den, w


def block_indices(n, rng, frac=0.10):
    L = max(2, int(round(frac * n)))
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(s, s + L) for s in starts])[:n]


def effects_for_case(root, case, runs, seeds, rng):
    """Paired hierarchical bootstrap of Delta_inv, Delta_con, Delta_int for one test case."""
    E = {}
    for m in MODELS:
        E[m] = {r: point_errors(root, m, r, case, seeds) for r in runs}
    w = E["M1"][runs[0]][2]
    den = E["M1"][runs[0]][1]
    n = len(w)

    def est(run_idx, seed_idx, pts):
        out = {}
        for m in MODELS:
            vals = []
            for r in run_idx:
                num = E[m][r][0]
                ww = w[pts]
                denom = (ww * den[pts]).sum()
                vals.append(np.sqrt((num[seed_idx][:, pts] * ww).sum(1) / denom))
            out[m] = float(np.mean(vals))
        return out

    allpts = np.arange(n)
    point = est(runs, np.arange(len(seeds)), allpts)
    fx = lambda e: {"inv": 0.5 * ((e["M2"] - e["M1"]) + (e["M4"] - e["M3"])),
                    "con": 0.5 * ((e["M3"] - e["M1"]) + (e["M4"] - e["M2"])),
                    "int": (e["M4"] - e["M3"]) - (e["M2"] - e["M1"]),
                    "M1_M2": e["M2"] - e["M1"], "M1_M3": e["M3"] - e["M1"], "M2_M4": e["M4"] - e["M2"],
                    "M3_M4": e["M4"] - e["M3"]}
    boot = []
    for _ in range(B):
        ri = list(rng.choice(runs, size=len(runs), replace=True))
        si = rng.integers(0, len(seeds), size=len(seeds))
        boot.append(fx(est(ri, si, block_indices(n, rng))))
    bt = pd.DataFrame(boot)
    f0 = fx(point)
    ci = {k: (float(bt[k].quantile(0.025)), float(bt[k].quantile(0.975))) for k in f0}
    return point, f0, ci


def run_level_means(df):
    g = df.groupby(["model", "outer", "case", "run"]).E_Pi.mean().reset_index()
    return g


def main(root, out, abl_root):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    df = collect(root)
    if df.empty:
        print("no completed jobs")
        return
    # ---- Stage D: verification -------------------------------------------------------------
    ver = []
    for st in sorted(root.glob("*/run*/*/status.json")):
        diff, n, ok = verify_job(st.parent)
        ver.append({"job": "/".join(st.parent.parts[-3:]), "max_abs_recomputed_minus_reported": diff, "n_files": n, "split_ok": ok})
    ver = pd.DataFrame(ver)
    ver.to_csv(out / "verification.csv", index=False)
    # ---- manifest and cost ----------------------------------------------------------------
    man = []
    for st in sorted(root.glob("*/run*/*/status.json")):
        s = json.loads(st.read_text())
        sel = json.loads((st.parent / "selected.json").read_text())
        man.append({"job": s["job"], "runtime_s": s["runtime_s"], "tune_s": s["tune_s"], "n_trials": s["n_trials"],
                    "refit_failures": len(s["refit_failures"]), "code_version": s["code_version"],
                    "sampler_seed": sel["sampler_seed"], **{k: sel[k] for k in ("n_layers", "width", "activation", "lr", "weight_decay", "epochs", "inner_val_E_Pi")},
                    "finished": s["finished"]})
    man = pd.DataFrame(man)
    man.to_csv(out / "run_manifest.csv", index=False)
    # ---- model x fold table -----------------------------------------------------------------
    rl = run_level_means(df)
    tab = rl.groupby(["model", "case"]).E_Pi.agg(primary_mean="mean", between_run_sd="std", run_min="min", run_max="max", n_runs="count").reset_index()
    tab["between_run_range"] = tab.run_max - tab.run_min
    sd = df.groupby(["model", "case", "run"]).E_Pi.std().groupby(["model", "case"]).mean().rename("mean_seed_sd")
    tab = tab.merge(sd.reset_index(), on=["model", "case"])
    # secondary estimate: best-inner run
    sec = []
    for (m, outer), g in man.assign(model=man.job.str.split("__").str[0], outer=man.job.str.split("__").str[2]).groupby(["model", "outer"]):
        r = int(g.loc[g.inner_val_E_Pi.idxmin(), "job"].split("__")[1][3:])
        sel = rl[(rl.model == m) & (rl.outer == outer) & (rl.run == r)]
        for _, x in sel.iterrows():
            sec.append({"model": m, "case": x.case, "best_inner_run": r, "secondary_E_Pi": x.E_Pi})
    if sec:
        tab = tab.merge(pd.DataFrame(sec), on=["model", "case"], how="left")
    tab.to_csv(out / "model_by_fold.csv", index=False)
    # regions and diagnostics
    cols = [c for c in df.columns if c.startswith("E_Pi_")] + ["eq_max", "eq_reflect_y_max", "trace_rel_max", "trace_rel_mean",
                                                              "iso_limit_max", "asym_max", "f32_f64_pred_rel_max", "T3_E_Pi_rotated_mean",
                                                              "D3_frac_negative", "D3_wmean", "offchannel_rel_max", "p1_E_Pi"]
    reg = df.groupby(["model", "case"])[[c for c in cols if c in df.columns]].agg(["mean", "max"])
    reg.to_csv(out / "regions_and_diagnostics.csv")
    struct = df.groupby("model")[["eq_max", "trace_rel_max", "iso_limit_max", "asym_max", "f32_f64_pred_rel_max"]].max()
    struct.to_csv(out / "structural_max_by_model.csv")
    # ---- effects ------------------------------------------------------------------------------
    eff_rows = []
    complete_cases = [c for c in sorted(df.case.unique())
                      if all(((df.model == m) & (df.case == c)).any() for m in MODELS)]
    for case in complete_cases:
        runs = sorted(set.intersection(*[set(df[(df.model == m) & (df.case == case)].run) for m in MODELS]))
        seeds = sorted(set.intersection(*[set(df[(df.model == m) & (df.case == case)].seed) for m in MODELS]))
        point, f0, ci = effects_for_case(root, case, runs, seeds, rng)
        rng_runs = {m: float(rl[(rl.model == m) & (rl.case == case) & rl.run.isin(runs)].E_Pi.agg(lambda x: x.max() - x.min())) for m in MODELS}
        involved = {"inv": MODELS, "con": MODELS, "int": MODELS, "M1_M2": ["M1", "M2"], "M1_M3": ["M1", "M3"],
                    "M2_M4": ["M2", "M4"], "M3_M4": ["M3", "M4"]}
        for k, v in f0.items():
            lo, hi = ci[k]
            dom = abs(v) <= max(rng_runs[m] for m in involved[k]) if len(runs) > 1 else None
            eff_rows.append({"case": case, "regime": "interp" if case in INTERP else ("extrap" if case in EXTRAP else "E2"),
                             "effect": k, "estimate": v, "ci_lo": lo, "ci_hi": hi, "resolved": bool(lo > 0 or hi < 0),
                             "tuning_dominated": dom, "n_runs": len(runs), "n_seeds": len(seeds),
                             **{f"E_{m}": point[m] for m in MODELS}})
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(out / "effects.csv", index=False)
    cons = []
    for k in ["inv", "con", "int", "M1_M2", "M1_M3", "M2_M4", "M3_M4"]:
        g = eff[(eff.effect == k) & (eff.regime == "interp")] if not eff.empty else eff
        for sign in (-1, 1):
            n_res = int(((g.resolved) & (np.sign(g.estimate) == sign) & (g.tuning_dominated == False)).sum()) if len(g) else 0  # noqa: E712
            cons.append({"effect": k, "sign": "lowers error" if sign < 0 else "raises error",
                         "interp_folds_resolved_not_tuning_dominated": n_res, "consistent": n_res >= 2, "interp_folds_available": len(g)})
    pd.DataFrame(cons).to_csv(out / "consistency.csv", index=False)
    # ---- ablations ----------------------------------------------------------------------------
    if abl_root.exists():
        arows = []
        for st in sorted(abl_root.glob("*/*/*/status.json")):
            d = st.parent
            s = json.loads(st.read_text())
            pm, _, outer = s["parent"].split("__")
            am = pd.read_csv(d / "metrics.csv")
            seeds = sorted(am.seed)
            pnum, den, w = point_errors(root, pm, 1, outer, seeds)
            anum = np.array([(((load_pred(d / "predictions" / f"{outer}__seed{x}.csv")[2] - load_pred(d / "predictions" / f"{outer}__seed{x}.csv")[1]) ** 2) * MULT).sum(1) for x in seeds])
            n = len(w)
            f = lambda num, si, pts: float(np.mean(np.sqrt((num[si][:, pts] * w[pts]).sum(1) / (w[pts] * den[pts]).sum())))
            base = f(anum, np.arange(len(seeds)), np.arange(n)) - f(pnum, np.arange(len(seeds)), np.arange(n))
            bt = []
            for _ in range(B):
                si, pts = rng.integers(0, len(seeds), len(seeds)), block_indices(n, rng)
                bt.append(f(anum, si, pts) - f(pnum, si, pts))
            lo, hi = np.percentile(bt, [2.5, 97.5])
            arows.append({"ablation": d.parts[-3], "variant": d.parts[-2], "case": outer, "parent": s["parent"],
                          "E_Pi_ablation": f(anum, np.arange(len(seeds)), np.arange(n)),
                          "E_Pi_parent_run1": f(pnum, np.arange(len(seeds)), np.arange(n)), "diff": base, "ci_lo": lo, "ci_hi": hi,
                          "resolved": bool(lo > 0 or hi < 0), "trace_rel_max": am.trace_rel_max.max(), "eq_max": am.eq_max.max(),
                          "T3_E_Pi_rotated_mean": am.T3_E_Pi_rotated_mean.mean(), "refit_failures": len(s["refit_failures"])})
        pd.DataFrame(arows).to_csv(out / "ablations.csv", index=False)
    summary = {"n_jobs_complete": int(len(man)), "verification_max_abs_diff": float(ver.max_abs_recomputed_minus_reported.max()),
               "all_splits_ok": bool(ver.split_ok.all()), "structural_max_by_model": struct.to_dict(),
               "total_runtime_core_hours": float(man.runtime_s.sum() / 3600)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results/paper2/main"))
    ap.add_argument("--abl", default=str(ROOT / "results/paper2/ablations"))
    ap.add_argument("--out", default=str(ROOT / "results/paper2/analysis"))
    a = ap.parse_args()
    main(Path(a.root), Path(a.out), Path(a.abl))
