"""Paper 3 Stage 3a: refit every selected model on all five Lee & Moser cases (training data only).

protocol_v1 selection_rule.refit: epochs = the selected trial's median inner best epoch, training
seeds 0-9, no validation split (Paper 2's refit procedure). Writes

  results/paper3/final/selected.json             selected configuration per job (+ M0)
  results/paper3/final/refits/<model>/seed<s>.pt  weights, target scale, config, seed, epochs
  results/paper3/final/refit_metrics.csv         in-sample E_Pi per LM case, rti, coef_norm, FLOPs

Resumable: an existing checkpoint is kept. A non-finite fit is retried once with the same seed
(as in Paper 2) and recorded in refit_failures.json if it fails again.

    python scripts/paper3/refit_selected.py --workers 2 [--only M0] [--allow-partial --out DIR]
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

import final_common as F
from ps2 import pipeline as PL
from ps3 import search


def refit_one(args):
    mid, spec, seed, out = args
    path = F.ckpt_path(out, mid, seed)
    if path.exists():
        return None
    c, ep = spec["config"], spec["epochs"]
    train = F.lm_split(c["c_t"])
    fails = []
    for attempt in (1, 2):
        fitted, info = search.fit(c, train, seed=seed, epochs=ep)
        if info["status"] == "ok":
            break
        fails.append({"model": mid, "seed": seed, "attempt": attempt, "status": info["status"]})
    if info["status"] != "ok":
        return {"failures": fails}
    F.save_fitted(path, fitted, c, seed, ep, info)
    cases = F.lm_case_splits(c["c_t"])
    row = {"model": mid, "seed": seed, "epochs": ep, "fit_seconds": round(info["fit_seconds"], 2), "n_params": info["n_params"]}
    for k, sp in cases.items():
        row[f"E_Pi_{k}"] = PL.e_pi(fitted.predict(sp), sp.P.numpy(), sp.w)
    row["rti"] = search.return_to_isotropy(fitted)["rti_viol_mean"]
    row["coef_norm"] = search.coef_norm(fitted, train)
    row.update({k: v for k, v in search.frame_and_trace(fitted, train).items()})
    row["flops"] = search.flops_per_point(fitted, train)
    return {"row": row, "failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=F.REFIT_SEEDS)
    ap.add_argument("--allow-partial", action="store_true", help="smoke test on an incomplete search")
    ap.add_argument("--out", type=Path, default=F.FINAL)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = F.selected_models(allow_partial=a.allow_partial)
    (a.out / "selected.json").write_text(json.dumps(sel, indent=1))
    todo = [(m, s, seed, a.out) for m, s in sel.items() if s["config"] is not None
            and (a.only is None or m in a.only) for seed in a.seeds]
    print(f"{len(sel)} models, {len(todo)} refits requested", flush=True)
    t0 = time.time()
    rows_f, fails_f = a.out / "refit_metrics.csv", a.out / "refit_failures.json"
    rows = pd.read_csv(rows_f).to_dict("records") if rows_f.exists() else []
    fails = json.loads(fails_f.read_text()) if fails_f.exists() else []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(refit_one, todo), 1):
            if r is None:
                continue
            fails += r["failures"]
            if "row" in r:
                rows.append(r["row"])
                print(f"[{time.time() - t0:7.0f}s] {i}/{len(todo)} {r['row']['model']} seed {r['row']['seed']} "
                      f"mean in-sample E_Pi {np.mean([v for k, v in r['row'].items() if k.startswith('E_Pi_')]):.4f}", flush=True)
            pd.DataFrame(rows).sort_values(["model", "seed"]).to_csv(rows_f, index=False)
            fails_f.write_text(json.dumps(fails, indent=1))


if __name__ == "__main__":
    main()
