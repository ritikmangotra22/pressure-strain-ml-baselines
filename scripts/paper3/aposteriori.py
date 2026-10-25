"""Paper 3 Stage 3b: a posteriori checks of every refitted selected model on the Lee & Moser cases
(training data only; protocol_v1 deviation 5).

For each model and LM case:
  * conditioning diagnostic (src/ps3/conditioning.py): first-Newton-step amplification of the a
    priori error on the frozen-k anisotropy problem; all 10 refit seeds;
  * frozen 1D Reynolds-stress solve (src/ps3/rsm1d.py) from the DNS state; refit seeds 0-2
    (a non-converging solve costs 30-90 s per case, so 10 seeds x 56 models x 5 cases would take
    about 40 core-hours), at most 200 Newton/pseudo-time iterations (the verified anchored oracle
    converges in 4-5); outcome class, final residual and, when converged, J_R and per-component
    errors. Failed solves are recorded, never dropped.
Classical references (published and refitted SSG, LRR-IP, LRR-IP+GL) and the anchored oracle are
run the same way. These choices (seeds 0-2, 200 iterations) were fixed on 23 Oct 2026 from a timing
test on M0 seed 0 only, before any selected model was assessed.

    python scripts/paper3/aposteriori.py --workers 2      ->  results/paper3/final/aposteriori_lm.jsonl
"""
import argparse
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import numpy as np
import final_common as F
from ps3 import conditioning, rsm1d, search

SOLVE_SEEDS = (0, 1, 2)
HOST = f"{platform.node()} {platform.system()} {platform.machine()}"   # provenance per row
MAX_ITER = 200


def assess(cc, closure, do_solve: bool) -> dict:
    """Conditioning and (optionally) the frozen solve for one closure on one ChannelCase."""
    row = {}
    t0 = time.time()
    try:
        row.update(conditioning.amplification(cc, closure))
    except Exception as e:                                  # recorded, never dropped
        row.update({"amplification": np.nan, "cond_error": repr(e)[:200]})
    row["cond_seconds"] = round(time.time() - t0, 2)
    if do_solve:
        t0 = time.time()
        try:
            r = rsm1d.solve(cc, closure, max_iter=MAX_ITER)
            row.update({"solve_status": r.status, "solve_iterations": r.iterations, "solve_residual": r.residual})
            if r.status in ("converged", "non_realizable"):
                row.update(rsm1d.stress_error(r.R, cc))
        except Exception as e:
            row.update({"solve_status": "error", "solve_error": repr(e)[:200]})
        row["solve_seconds"] = round(time.time() - t0, 2)
    return row


def classical_closures():
    cl = json.loads((F.FINAL / "classical.json").read_text())
    return {k: rsm1d.classical(v["kind"], v["coefs"]) for k, v in cl.items()}


def task(args):
    mid, seed, out_root = args
    rows = []
    if mid.startswith("classical:") or mid == "anchored_oracle":
        closure_for = (lambda cc: rsm1d.anchored_oracle(cc)) if mid == "anchored_oracle" else \
            (lambda cc, k=mid.split(":", 1)[1]: classical_closures()[k])
        do_solve = True
    else:
        fitted, ck = F.load_fitted(F.ckpt_path(out_root, mid, seed))
        closure_for = lambda cc: rsm1d.learned(fitted, ck["config"]["c_t"])
        do_solve = seed in SOLVE_SEEDS
    for case in F.CASES_LM:
        cc = search._channel_case(case)
        rows.append({"model": mid, "seed": seed, "case": case, **assess(cc, closure_for(cc), do_solve), "host": HOST})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=F.FINAL)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--recheck", nargs="*", default=None,
                    help="recompute these already-finished (model, seed) tasks, given as model:seed, into "
                         "aposteriori_recheck.jsonl (cross-machine reproducibility check; main file untouched)")
    a = ap.parse_args()
    sel = json.loads((a.out / "selected.json").read_text())
    res_f = a.out / "aposteriori_lm.jsonl"
    done = set()
    if res_f.exists():
        done = {(r["model"], r["seed"]) for r in (json.loads(l) for l in res_f.read_text().splitlines() if l.strip())}
    todo = [("anchored_oracle", -1, a.out)] + [(f"classical:{k}", -1, a.out) for k in classical_closures()]
    todo += [(m, s, a.out) for m in sel if sel[m]["config"] is not None for s in F.REFIT_SEEDS
             if F.ckpt_path(a.out, m, s).exists()]
    if a.recheck:
        want = {(m.rsplit(":", 1)[0], int(m.rsplit(":", 1)[1])) for m in a.recheck}
        todo = [t for t in todo if (t[0], t[1]) in want]
        res_f = a.out / "aposteriori_recheck.jsonl"
    else:
        todo = [t for t in todo if (t[0], t[1]) not in done and (a.only is None or t[0] in a.only)]
    print(f"{len(todo)} (model, seed) tasks", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(task, todo), 1):
            with res_f.open("a") as fh:
                fh.writelines(json.dumps(r, default=float) + "\n" for r in rows)
            amp = np.nanmedian([r.get("amplification", np.nan) for r in rows])
            st = [r.get("solve_status", "-") for r in rows]
            print(f"[{time.time() - t0:7.0f}s] {i}/{len(todo)} {rows[0]['model']} seed {rows[0]['seed']} "
                  f"median amplification {amp:.3g} solves {st}", flush=True)


if __name__ == "__main__":
    main()
