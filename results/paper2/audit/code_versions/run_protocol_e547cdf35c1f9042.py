"""Protocol v1 job runner: main experiment (M1-M4 x 3 tuning runs x 6 outer splits) and ablations.

Every job writes into its own directory and a status.json when complete, so the queue can be
stopped and resumed; an incomplete job is rerun from scratch (never partially reused).

    python scripts/paper2/run_protocol.py --list main
    python scripts/paper2/run_protocol.py --queue main --parallel 2 --start-budget-min 55
    python scripts/paper2/run_protocol.py --queue ablations --parallel 2 --start-budget-min 55
    python scripts/paper2/run_protocol.py --pilot          # Stage B smoke test, results/paper2/pilot
    python scripts/paper2/run_protocol.py --job M2__run1__LM_Channel_0550   # one job
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, pipeline as PL  # noqa: E402

PROTO = json.loads((ROOT / "configs/paper2/protocol_v1.json").read_text())
E1 = PROTO["folds"]["E1"]
MODELS = ["M1", "M2", "M3", "M4"]
TUNER_SEEDS = PROTO["tuning"]["tuner_sampler_seeds"]
SEEDS = PROTO["training"]["seeds_final_refit"]
N_TRIALS = PROTO["tuning"]["trials_per_run"]
C_T = PROTO["inputs"]["time_scale"]["C_T"]
OUT_MAIN = ROOT / "results/paper2/main"
OUT_ABL = ROOT / "results/paper2/ablations"
OUT_PILOT = ROOT / "results/paper2/pilot"


def code_version():
    import hashlib
    h = hashlib.sha256()
    for p in sorted((ROOT / "src/ps2").glob("*.py")) + [Path(__file__)]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def splits(scheme="durbin"):
    lm, hj = data.load("LM"), data.load("HJ")
    S = {c: PL.prepare(lm[lm.case == c], scheme, C_T) for c in E1}
    S.update({c: PL.prepare(hj[hj.case == c], scheme, C_T) for c in sorted(hj.case.unique())})
    return S


def main_jobs():
    jobs = []
    for k in range(1, len(TUNER_SEEDS) + 1):           # run 1 for everything first
        for outer in E1 + ["E2"]:
            for m in MODELS:
                jobs.append(f"{m}__run{k}__{outer}")
    return jobs


def ablation_jobs():
    jobs = []
    for outer in E1:
        for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            jobs.append(f"A-soft__M3_soft-lam{lam:g}__{outer}")
        for m in MODELS:
            jobs.append(f"A-scale__{m}__{outer}")
        jobs.append(f"A-basis__M4_basis3__{outer}")
        for m in ("M2", "M4"):
            jobs.append(f"A-inv-5__{m}__{outer}")
            jobs.append(f"A-inv-4__{m}__{outer}")
        jobs.append(f"A-lin__M4_lin__{outer}")
        jobs.append(f"A-aug__M1__{outer}")
    return jobs


def job_dir(job, pilot=False):
    if pilot:
        return OUT_PILOT / job.replace("__", "/")
    return (OUT_MAIN if not job.startswith("A-") else OUT_ABL) / job.replace("__", "/")


def refit_and_evaluate(name, train, tests, params, d: Path, inv_set="P1", lam_tr=0.0, seeds=SEEDS,
                       augment=0):
    rows, failures = [], []
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "histories").mkdir(exist_ok=True)
    for s in seeds:
        tr = PL.augment_rotations(train, augment, s) if augment else train
        for attempt in (1, 2):
            fitted, info, hist = PL.fit(name, tr, params, seed=s, epochs=params["epochs"], inv_set=inv_set, lam_tr=lam_tr)
            if info["status"] == "ok":
                break
            failures.append({"seed": s, "attempt": attempt, "status": info["status"]})
        if info["status"] != "ok":
            continue
        torch.save({"state_dict": fitted.model.state_dict(), "s": fitted.s, "params": params, "seed": s},
                   d / "checkpoints" / f"seed{s}.pt")
        pd.DataFrame(hist).to_csv(d / "histories" / f"seed{s}.csv", index=False)
        for case, te in tests.items():
            t0 = time.perf_counter()
            met, P_hat = PL.evaluate(fitted, te)
            infer = (time.perf_counter() - t0)
            rows.append({"case": case, "seed": s, **info, **met, "eval_seconds_incl_diagnostics": round(infer, 2)})
            pr = te.df[["case", "y_plus", "y_delta", "region", "w"]].copy()
            for (i, j), nm in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)], ["11", "22", "33", "12", "13", "23"]):
                pr[f"Pi{nm}"] = te.P.numpy()[:, i, j]
                pr[f"Pi{nm}_hat"] = P_hat[:, i, j]
            pr.to_csv(d / "predictions" / f"{case}__seed{s}.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "metrics.csv", index=False)
    return rows, failures


def run_main_job(job, pilot=False):
    m, run, outer = job.split("__")
    k = int(run[3:])
    d = job_dir(job, pilot)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = splits()
    if outer == "E2":
        train_cases = {c: S[c] for c in E1}
        tests = {c: S[c] for c in S if c.startswith("HJ")}
    else:
        train_cases = {c: S[c] for c in E1 if c != outer}
        tests = {outer: S[outer]}
    assert all(t not in train_cases for t in tests), "held-out case in training cases"
    logf = open(d / "log.txt", "a")
    log = lambda s: (logf.write(s + "\n"), logf.flush())
    log(f"job {job} start {time.ctime()} code {code_version()}")
    n_trials = 3 if pilot else N_TRIALS
    params, records = PL.tune(m, train_cases, TUNER_SEEDS[k - 1], n_trials=n_trials, inv_set="P1", log=log)
    pd.DataFrame(records).to_csv(d / "trials.csv", index=False)
    PL.save_json(d / "selected.json", {**params, "sampler_seed": TUNER_SEEDS[k - 1], "train_cases": list(train_cases),
                                       "test_cases": list(tests)})
    seeds = [0, 1] if pilot else SEEDS
    rows, failures = refit_and_evaluate(m, PL.concat(list(train_cases.values())), tests, params, d, seeds=seeds)
    status = {"job": job, "done": True, "pilot": pilot, "runtime_s": round(time.time() - t0, 1),
              "tune_s": params["tune_seconds"], "n_trials": len(records), "refit_failures": failures,
              "code_version": code_version(), "torch": torch.__version__, "precision": PROTO["precision"],
              "finished": time.ctime()}
    PL.save_json(d / "status.json", status)
    log(f"job {job} done in {status['runtime_s']} s")


def run_ablation_job(job, pilot=False):
    abl, variant, outer = job.split("__")
    d = job_dir(job, pilot)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lam, aug, scheme, inv_set = 0.0, 0, "durbin", "P1"
    if abl == "A-soft":
        name, lam, parent = "M3_soft", float(variant.split("lam")[1]), "M3"
    elif abl == "A-scale":
        name, parent, scheme = variant, variant, "k/eps"
    elif abl == "A-basis":
        name, parent = "M4_basis3", "M4"
    elif abl in ("A-inv-5", "A-inv-4"):
        name, parent, inv_set = variant, variant, ("E4" if abl == "A-inv-5" else "four")
    elif abl == "A-lin":
        name, parent = "M4_lin", "M4"
    elif abl == "A-aug":
        name, parent, aug = "M1", "M1", 8
    sel = (OUT_PILOT if pilot else OUT_MAIN) / parent / "run1" / outer / "selected.json"
    if not (sel.parent / "status.json").exists():
        raise RuntimeError(f"parent job {parent}__run1__{outer} not complete")
    params = json.loads(sel.read_text())
    S = splits(scheme)
    train = PL.concat([S[c] for c in E1 if c != outer])
    rows, failures = refit_and_evaluate(name, train, {outer: S[outer]}, params, d, inv_set=inv_set, lam_tr=lam, augment=aug,
                                        seeds=[0, 1] if pilot else SEEDS)
    PL.save_json(d / "status.json", {"job": job, "done": True, "parent": f"{parent}__run1__{outer}",
                                     "variant": {"name": name, "lambda_tr": lam, "augment": aug, "scheme": scheme, "inv_set": inv_set},
                                     "runtime_s": round(time.time() - t0, 1), "refit_failures": failures,
                                     "code_version": code_version(), "finished": time.ctime()})


def run_job(job, pilot=False):
    try:
        (run_ablation_job if job.startswith("A-") else run_main_job)(job, pilot)
    except Exception:
        d = job_dir(job, pilot)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"error_{int(time.time())}.txt").write_text(traceback.format_exc())
        raise


def pending(kind):
    jobs = main_jobs() if kind == "main" else ablation_jobs()
    return [j for j in jobs if not (job_dir(j) / "status.json").exists()]


def queue(kind, parallel, start_budget_min):
    """Run pending jobs as subprocesses, `parallel` at a time; start no new job after the budget."""
    t0 = time.time()
    running = {}
    todo = pending(kind)
    if kind == "ablations":   # only ablations whose parent run-1 job is complete
        todo = [j for j in todo if (OUT_MAIN / {"A-soft": "M3", "A-basis": "M4", "A-lin": "M4", "A-aug": "M1"}.get(
            j.split("__")[0], j.split("__")[1]) / "run1" / j.split("__")[2] / "status.json").exists()]
    while todo or running:
        while todo and len(running) < parallel and (time.time() - t0) / 60 < start_budget_min:
            j = todo.pop(0)
            log = (job_dir(j).parent / f"{j.split('__')[-1]}.stdout.txt")
            log.parent.mkdir(parents=True, exist_ok=True)
            running[j] = subprocess.Popen([sys.executable, __file__, "--job", j], stdout=open(log, "a"), stderr=subprocess.STDOUT)
            print(f"[{time.strftime('%H:%M:%S')}] start {j}", flush=True)
        if (time.time() - t0) / 60 >= start_budget_min:
            todo = []
        for j, p in list(running.items()):
            if p.poll() is not None:
                print(f"[{time.strftime('%H:%M:%S')}] end {j} rc={p.returncode} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                del running[j]
        time.sleep(5)
    left = pending(kind)
    print(f"queue {kind}: {len(left)} jobs pending", flush=True)


def pilot():
    for m in MODELS:
        run_job(f"{m}__run1__LM_Channel_0550", pilot=True)
    run_job("M4__run1__E2", pilot=True)
    o = "LM_Channel_0550"
    for j in ["A-soft__M3_soft-lam1__", "A-scale__M2__", "A-basis__M4_basis3__", "A-inv-5__M4__", "A-inv-4__M2__",
              "A-lin__M4_lin__", "A-aug__M1__"]:
        run_job(j + o, pilot=True)
    print("pilot done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--queue", choices=["main", "ablations"])
    ap.add_argument("--list", choices=["main", "ablations"])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--start-budget-min", type=float, default=55)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        pilot()
    elif a.job:
        run_job(a.job)
    elif a.queue:
        queue(a.queue, a.parallel, a.start_budget_min)
    elif a.list:
        jobs = main_jobs() if a.list == "main" else ablation_jobs()
        done = [j for j in jobs if (job_dir(j) / "status.json").exists()]
        print(f"{len(done)}/{len(jobs)} done; pending: {len(jobs) - len(done)}")
