"""Paper 3 search runner (draft protocol; main runs start only after protocol_v1 is frozen).

One job = one strategy x one arm x one optimizer seed, N_EVAL candidate evaluations, scored on
the five Lee & Moser cases by inner leave-one-Re_tau-out (1 training seed per candidate). No test
data are read. Every trial, including failed fits, is written to evals.jsonl and counts against
the budget. Resumable: an interrupted job is restarted from scratch (never partially reused).

    python scripts/paper3/run_search.py --strategy nsga2 --arm A --seed 3001 [--n-eval 200]
"""
import argparse
import json
import math
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, pipeline as PL  # noqa: E402
from ps3 import search  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
REF = json.loads((ROOT / "results/paper3/reference_m0/m0_inner.json").read_text())
OBJ = ["f_err", "f_gen", "f_phys", "f_cost"]


def suggest(trial, arm):
    """Static search space: n_basis and inv_set are always drawn and ignored for M1/M3, so every
    sampler (BoTorch included) sees the same fixed space."""
    c = {"arch": "M4" if arm == "A" else trial.suggest_categorical("arch", ["M1", "M2", "M3", "M4"]),
         "n_layers": trial.suggest_int("n_layers", 2, 5),
         "width": trial.suggest_categorical("width", [32, 64, 128]),
         "activation": trial.suggest_categorical("activation", ["tanh", "silu"]),
         "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
         "dropout": trial.suggest_float("dropout", 0.0, 0.2),
         "c_t": trial.suggest_categorical("c_t", [3.0, 6.0, 10.0]),
         "n_basis": trial.suggest_categorical("n_basis", [3, 8]),
         "inv_set": trial.suggest_categorical("inv_set", ["four", "five", "six"])}
    return c


def make_study(strategy, seed):
    if strategy == "tpe":
        return optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    dirs = ["minimize"] * 4
    if strategy == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif strategy == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(population_size=40, seed=seed)
    elif strategy == "qnehvi":
        from optuna_integration import BoTorchSampler
        from optuna_integration.botorch.botorch import qnehvi_candidates_func
        sampler = BoTorchSampler(candidates_func=qnehvi_candidates_func, n_startup_trials=20,
                                 independent_sampler=optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed, scramble=True),
                                 seed=seed)
    else:
        raise ValueError(strategy)
    return optuna.create_study(directions=dirs, sampler=sampler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["random", "tpe", "nsga2", "qnehvi"])
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results/paper3/search"))
    a = ap.parse_args()
    job = f"{a.strategy}__arm{a.arm}__seed{a.seed}"
    d = Path(a.out) / job
    if (d / "status.json").exists():
        print(job, "already complete")
        return
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    lm = data.load("LM")
    train = {c: PL.prepare(lm[lm.case == c]) for c in search.CASES_LM}
    study = make_study(a.strategy, a.seed)
    log = (d / "evals.jsonl").open("a")
    t0 = time.time()

    def objective(trial):
        c = suggest(trial, a.arm)
        r = search.evaluate_candidate(c, train, seeds=(0,))
        f = search.objective_vector(r, REF, a.arm)
        rec = {"trial": trial.number, "elapsed_s": round(time.time() - t0, 1), **r, "objectives": f}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {job} trial {trial.number + 1}/{a.n_eval} "
              f"{'FAIL' if f is None else ' '.join(f'{k}={f[k]:.4g}' for k in OBJ)}", flush=True)
        if f is None:
            raise RuntimeError("non-finite fit")   # trial state FAIL: counted in the budget, infeasible
        for k, v in f.items():
            trial.set_user_attr(k, v)
        return f["f_err"] if a.strategy == "tpe" else tuple(f[k] for k in OBJ)

    study.optimize(objective, n_trials=a.n_eval, catch=(RuntimeError,))
    (d / "status.json").write_text(json.dumps({"job": job, "n_trials": len(study.trials),
                                               "n_failed": sum(t.state.name == "FAIL" for t in study.trials),
                                               "wall_seconds": round(time.time() - t0, 1),
                                               "optuna": optuna.__version__}, indent=1))


if __name__ == "__main__":
    main()
