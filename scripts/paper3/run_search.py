"""Paper 3 search runner (protocol_v1, frozen 21 Oct 2026).

One job = one strategy x one arm x one optimizer seed, N_EVAL candidate evaluations, scored on the
five Lee & Moser cases by inner leave-one-Re_tau-out (1 training seed per candidate). No test data
are read. Every trial, including failed fits, is in the job's Optuna SQLite study and evals.jsonl.

Resumable (protocol_v1 "execution"): a session stops starting new trials after --max-minutes; the
next session reopens the study. Trials left RUNNING by a killed session are marked FAIL with
interrupted=true and do not count against the budget. The sampler seed of a resumed session is
seed * 1000 + trials already in the study (recorded in sessions.jsonl).

    python scripts/paper3/run_search.py --strategy nsga2 --arm A --seed 3001 [--max-minutes 100]
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, pipeline as PL  # noqa: E402
from ps3 import search  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
PROTO = json.loads((ROOT / "configs/paper3/protocol_v1.json").read_text())
REF = json.loads((ROOT / "results/paper3/reference_m0/m0_inner.json").read_text())
STRAT = PROTO["strategies"]
N_EVAL = STRAT["budget_evaluations"]
OBJ = ["f_err", "f_gen", "f_phys", "f_cost"]
SPACE = PROTO["search_space"]


def code_version():
    h = hashlib.sha256()
    for p in sorted((ROOT / "src/ps3").glob("*.py")) + [Path(__file__), ROOT / "configs/paper3/protocol_v1.json"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def suggest(trial, arm):
    """Static search space read from protocol_v1: n_basis and inv_set are always drawn and are
    ignored for M1/M3, so every sampler (BoTorch included) sees one fixed space."""
    s = SPACE
    return {"arch": "M4" if arm == "A" else trial.suggest_categorical("arch", s["arch"]["values"]),
            "n_layers": trial.suggest_int("n_layers", *s["n_layers"]["range"]),
            "width": trial.suggest_categorical("width", s["width"]["values"]),
            "activation": trial.suggest_categorical("activation", s["activation"]["values"]),
            "lr": trial.suggest_float("lr", *s["lr"]["range"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *s["weight_decay"]["range"], log=True),
            "dropout": trial.suggest_float("dropout", *s["dropout"]["range"]),
            "c_t": trial.suggest_categorical("c_t", s["c_t"]["values"]),
            "n_basis": trial.suggest_categorical("n_basis", s["n_basis"]["values"]),
            "inv_set": trial.suggest_categorical("inv_set", s["inv_set"]["values"])}


def make_sampler(strategy, seed):
    if strategy == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=STRAT["tpe"]["n_startup"])
    if strategy == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if strategy == "nsga2":
        return optuna.samplers.NSGAIISampler(population_size=STRAT["nsga2"]["population"], seed=seed)
    if strategy == "qnehvi":
        from optuna_integration import BoTorchSampler
        from optuna_integration.botorch.botorch import qnehvi_candidates_func
        return BoTorchSampler(candidates_func=qnehvi_candidates_func, n_startup_trials=STRAT["qnehvi"]["n_startup"],
                              independent_sampler=optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed, scramble=True),
                              seed=seed)
    raise ValueError(strategy)


def counted(study):
    """Trials that count against the budget: complete, plus failed fits that were not interrupted."""
    return [t for t in study.trials if t.state == TrialState.COMPLETE or
            (t.state == TrialState.FAIL and not t.user_attrs.get("interrupted"))]


def job_name(strategy, arm, seed):
    return f"{strategy}__arm{arm}__seed{seed}"


def run(strategy, arm, seed, n_eval=N_EVAL, out=ROOT / "results/paper3/search", max_minutes=100.0):
    job = job_name(strategy, arm, seed)
    d = Path(out) / job
    if (d / "status.json").exists():
        return "complete"
    d.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{d / 'study.db'}")
    dirs = ["minimize"] if strategy == "tpe" else ["minimize"] * 4
    names = optuna.get_all_study_names(storage)
    if job in names:
        study = optuna.load_study(study_name=job, storage=storage)
        for t in study.trials:                      # stale RUNNING trial from a killed session
            if t.state == TrialState.RUNNING:
                storage.set_trial_user_attr(t._trial_id, "interrupted", True)
                storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
    else:
        study = optuna.create_study(study_name=job, storage=storage, directions=dirs)
    n_prev = len(study.trials)
    eff_seed = seed * 1000 + n_prev if n_prev else seed
    study.sampler = make_sampler(strategy, eff_seed)
    remaining = n_eval - len(counted(study))
    with (d / "sessions.jsonl").open("a") as f:
        f.write(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials_in_study": n_prev,
                            "counted": n_eval - remaining, "sampler_seed": eff_seed,
                            "code_version": code_version()}) + "\n")
    if remaining <= 0:
        return finish(study, d, job, n_eval)
    lm = data.load("LM")
    train = {c: PL.prepare(lm[lm.case == c]) for c in search.CASES_LM}
    log = (d / "evals.jsonl").open("a")
    t0 = time.time()

    def objective(trial):
        c = suggest(trial, arm)
        ts = time.time()
        r = search.evaluate_candidate(c, train, seeds=(0,))
        f = search.objective_vector(r, REF, arm)
        rec = {"trial": trial.number, "propose_to_done_s": round(time.time() - ts, 1), **r, "objectives": f}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {job} {len(counted(study)) + 1}/{n_eval} "
              f"{'FAIL' if f is None else ' '.join(f'{k}={f[k]:.4g}' for k in OBJ)}", flush=True)
        if f is None:
            raise RuntimeError("non-finite fit")    # trial FAIL: counted in the budget, infeasible
        for k, v in f.items():
            trial.set_user_attr(k, v)
        return f["f_err"] if strategy == "tpe" else tuple(f[k] for k in OBJ)

    def stop_cb(study, trial):
        if len(counted(study)) >= n_eval or (time.time() - t0) / 60 > max_minutes:
            study.stop()

    study.optimize(objective, n_trials=remaining, catch=(RuntimeError,), callbacks=[stop_cb])
    if len(counted(study)) >= n_eval:
        return finish(study, d, job, n_eval)
    return "paused"


def finish(study, d, job, n_eval):
    ts = study.trials
    (d / "status.json").write_text(json.dumps({
        "job": job, "counted": len(counted(study)), "n_trials_total": len(ts),
        "n_failed_fits": sum(t.state == TrialState.FAIL and not t.user_attrs.get("interrupted") for t in ts),
        "n_interrupted": sum(bool(t.user_attrs.get("interrupted")) for t in ts),
        "first_start": min(t.datetime_start for t in ts).isoformat(),
        "last_complete": max(t.datetime_complete for t in ts if t.datetime_complete).isoformat(),
        "optuna": optuna.__version__, "code_version": code_version()}, indent=1))
    return "complete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["random", "tpe", "nsga2", "qnehvi"])
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--out", default=str(ROOT / "results/paper3/search"))
    a = ap.parse_args()
    print(run(a.strategy, a.arm, a.seed, a.n_eval, a.out, a.max_minutes))


if __name__ == "__main__":
    main()
