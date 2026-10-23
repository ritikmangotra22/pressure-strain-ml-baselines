"""Shared helpers for the Paper 3 post-search stage (refit, classical references, a posteriori
checks, confirmatory test). Reads training data (Lee & Moser) and search records only.

Selected models: one per completed search job (protocol_v1 selection rule as implemented in
analyse_search.select), plus M0. Refit (protocol_v1 selection_rule.refit): all five LM cases,
epochs = the selected trial's median inner best epoch, training seeds 0-9. M0 uses Paper 2's own
refit epochs (381, results/paper2/main/M4/run1/E2/selected.json).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper3"))
from ps2 import data, pipeline as PL  # noqa: E402
from ps3 import search  # noqa: E402

PROTO = json.loads((ROOT / "configs/paper3/protocol_v1.json").read_text())
CASES_LM = search.CASES_LM
REFIT_SEEDS = list(range(10))
SEARCH = ROOT / "results/paper3/search"
FINAL = ROOT / "results/paper3/final"
M0_EPOCHS = json.loads((ROOT / "results/paper2/main/M4/run1/E2/selected.json").read_text())["epochs"]
M0_CONFIG = PROTO["reference_M0"]["config"]
N_JOBS = sum(PROTO["jobs"][k]["total_jobs"] for k in ("primary", "conditioning", "ablations"))


def lm_split(c_t: float) -> PL.Split:
    """All five LM cases, inputs built with time-scale constant c_t, weights normalised per case."""
    lm = data.load("LM")
    return PL.concat([PL.prepare(lm[lm.case == c], "durbin", c_t) for c in CASES_LM])


def lm_case_splits(c_t: float) -> dict:
    lm = data.load("LM")
    return {c: PL.prepare(lm[lm.case == c], "durbin", c_t) for c in CASES_LM}


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


CONFIG_KEYS = ("arch", "n_layers", "width", "activation", "lr", "weight_decay", "dropout", "c_t", "n_basis", "inv_set")


def selected_models(allow_partial: bool = False) -> dict:
    """{model_id: {"config", "epochs", "source", ...}} for every completed job plus M0.
    Refuses to run on an incomplete search unless allow_partial (smoke tests only)."""
    import analyse_search as A
    out = {"M0": {"config": dict(M0_CONFIG), "epochs": int(M0_EPOCHS), "source": "Paper 2 run-1 M4 (E2)",
                  "strategy": "M0", "arm": "A", "seed": None}}
    done = 0
    for d in sorted(SEARCH.iterdir()):
        st = d / "status.json"
        if not st.exists() or json.loads(st.read_text()).get("counted") != A_N_EVAL():
            continue
        done += 1
        strategy, arm, seed = d.name.split("__")
        arm, seed = arm[3:], int(seed[4:])
        df = A.load(d.name)
        sel = A.select(df, arm, strategy)
        if sel is None:
            out[d.name] = {"config": None, "epochs": None, "source": "no feasible front member",
                           "strategy": strategy, "arm": arm, "seed": seed}
            continue
        c = {k: _clean(sel[k]) for k in CONFIG_KEYS if k in sel and not (isinstance(sel[k], float) and np.isnan(sel[k]))}
        out[d.name] = {"config": c, "epochs": int(round(float(sel["median_best_epoch"]))), "trial": int(sel["trial"]),
                       "inner_objectives": {k: _clean(sel[k]) for k in ("f_err", "f_gen", "f_phys", "f_cost", "f_cond") if k in sel},
                       "strategy": strategy, "arm": arm, "seed": seed, "source": f"search job {d.name}"}
    if done < N_JOBS and not allow_partial:
        raise SystemExit(f"only {done}/{N_JOBS} search jobs complete; refusing (use --allow-partial for a smoke test)")
    return out


def A_N_EVAL():
    return PROTO["strategies"]["budget_evaluations"]


def ckpt_path(root: Path, model_id: str, seed: int) -> Path:
    return root / "refits" / model_id / f"seed{seed}.pt"


def save_fitted(path: Path, fitted: PL.Fitted, config: dict, seed: int, epochs: int, info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fitted.model.state_dict(), "s": fitted.s, "config": config, "seed": seed,
                "epochs": epochs, "info": info}, path)


def load_fitted(path: Path, train: PL.Split | None = None) -> tuple[PL.Fitted, dict]:
    """Rebuild the network (scalers are refitted on the same LM data, hence identical) and load weights."""
    ck = torch.load(path, weights_only=False)
    c = ck["config"]
    train = train if train is not None else lm_split(c["c_t"])
    m = search.build(c, train, ck["seed"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return PL.Fitted(m, ck["s"]), ck


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
