"""The search runner must use the frozen protocol_v1 values, and the search space must match it."""
import json
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper3"))
sys.path.insert(0, str(ROOT / "src"))
import run_search as RS  # noqa: E402

P = json.loads((ROOT / "configs/paper3/protocol_v1.json").read_text())


def test_protocol_frozen_and_approved():
    assert P["_status"].startswith("FROZEN")
    assert P["approvals"]["author"]["date"] == "2026-10-21"


def test_runner_reads_frozen_values():
    assert RS.N_EVAL == P["strategies"]["budget_evaluations"] == 200
    assert RS.make_sampler("nsga2", 1)._population_size == P["strategies"]["nsga2"]["population"]
    import run_queue as RQ
    assert len(RQ.jobs()) == P["jobs"]["primary"]["total_jobs"] == 35


def test_suggest_stays_inside_space():
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for arm in ("A", "B"):
        for _ in range(50):
            t = st.ask()
            c = RS.suggest(t, arm)
            assert P["search_space"]["n_layers"]["range"][0] <= c["n_layers"] <= P["search_space"]["n_layers"]["range"][1]
            assert P["search_space"]["lr"]["range"][0] <= c["lr"] <= P["search_space"]["lr"]["range"][1]
            assert c["c_t"] in P["search_space"]["c_t"]["values"]
            assert (c["arch"] == "M4") if arm == "A" else (c["arch"] in P["search_space"]["arch"]["values"])
            st.tell(t, 0.0)
