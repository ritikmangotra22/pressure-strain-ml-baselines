"""Validation of protocol v1 (configs/paper2/protocol_v1.json) before any M1-M4 training.

Covers design decisions 1, 2 and 4: the six-invariant set, float32-built-and-cast initialisation,
float32 forward passes for training and float64 evaluation for the exact property tests.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
from ps2 import data, models  # noqa: E402
from ps2 import tensors as T  # noqa: E402

PROTO = json.loads((ROOT / "configs/paper2/protocol_v1.json").read_text())
TOL = PROTO["tolerances_exact"]


@pytest.fixture(scope="module")
def states():
    df = pd.concat([data.load("LM"), data.load("HJ")], ignore_index=True)
    b, G = T.channel_tensors(df, PROTO["inputs"]["time_scale"]["scheme"], PROTO["inputs"]["time_scale"]["C_T"])
    return df, b, G, torch.tensor(df.Re_t.to_numpy(float), dtype=T.DT)


def test_protocol_is_internally_consistent():
    assert PROTO["data"]["train_and_E1"]["sha256"] == data.SHA256["LM"]
    assert PROTO["data"]["E2_external"]["sha256"] == data.SHA256["HJ"]
    assert set(PROTO["folds"]["interpolation"]) | set(PROTO["folds"]["extrapolation"]) == set(PROTO["folds"]["E1"])
    t = PROTO["tuning"]
    assert t["same_budget_and_space_for_M1_to_M4"] and t["independent_runs_per_model_and_fold"] == len(t["tuner_sampler_seeds"])
    assert len(set(t["tuner_sampler_seeds"])) == len(t["tuner_sampler_seeds"])
    assert T.INVARIANT_SETS["P1"] == T.INVARIANT_SETS["E4+bW2"]


def test_six_invariants_reconstruct_the_channel_state_exactly(states):
    """Decision 1: for G12 != 0, tr bS*, tr bS*W*, tr bW*^2 and G12 give b11, b22, b12 exactly."""
    df, b, G, _ = states
    S, W = T.sym_skew(G)
    I = T.invariants(b, S, W, T.INVARIANT_SETS["P1"]).numpy()
    g = G[:, 0, 1].numpy()
    m = g > 0
    b12 = I[m, 3] / g[m]
    d = 4 * I[m, 4] / g[m] ** 2
    s = -4 * I[m, 5] / g[m] ** 2
    rec = np.c_[(s - d) / 2, (s + d) / 2, b12]
    assert np.abs(rec - df.loc[m, ["b11", "b22", "b12"]].to_numpy()).max() < 1e-12
    assert (~m).sum() == 4        # the HJ centreline points (dU/dy = 0)


@pytest.mark.parametrize("name", ["M1", "M2", "M3", "M4"])
def test_cast_initialisation_is_reproducible_and_precision_independent(states, name):
    """Decision 2: the same seed gives the same initial weights on every build and in both precisions."""
    _, b, G, r = states
    kw = dict(seed=5, inv_set="P1", init="float32_cast", n_layers=3, width=32, activation="silu")
    a, a2 = models.build(name, b, G, r, **kw), models.build(name, b, G, r, **kw)
    a32 = models.build(name, b, G, r, dtype=torch.float32, **kw)
    for p, q, p32 in zip(a.parameters(), a2.parameters(), a32.parameters()):
        assert torch.equal(p, q)
        assert torch.equal(p, p32.double())
    nat = models.build(name, b, G, r, seed=5, inv_set="P1", init="native", n_layers=3, width=32, activation="silu")
    assert max(float((p - q).detach().abs().max()) for p, q in zip(a.parameters(), nat.parameters())) > 1e-2


@pytest.mark.parametrize("name", ["M2", "M4"])
def test_float32_model_meets_exact_tolerances_only_in_float64(states, name):
    """Decision 4: float32 weights, cast to float64 for evaluation, satisfy the exact tolerances."""
    _, b, G, r = states
    m32 = models.build(name, b, G, r, seed=1, inv_set="P1", init="float32_cast", dtype=torch.float32)
    m64 = models.build(name, b, G, r, seed=1, inv_set="P1", init="float32_cast", dtype=torch.float32).double()
    Q = T.random_rotations(1, 9)[0].expand(len(b), 3, 3)
    with torch.no_grad():
        P = m64(b, G, r)
        e64 = float((T.frob(m64(T.rotate(Q, b), T.rotate(Q, G), r) - T.rotate(Q, P)) / T.frob(P)).max())
        P32 = m32(b.float(), G.float(), r.float())
        Qf = Q.float()
        e32 = float((T.frob(m32(T.rotate(Qf, b.float()), T.rotate(Qf, G.float()), r.float()) - T.rotate(Qf, P32)) / T.frob(P32)).max())
        cons = float((T.frob(P32.double() - P) / T.frob(P)).max())
    assert e64 <= TOL["equivariance_rel"]
    assert e32 > TOL["equivariance_rel"]          # float32 cannot resolve the exact tolerance
    assert cons <= 1e-4                            # protocol float32/float64 consistency check


def test_float32_forward_and_r0_training_are_deterministic():
    """Same seed, same machine: bitwise-identical float32 training (R0 network, fast fold)."""
    import check3_r0 as C
    from ps2.train import R0Model
    tabs = C.tables()
    held = "LM_Channel_0550"
    tr = C.renorm(pd.concat([tabs[c] for c in C.CASES if c != held]))
    p = C.recorded_params(held)
    out = []
    for _ in range(2):
        m = R0Model(seed=3, dtype="float32", init="float32_cast", **p)
        m.fit(tr)
        out.append(m.predict(tabs[held]))
    assert np.array_equal(out[0], out[1])
