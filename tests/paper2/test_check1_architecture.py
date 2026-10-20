"""Section H item 1 as automated tests (fast version of scripts/paper2/check1_architecture.py).

Real DNS states from the released Paper 1 tables; untrained networks; 20 random rotations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ps2 import data, models  # noqa: E402
from ps2 import tensors as T  # noqa: E402


@pytest.fixture(scope="module")
def states():
    df = pd.concat([data.load("LM"), data.load("HJ")], ignore_index=True)
    b, G = T.channel_tensors(df, "durbin", 6.0)
    re_t = torch.tensor(df["Re_t"].to_numpy(float), dtype=T.DT)
    return b, G, re_t


def _eq_err(m, b, G, re_t, Q):
    with torch.no_grad():
        P = m(b, G, re_t)
        Qn = Q.expand(len(b), 3, 3)
        Pq = m(T.rotate(Qn, b), T.rotate(Qn, G), re_t)
        return float((T.frob(Pq - T.rotate(Qn, P)) / T.frob(P).clamp(min=1e-300)).max())


TRANSFORMS = [("reflect_y", T.REFLECT_Y), ("rot_z_45", T.rot_about_z(45))] + \
             [(f"rand{i}", q) for i, q in enumerate(T.random_rotations(20, 7))]


@pytest.mark.parametrize("name", ["M2", "M4"])
@pytest.mark.parametrize("seed", [0, 1])
def test_basis_models_are_equivariant(states, name, seed):
    b, G, re_t = states
    m = models.build(name, b, G, re_t, seed=seed)
    assert max(_eq_err(m, b, G, re_t, Q) for _, Q in TRANSFORMS) <= 1e-10


@pytest.mark.parametrize("name", ["M1", "M3"])
def test_cartesian_models_are_not_equivariant(states, name):
    b, G, re_t = states
    m = models.build(name, b, G, re_t, seed=0)
    assert max(_eq_err(m, b, G, re_t, Q) for _, Q in TRANSFORMS) > 1e-3


@pytest.mark.parametrize("name,should_hold", [("M1", False), ("M2", False), ("M3", True), ("M4", True)])
def test_trace_and_isotropic_limit(states, name, should_hold):
    b, G, re_t = states
    m = models.build(name, b, G, re_t, seed=3, n_layers=4, activation="silu")
    with torch.no_grad():
        P = m(b, G, re_t)
        tr_rel = float((T.tr(P).abs() / T.frob(P)).max())
        z = torch.zeros((50, 3, 3), dtype=T.DT)
        iso = float(T.frob(m(z, z, torch.logspace(0, 4, 50, dtype=T.DT))).max())
    if should_hold:
        assert tr_rel <= 1e-12 and iso <= 1e-12
    else:
        assert tr_rel > 1e-3 and iso > 1e-3


@pytest.mark.parametrize("name", ["M1", "M2", "M3", "M4"])
def test_outputs_symmetric_and_float64(states, name):
    b, G, re_t = states
    with torch.no_grad():
        P = models.build(name, b, G, re_t, seed=0)(b, G, re_t)
    assert P.dtype == torch.float64
    assert float((P - P.transpose(1, 2)).abs().max()) <= 1e-14


def test_basis_elements_traceless_symmetric_equivariant(states):
    b, G, _ = states
    S, W = T.sym_skew(G)
    Q = T.random_rotations(1, 3)[0].expand(len(b), 3, 3)
    Sq, Wq = T.sym_skew(T.rotate(Q, G))
    for B, Bq in zip(T.basis_dev(b, S, W), T.basis_dev(T.rotate(Q, b), Sq, Wq)):
        assert float(T.tr(B).abs().max()) <= 1e-15
        assert float((B - B.transpose(1, 2)).abs().max()) <= 1e-15
        assert float((Bq - T.rotate(Q, B)).abs().max()) <= 1e-13


def test_invariants_unchanged_by_rotation_and_reflection(states):
    b, G, _ = states
    S, W = T.sym_skew(G)
    names = T.INVARIANT_SETS["E4+bW2"]
    I0 = T.invariants(b, S, W, names)
    for _, Q in TRANSFORMS:
        Qn = Q.expand(len(b), 3, 3)
        Sq, Wq = T.sym_skew(T.rotate(Qn, G))
        Iq = T.invariants(T.rotate(Qn, b), Sq, Wq, names)
        assert float((Iq - I0).abs().max()) <= 1e-12 * max(1.0, float(I0.abs().max()))


def test_channel_tensor_conventions(states):
    """G_12 = tau dU/dy >= 0, b symmetric and traceless, tau >= k/eps."""
    b, G, _ = states
    assert float(G[:, 0, 1].min()) >= 0 and float(G[:, 1, 0].abs().max()) == 0
    assert float(T.tr(b).abs().max()) < 1e-15
    assert np.all(T.time_scale_ratio([1e-3, 1.0, 36.0, 1e4], "durbin", 6.0) >= 1.0)
    assert np.allclose(T.time_scale_ratio([36.0, 144.0], "durbin", 6.0), [1.0, 1.0])
    assert np.isclose(T.time_scale_ratio([9.0], "durbin", 6.0)[0], 2.0)


def test_checks_detect_deliberately_broken_models(states):
    """Mutation tests: the property checks above must FAIL for broken variants."""
    b, G, re_t = states
    m4 = models.build("M4", b, G, re_t, seed=0)
    const = torch.zeros((3, 3), dtype=T.DT)
    const[0, 1] = const[1, 0] = 0.1          # a fixed, frame-attached tensor breaks equivariance
    broken_eq = lambda bb, GG, rr: m4(bb, GG, rr) + const
    assert max(_eq_err(broken_eq, b, G, re_t, Q) for _, Q in TRANSFORMS) > 1e-3
    broken_tr = lambda bb, GG, rr: m4(bb, GG, rr) + 0.05 * T.EYE   # trace-carrying, still equivariant
    with torch.no_grad():
        P = broken_tr(b, G, re_t)
        assert float((T.tr(P).abs() / T.frob(P)).max()) > 1e-3
    assert max(_eq_err(broken_tr, b, G, re_t, Q) for _, Q in TRANSFORMS) <= 1e-10
