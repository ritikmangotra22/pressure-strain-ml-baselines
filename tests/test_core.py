"""Unit tests: tensor algebra of the classical closures, metrics, leakage guards."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ps.classical import (GL_CMU, GL_KAPPA, LRR_IP_GL_PUBLISHED, LRR_IP_PUBLISHED, SSG_PUBLISHED,  # noqa: E402
                          lrr_ip_basis, lrr_ip_gl_basis, ssg_basis)
from ps.features import check_no_leakage  # noqa: E402
from ps.metrics import e_pi, frob2  # noqa: E402


def random_channel_states(n=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        A = rng.standard_normal((2, 2))
        R2 = A @ A.T + 0.1 * np.eye(2)          # SPD 2x2 block (x, y)
        r33 = rng.uniform(0.1, 2.0)
        R = np.zeros((3, 3)); R[:2, :2] = R2; R[2, 2] = r33
        k = 0.5 * np.trace(R)
        eps = rng.uniform(0.1, 2.0)
        dU = rng.uniform(0.1, 5.0)
        rows.append((R, k, eps, dU))
    return rows


def general_ssg(R, k, eps, dU, C):
    G = np.zeros((3, 3)); G[0, 1] = dU               # dU_i/dx_j, only dU1/dx2
    S = 0.5 * (G + G.T); W = 0.5 * (G - G.T)
    b = R / (2 * k) - np.eye(3) / 3
    II = np.sum(b * b)
    P = -np.sum(R * G)                                 # production of k = -R_ij dU_i/dx_j
    I = np.eye(3)
    Pi = (-(C["C1"] * eps + C["C1s"] * P) * b + C["C2"] * eps * (b @ b - II / 3 * I)
          + (C["C3"] - C["C3s"] * np.sqrt(II)) * k * S
          + C["C4"] * k * (b @ S.T + S @ b.T - 2 / 3 * np.sum(b * S) * I)
          + C["C5"] * k * (b @ W.T + W @ b.T))
    return Pi / eps


def general_lrr_ip(R, k, eps, dU, C):
    G = np.zeros((3, 3)); G[0, 1] = dU
    Pij = -(R @ G.T + G @ R)                            # P_ij = -R_ik dU_j/dx_k - R_jk dU_i/dx_k
    Pi = -C["CR"] * eps * (R / k - 2 / 3 * np.eye(3)) - C["CP"] * (Pij - np.trace(Pij) / 3 * np.eye(3))
    return Pi / eps


def as_df(states):
    rows = []
    for R, k, eps, dU in states:
        b = R / (2 * k) - np.eye(3) / 3
        rows.append({"b11": b[0, 0], "b22": b[1, 1], "b33": b[2, 2], "b12": b[0, 1],
                     "eta": k * dU / eps, "P_over_eps": -R[0, 1] * dU / eps})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("which", ["ssg", "lrr"])
def test_channel_basis_matches_general_tensor_form(which):
    states = random_channel_states()
    df = as_df(states)
    if which == "ssg":
        C = SSG_PUBLISHED
        pred = ssg_basis(df) @ np.array(list(C.values()))
        ref = [general_ssg(*s, C) for s in states]
    else:
        C = LRR_IP_PUBLISHED
        pred = lrr_ip_basis(df) @ np.array(list(C.values()))
        ref = [general_lrr_ip(*s, C) for s in states]
    ref = np.array([[r[0, 0], r[1, 1], r[2, 2], r[0, 1]] for r in ref])
    np.testing.assert_allclose(pred, ref, rtol=1e-10, atol=1e-10)


def openfoam_ssg(R, k, eps, dU, C):
    """SSG exactly as written in OpenFOAM v12 SSG.C, using OpenFOAM's convention
    gradU_ij = dU_j/dx_i (transpose of the Jacobian):
        b = dev(R)/(2k); S = symm(gradU); Omega = skew(gradU); P = -twoSymm(R & gradU); G = 0.5 tr(P)
        Pi = -(C1 eps + C1s G) b + C2 eps dev(b.b) + k[(C3 - C3s|b|) dev(S) + C4 dev(twoSymm(b.S)) + C5 twoSymm(b.Omega)]
    """
    gradU = np.zeros((3, 3)); gradU[1, 0] = dU      # d U_x / d y in OpenFOAM index order
    I = np.eye(3)
    dev = lambda A: A - np.trace(A) / 3 * I
    two_symm = lambda A: A + A.T
    b = dev(R) / (2 * k)
    S = 0.5 * (gradU + gradU.T); Om = 0.5 * (gradU - gradU.T)
    P = -two_symm(R @ gradU); G = 0.5 * np.trace(P)
    Pi = (-(C["C1"] * eps + C["C1s"] * G) * b + C["C2"] * eps * dev(b @ b)
          + k * ((C["C3"] - C["C3s"] * np.sqrt(np.sum(b * b))) * dev(S) + C["C4"] * dev(two_symm(b @ S))
                 + C["C5"] * two_symm(b @ Om)))
    return Pi / eps


def test_ssg_matches_openfoam_convention_including_rotation_sign():
    states = random_channel_states(seed=3)
    df = as_df(states)
    pred = ssg_basis(df) @ np.array(list(SSG_PUBLISHED.values()))
    ref = np.array([[r[0, 0], r[1, 1], r[2, 2], r[0, 1]] for r in (openfoam_ssg(*s, SSG_PUBLISHED) for s in states)])
    np.testing.assert_allclose(pred, ref, rtol=1e-10, atol=1e-10)
    # Flipping the rotation-term sign must break agreement (the test is sensitive to it).
    C = dict(SSG_PUBLISHED); C["C5"] = -C["C5"]
    flipped = ssg_basis(df) @ np.array(list(C.values()))
    assert np.max(np.abs(flipped - ref)) > 1e-3


def test_published_coefficients():
    assert SSG_PUBLISHED == {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}
    assert LRR_IP_PUBLISHED == {"CR": 1.8, "CP": 0.6}


def test_classical_models_are_trace_free():
    df = as_df(random_channel_states())
    for B, C in ((ssg_basis(df), SSG_PUBLISHED), (lrr_ip_basis(df), LRR_IP_PUBLISHED)):
        p = B @ np.array(list(C.values()))
        np.testing.assert_allclose(p[:, 0] + p[:, 1] + p[:, 2], 0, atol=1e-10)


def test_frobenius_counts_offdiagonal_twice():
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert frob2(a)[0] == 1 + 4 + 9 + 2 * 16


def test_e_pi_zero_for_perfect_prediction():
    y = np.random.default_rng(1).standard_normal((20, 4))
    assert e_pi(y, y, np.ones(20)) == 0.0


def test_leakage_guard():
    with pytest.raises(ValueError):
        check_no_leakage(["b11", "y_plus"])
    with pytest.raises(ValueError):
        check_no_leakage(["b11", "production"])
    check_no_leakage(["b11", "b22", "b12", "eta", "Re_t"])


def test_lrr_ip_equals_ssg_form_with_mapped_coefficients():
    """Independent check of the SSG rotation-term sign: LRR-IP, built directly from P_ij, must equal
    the SSG tensor form with C1 = 2*C_R, C3 = 4/3*C_P, C4 = C5 = 2*C_P and the other constants zero
    (ANSYS CFX lists LRR-IP as Cs1 = 1.8, Cr2 = 0.8, Cr4 = Cr5 = 0.6 in the a = 2b normalisation)."""
    df = as_df(random_channel_states(seed=7))
    lrr = lrr_ip_basis(df) @ np.array([1.8, 0.6])
    ssg = ssg_basis(df) @ np.array([3.6, 0, 0, 0.8, 0, 1.2, 1.2])
    np.testing.assert_allclose(lrr, ssg, rtol=1e-10, atol=1e-10)
    flipped = ssg_basis(df) @ np.array([3.6, 0, 0, 0.8, 0, 1.2, -1.2])
    assert np.max(np.abs(flipped - lrr)) > 1e-3


def general_gl_wall(R, k, eps, dU, y, C):
    """Gibson & Launder (1978) wall reflection in general tensor form, wall normal n = e_y."""
    G = np.zeros((3, 3)); G[0, 1] = dU
    Pij = -(R @ G.T + G @ R)
    I = np.eye(3); n = np.array([0.0, 1.0, 0.0])
    phi2 = -C["CP"] * (Pij - np.trace(Pij) / 3 * I)
    X = C["Cref1"] * eps / k * R + (C["D_Cref2xCP"] / C["CP"]) * phi2
    f = GL_CMU ** 0.75 * k ** 1.5 / (GL_KAPPA * eps * y)
    Xn = X @ n
    W = f * ((n @ X @ n) * I - 1.5 * np.outer(Xn, n) - 1.5 * np.outer(n, Xn))
    return general_lrr_ip(R, k, eps, dU, C) + W / eps


def test_gl_wall_reflection_matches_general_tensor_form_and_is_trace_free():
    states = random_channel_states(seed=11)
    rng = np.random.default_rng(5)
    ys = rng.uniform(0.5, 50.0, len(states))
    df = as_df(states)
    df["k"] = [s[1] for s in states]; df["eps"] = [s[2] for s in states]; df["y_plus"] = ys
    C = LRR_IP_GL_PUBLISHED
    pred = lrr_ip_gl_basis(df) @ np.array(list(C.values()))
    ref = np.array([[r[0, 0], r[1, 1], r[2, 2], r[0, 1]]
                    for r in (general_gl_wall(*s, y, C) for s, y in zip(states, ys))])
    np.testing.assert_allclose(pred, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pred[:, 0] + pred[:, 1] + pred[:, 2], 0, atol=1e-10)
    assert LRR_IP_GL_PUBLISHED == {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.3 * 0.6}


def test_tuning_never_touches_the_held_out_case(monkeypatch):
    """Leakage test for the nested protocol: during inner tuning, no model is fitted on, scaled with,
    or scored on the held-out Re_tau. Uses a spy model that records every table it sees."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_e1
    from ps.features import TARGETS
    seen = []

    class Spy:
        def __init__(self, *a, **k):
            pass

        def fit(self, train, val=None):
            seen.extend(train["case"].unique())
            if val is not None:
                seen.extend(val["case"].unique())
            return {}

        def predict(self, df):
            seen.extend(df["case"].unique())
            return np.zeros((len(df), 4))

    rng = np.random.default_rng(0)
    tables = {}
    for c in ("A", "B", "C", "HELD"):
        t = pd.DataFrame(rng.standard_normal((20, 4)), columns=TARGETS)
        t["case"] = c; t["w"] = 1 / 20; t["w_eval"] = 1 / 20; t["x"] = rng.standard_normal(20)
        tables[c] = t
    monkeypatch.setattr(run_e1, "make_model", lambda *a, **k: Spy())
    monkeypatch.setitem(run_e1.SPACES, "spy", lambda trial: {"a": trial.suggest_float("a", 0, 1)})
    run_e1.tune("spy", ["A", "B", "C"], tables, ["x"], n_trials=3, seed=0)
    assert seen and "HELD" not in seen
