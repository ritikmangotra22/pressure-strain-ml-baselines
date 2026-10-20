"""Section H item 2 as regression tests: the Section E.2-E.4 diagnostics on the real DNS tables."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ps2 import data, diagnostics as D  # noqa: E402
from ps2 import tensors as T  # noqa: E402


@pytest.fixture(scope="module")
def lm():
    return data.load("LM")


@pytest.fixture(scope="module")
def hj():
    return data.load("HJ")


def test_processed_tables_are_consistent(lm, hj):
    for df in (lm, hj):
        r = data.consistency_report(df)
        assert r["max|b11+b22+b33|"] < 1e-14
        assert r["max rel |eta - k|dU|/eps|"] < 1e-10 and r["max rel |Re_t - k^2/eps|"] < 1e-10
        assert r["max|P/eps + 2 b12 eta|"] < 1e-13
        assert r["min dU/dy"] >= 0 and r["max b12"] <= 0
    assert data.consistency_report(lm)["max|Pi_ii| (Pi/eps)"] < 1e-14


def test_basis_represents_lm_exactly_with_both_time_scales(lm):
    for sc in ("k/eps", "durbin"):
        r = D.adequacy(lm, sc, 6.0, bounds=())
        assert r["rank"].min() == 3
        assert r["res_free"].max() < 1e-11


def test_hj_residual_is_its_own_trace(hj):
    r = D.adequacy(hj, "durbin", 6.0, bounds=())
    assert np.abs(r["res_free"] - r["trace_rel"] / np.sqrt(3)).max() < 1e-12
    # the only rank-deficient points are the 4 centreline points (dU/dy = 0, Pi12 = 0)
    assert (r["rank"] < 3).sum() == 4 and set(hj.loc[r["rank"] < 3, "eta"]) == {0.0}


def test_kolmogorov_bound_removes_sublayer_ill_conditioning(lm):
    v = (lm.region == "viscous").to_numpy()
    g_k = D.adequacy(lm, "k/eps", 6.0, bounds=())["g_norm"][v]
    g_d = D.adequacy(lm, "durbin", 6.0, bounds=())["g_norm"][v]
    assert g_k.max() > 1e5 and (g_k > 100).sum() == 16
    assert g_d.max() < 3.0
    for ct in (3.0, 10.0):   # not an artefact of the provisional C_T = 6
        assert D.adequacy(lm, "durbin", ct, bounds=())["g_norm"][v].max() < 5.0


def test_four_invariants_cannot_tell_b11_from_b22(lm):
    """Swapping b11 and b22 equals reflecting x1<->x2 and reversing W: the four-invariant set is
    blind to it, tr(bS*W*) is not."""
    b, G = T.channel_tensors(lm, "durbin", 6.0)
    S, W = T.sym_skew(G)
    P = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=T.DT).expand(len(b), 3, 3)
    assert float((T.rotate(P, W) + W).abs().max()) == 0.0
    b_sw = T.rotate(P, b)
    I4, I4s = (T.invariants(x, S, W, T.INVARIANT_SETS["four"]) for x in (b, b_sw))
    assert float((I4 - I4s).abs().max()) <= 1e-15
    m = (lm.b11 - lm.b22).abs().to_numpy() > 1e-3
    bsw = T.invariants(b, S, W, ["tr_bSW"])[:, 0], T.invariants(b_sw, S, W, ["tr_bSW"])[:, 0]
    assert torch.allclose(bsw[0][m], -bsw[1][m])


def test_e4_set_has_a_realizable_twin_off_the_data_range_and_bw2_resolves_it():
    d0, b12 = 0.1, -0.05
    s0 = np.sqrt(d0 ** 2 + 4 * b12 ** 2)
    xa = np.array([(s0 - d0) / 2, (s0 + d0) / 2, b12, 5.0])
    xb = np.array([(-s0 - d0) / 2, (-s0 + d0) / 2, b12, 5.0])
    assert D.realizable(xa) and D.realizable(xb)
    e4 = T.INVARIANT_SETS["E4"]
    assert np.allclose(D.channel_state_invariants(xa, e4), D.channel_state_invariants(xb, e4), atol=1e-14)
    assert not np.isclose(D.channel_state_invariants(xa, ["tr_bW2"])[0], D.channel_state_invariants(xb, ["tr_bW2"])[0])


def test_dns_states_stay_far_from_the_e4_degenerate_set(lm, hj):
    for df in (lm, hj):
        s, d, b12 = (df.b11 + df.b22), (df.b22 - df.b11), df.b12
        q = (d ** 2 + 4 * b12 ** 2 - s ** 2) / (d ** 2 + 4 * b12 ** 2 + s ** 2)
        assert q.min() > 0.7
