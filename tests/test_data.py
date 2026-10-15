"""Tests on the real DNS files (skipped when the data are not present)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ps.features import FEATURE_SETS, TARGETS, build_table  # noqa: E402
from ps.io_lm import load_lm_case, validate_case  # noqa: E402

LM = ROOT / "data/raw/lee_moser_2015"
HJ = ROOT / "data/raw/hoyas_jimenez_2008"
need_lm = pytest.mark.skipif(not LM.exists(), reason="Lee & Moser data not present")
need_hj = pytest.mark.skipif(not HJ.exists(), reason="Hoyas & Jimenez data not present")


@need_lm
@pytest.mark.parametrize("lab", ["0180", "0550", "1000", "2000", "5200"])
def test_lm_case_validates_and_features_match_definitions(lab):
    c = load_lm_case(LM, lab)
    validate_case(c)
    t = build_table(c)
    i = np.searchsorted(c.y_plus, t["y_plus"].to_numpy())
    np.testing.assert_allclose(t["b11"], c.uu[i] / (2 * c.k[i]) - 1 / 3)
    np.testing.assert_allclose(t["b12"], c.uv[i] / (2 * c.k[i]))
    np.testing.assert_allclose(t["eta"], c.k[i] * np.abs(c.dUdy[i]) / c.eps[i])
    np.testing.assert_allclose(t["Pi22"], c.budgets["vv"]["pressure_strain"][i] / c.eps[i])
    np.testing.assert_allclose(t[["Pi11", "Pi22", "Pi33"]].sum(1), 0, atol=1e-12)   # DNS trace is round-off
    assert np.isclose(t["w"].sum(), 1.0)
    assert (t["y_plus"] > 1e-3).all()                      # no wall point (regression: Re_tau=180 bug)
    assert len(t) == len(c.y_plus) - 1                      # exactly one point (the wall) removed
    assert t["k"].min() > 1e-12
    assert set(FEATURE_SETS["F1"]).isdisjoint({"y_plus", "y_delta", "re_tau"})
    assert np.isfinite(t[FEATURE_SETS["F1"] + TARGETS].to_numpy()).all()


@need_hj
@pytest.mark.parametrize("re", [180, 550, 950, 2000])
def test_hj_case_validates(re):
    from ps.io_hj import load_hj_case, validate_hj
    c = load_hj_case(HJ, re)
    d = validate_hj(c)
    assert d["dns_trace_rel_max"] < 1e-3
    t = build_table(c)
    assert np.isfinite(t[FEATURE_SETS["F1"] + TARGETS].to_numpy()).all()
