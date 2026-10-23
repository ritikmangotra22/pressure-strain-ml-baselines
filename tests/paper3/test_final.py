"""Post-search stage: solver grid handling and confirmatory-test guards (no test data are read)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper3"))
from ps3 import rsm1d  # noqa: E402


def _grid(n, centre, on_centre):
    s = np.linspace(0, 1, n) if on_centre else np.linspace(0, 1, n + 1)[:-1]
    return centre * (1 - np.cos(np.pi * s / 2))           # clustered at the wall


def test_diffusion_exact_for_symmetric_quadratic_both_grid_types():
    c = 180.0
    for on_centre in (True, False):
        y = _grid(40, c, on_centre)
        rsm1d.CENTRE[0] = c
        De, Do = rsm1d._diffusion_matrix(y)
        q = (y[1:] - c) ** 2 - c ** 2                       # even about the centreline, zero at the wall
        assert np.allclose(De @ q, 2.0, atol=1e-9), on_centre


def test_odd_component_vanishes_on_centreline_node():
    c = 180.0
    y = _grid(40, c, True)
    rsm1d.CENTRE[0] = c
    _, Do = rsm1d._diffusion_matrix(y)
    row = Do.getrow(Do.shape[0] - 1).toarray().ravel()
    h = y[-1] - y[-2]
    assert row[-2] == 0.0 and np.isclose(row[-1], -2 / h ** 2)


def test_confirmatory_refuses_after_lock(tmp_path, monkeypatch):
    import confirmatory_test as CT
    monkeypatch.setattr(CT, "OUT", tmp_path)
    (tmp_path / "RUN_LOCK.json").write_text(json.dumps({"started": "x"}))
    bad = CT.preconditions(tmp_path)
    assert any("RUN_LOCK" in b for b in bad)


def test_confirmatory_requires_supervisor_and_tag(tmp_path, monkeypatch):
    import confirmatory_test as CT
    monkeypatch.setattr(CT, "OUT", tmp_path)
    monkeypatch.setitem(CT.F.PROTO["approvals"], "supervisor", {"status": "PENDING"})
    bad = CT.preconditions(tmp_path)
    assert any("supervisor" in b for b in bad)
