"""Section H item 3 (R0) as regression tests, on one fast fold (Re_tau = 550, 267 epochs)."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
import check3_r0 as C  # noqa: E402
from ps.models import make_model  # noqa: E402
from ps2.train import ChannelInputNet, R0Model  # noqa: E402

HELD = "LM_Channel_0550"


@pytest.fixture(scope="module")
def fold():
    tabs = C.tables()
    tr = C.renorm(pd.concat([tabs[c] for c in C.CASES if c != HELD]))
    rec = pd.read_csv(C.P1 / "metrics_long.csv")
    ref = float(rec[(rec.model == "mlp") & (rec.held_out == HELD) & (rec.seed == 0)].E_Pi.iloc[0])
    return tr, tabs[HELD], C.recorded_params(HELD), ref


import os
import platform

# Paper 1's recorded runs were made on Linux x86-64. Bitwise (1e-12) agreement is expected only there; other
# processors round the last digits of float32 training differently (Apple M-series, macOS 26.2: 2.4e-8 on this fold).
RECORDING_PLATFORM = platform.system() == "Linux" and platform.machine() == "x86_64"
EXPECT_BITWISE = RECORDING_PLATFORM or os.environ.get("PS2_EXPECT_BITWISE") == "1"


def test_paper1_code_reproduces_recorded_run(fold):
    """Bitwise on the recording platform; elsewhere within 1e-6, the tolerance of the float32 test below."""
    tr, te, p, ref = fold
    m = make_model("mlp", C.F1, 0, p)
    m.fit(tr)
    assert abs(C.score(m, te) - ref) < (1e-12 if EXPECT_BITWISE else 1e-6)


def test_paper2_trainer_float32_matches_paper1(fold):
    tr, te, p, ref = fold
    m = R0Model(seed=0, dtype="float32", **p)
    m.fit(tr)
    assert abs(C.score(m, te) - ref) < 1e-6


def test_paper2_trainer_float64_cast_init_matches_paper1(fold):
    tr, te, p, ref = fold
    m = R0Model(seed=0, dtype="float64", init="float32_cast", **p)
    m.fit(tr)
    assert abs(C.score(m, te) - ref) < 1e-3


def test_native_float64_layers_draw_different_initial_weights():
    """The cause of the native-float64 deviation found in check 3."""
    torch.manual_seed(0)
    a = ChannelInputNet(5, dtype=torch.float32).net[0].weight.double()
    torch.manual_seed(0)
    b = ChannelInputNet(5, dtype=torch.float64, init="float32_cast").net[0].weight
    torch.manual_seed(0)
    c = ChannelInputNet(5, dtype=torch.float64).net[0].weight
    assert torch.equal(a, b)
    assert (a - c).abs().max() > 1e-2
