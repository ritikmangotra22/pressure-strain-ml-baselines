"""Confirmatory-statistics code on synthetic predictions (no real test data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper3"))
import analyse_confirmatory as AC  # noqa: E402

Y = AC.Y


def synthetic(err_by_cell, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for case in AC.MKM:
        P = rng.normal(size=(n, 4))
        w = np.full(n, 1 / n)
        for cell, rel in err_by_cell.items():
            models = ["M0"] if cell == "M0" else [f"{cell.split('/')[0]}__arm{cell.split('/')[1]}__seed{s}" for s in AC.OPT_SEEDS]
            for m in models:
                for s in AC.REFIT_SEEDS:
                    H = P + rel * rng.normal(size=P.shape) * np.abs(P).mean()
                    d = pd.DataFrame(P, columns=Y)
                    for j, c in enumerate(Y):
                        d[c + "_hat"] = H[:, j]
                    d["w"], d["case"], d["model"], d["seed"] = w, case, m, s
                    rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_clear_difference_is_resolved_and_signed():
    pred = synthetic({"tpe/A": 0.05, "nsga2/A": 0.30, "M0": 0.10})
    comps = [c for c in AC.COMPARISONS if c[0] in ("nsga2_A - tpe_A", "tpe_A - M0")]
    for case in AC.MKM:
        cd = AC.CaseData(pred, case)
        point, rows = AC.analyse_case(cd, comps, np.random.default_rng(1), n_boot=200)
        r = {x["comparison"]: x for x in rows}
        assert r["nsga2_A - tpe_A"]["diff"] > 0 and r["nsga2_A - tpe_A"]["ci_lo"] > 0
        assert r["tpe_A - M0"]["diff"] < 0 and r["tpe_A - M0"]["ci_hi"] < 0


def test_point_estimate_equals_mean_e_pi():
    pred = synthetic({"tpe/A": 0.1})
    case = AC.MKM[0]
    cd = AC.CaseData(pred, case)
    allp = np.arange(cd.n)
    est = cd.cell_error("tpe/A", list(range(5)), list(range(10)), allp)
    d = pred[pred.case == case]
    direct = []
    for _, g in d.groupby(["model", "seed"]):
        P, H = g[Y].to_numpy(), g[[c + "_hat" for c in Y]].to_numpy()
        direct.append(np.sqrt((((H - P) ** 2) * AC.MULT).sum() / ((P ** 2) * AC.MULT).sum()))
    assert np.isclose(est, np.mean(direct))
