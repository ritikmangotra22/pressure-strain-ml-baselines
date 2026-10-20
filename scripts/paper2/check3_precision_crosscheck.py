"""Check 3, independent cross-check of the float64 deviation.

Paper 1's own MLP code (ps.models.MLPModel), with its float32 tensors switched to float64 by
monkey-patching torch.float32 -> torch.float64 inside its fit/predict, is compared run by run
with the Paper 2 float64 stage (results/paper2/check3/r0_p2_f64.csv). Agreement means the
float64 deviation from Paper 1 is a property of the precision, not of the Paper 2 code path.

    python scripts/paper2/check3_precision_crosscheck.py
"""
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
import ps.models as P1  # noqa: E402
import check3_r0 as C  # noqa: E402


class MLP64(P1.MLPModel):
    def _net(self, p):
        return super()._net(p).double()

    def fit(self, train, val=None):
        orig = torch.float32
        torch.float32 = torch.float64          # Paper 1 builds its tensors with dtype=torch.float32
        try:
            return super().fit(train, val)
        finally:
            torch.float32 = orig

    def predict(self, df):
        orig = torch.float32
        torch.float32 = torch.float64
        try:
            return super().predict(df)
        finally:
            torch.float32 = orig


def main():
    tabs = C.tables()
    p2 = pd.read_csv(C.OUT / "r0_p2_f64.csv")
    rows = []
    for held in ["LM_Channel_0180", "LM_Channel_5200"]:
        p = C.recorded_params(held)
        tr = C.renorm(pd.concat([tabs[c] for c in C.CASES if c != held]))
        for s in C.SEEDS:
            m = MLP64(C.F1, seed=s, **p)
            m.fit(tr)
            e = C.score(m, tabs[held])
            ref = float(p2[(p2.held_out == held) & (p2.seed == s)].E_Pi.iloc[0])
            rows.append({"held_out": held, "seed": s, "E_Pi_paper1code_f64": e, "E_Pi_paper2code_f64": ref,
                         "abs_diff": abs(e - ref)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "r0_precision_crosscheck.csv", index=False)
    print(df.groupby("held_out")[["E_Pi_paper1code_f64", "E_Pi_paper2code_f64", "abs_diff"]].agg(["mean", "max"]).to_string())


if __name__ == "__main__":
    main()
