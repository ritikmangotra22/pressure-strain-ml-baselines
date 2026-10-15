"""Ablation A2 (trace), evaluated post hoc on the saved E1 test predictions.

Variants (no retraining):
  raw          predictions as produced (unconstrained)
  reconstruct  Pi33 := -(Pi11 + Pi22)   [planned A2 form]
  project      orthogonal projection onto trace-free tensors: Pi_ii -= tr/3 (diagonal only)

Because the DNS target is exactly trace-free, the orthogonal projection can never increase the
Frobenius error at any point; `reconstruct` can. Deviation from plan: A2 was planned as retraining
on (Pi11, Pi22, Pi12) only. For ridge and LightGBM (one independent fit per component) the
post-hoc reconstruction is identical to that; for RF and MLP (joint multi-output fits) it is an
approximation, and this is recorded in the manuscript.

    python scripts/ablation_trace.py --results results/e1_F1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ps.metrics import e_pi, trace_residual  # noqa: E402

COMP = ["Pi11", "Pi22", "Pi33", "Pi12"]


def main(res):
    res = Path(res)
    tab = pd.read_csv(res / "processed_table.csv")
    rows = []
    for f in sorted((res / "predictions").glob("*.csv")):
        model, held, seed = f.stem.split("__")
        if model not in ("ridge", "rf", "lgbm", "mlp"):
            continue
        p = pd.read_csv(f).merge(tab[["case", "y_plus", "w"]], on=["case", "y_plus"], how="left", validate="1:1")
        if p["w"].isna().any():
            raise ValueError(f"weight join failed for {f.name}")
        y, w = p[COMP].to_numpy(), p["w"].to_numpy()
        yh = p[[c + "_hat" for c in COMP]].to_numpy()
        rec = yh.copy(); rec[:, 2] = -(yh[:, 0] + yh[:, 1])
        prj = yh.copy(); tr = yh[:, :3].sum(1); prj[:, :3] -= tr[:, None] / 3
        rows.append({"model": model, "held_out": held, "re_tau": int(held.split("_")[-1]), "seed": int(seed[4:]),
                     "E_Pi_raw": e_pi(yh, y, w), "E_Pi_reconstruct": e_pi(rec, y, w), "E_Pi_project": e_pi(prj, y, w),
                     "trace_mean_raw": trace_residual(yh, w)["trace_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(res / "ablation_A2_trace.csv", index=False)
    s = df.groupby("model")[["E_Pi_raw", "E_Pi_reconstruct", "E_Pi_project", "trace_mean_raw"]].mean()
    s["n_runs"] = df.groupby("model").size()
    md = "### Table 13b — Ablation A2 (post hoc, E1 test predictions averaged over folds and seeds)\n\n" + s.to_markdown(floatfmt=".4f")
    (res / "table13b_A2.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    main(ap.parse_args().results)
