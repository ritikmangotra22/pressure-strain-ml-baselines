"""Merge E2 parts and build Table 12 (cross-database test on Hoyas & Jimenez).

    python scripts/summarise_e2.py --parts results/e2_parts --out results/e2
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ORDER = ["ssg_published", "ssg_refit", "lrr_published", "lrr_refit", "lrr_gl_published", "lrr_gl_refit", "ridge", "rf", "lgbm", "mlp"]
NICE = {"ssg_published": "Rotta+SSG, published", "ssg_refit": "Rotta+SSG, refitted",
        "lrr_published": "Rotta+LRR-IP, published", "lrr_refit": "Rotta+LRR-IP, refitted", "lrr_gl_published": "Rotta+LRR-IP+GL wall, published", "lrr_gl_refit": "Rotta+LRR-IP+GL wall, refitted",
        "ridge": "Ridge", "rf": "Random forest", "lgbm": "Gradient-boosted trees (LightGBM)", "mlp": "MLP"}
CASES = ["HJ_Channel_0180", "HJ_Channel_0550", "HJ_Channel_0950", "HJ_Channel_2000"]
HEAD = {"HJ_Channel_0180": "HJ 186", "HJ_Channel_0550": "HJ 547", "HJ_Channel_0950": "HJ 934", "HJ_Channel_2000": "HJ 2003"}


def ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def main(parts, out):
    parts, out = Path(parts), Path(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    frames, params = [], {}
    for p in sorted(parts.iterdir()):
        if not (p / "metrics_long.csv").exists():
            continue
        frames.append(pd.read_csv(p / "metrics_long.csv"))
        params.update(json.loads((p / "params.json").read_text()))
        for f in (p / "predictions").glob("*.csv"):
            shutil.copy(f, out / "predictions" / f.name)
        for f in ("validation_hj.json", "processed_table_hj.csv"):
            if (p / f).exists():
                shutil.copy(p / f, out / f)
    df = pd.concat(frames)
    df.to_csv(out / "metrics_long.csv", index=False)
    (out / "params.json").write_text(json.dumps(params, indent=2, default=str))
    models = [m for m in ORDER if m in set(df.model)]
    rows, rows_ci, rows_reg = [], [], []
    for m in models:
        d = df[df.model == m]
        rows.append({"Model": NICE[m], **{HEAD[c]: ms(d[d.test_case == c]["E_Pi"]) for c in CASES}})
        rows_ci.append({"Model": NICE[m], **{HEAD[c]: f"[{d[d.test_case == c]['E_Pi_ci_lo'].mean():.3f}, "
                                                      f"{d[d.test_case == c]['E_Pi_ci_hi'].mean():.3f}]" for c in CASES}})
        rows_reg.append({"Model": NICE[m], **{r: f"{d['E_Pi_' + r].mean():.3f}" for r in ("viscous", "buffer", "log", "outer")},
                         "Mean t_n": f"{d['trace_mean'].mean():.1e}"})
    md = ["### Table 12 — E2: E_Π on Hoyas & Jiménez (trained on all five Lee & Moser cases; mean ± SD over seeds)\n",
          pd.DataFrame(rows).to_markdown(index=False), "",
          "### Table 12b — E2: 95 % block-bootstrap interval of E_Π (seed-averaged bounds)\n",
          pd.DataFrame(rows_ci).to_markdown(index=False), "",
          "### Table 12c — E2: E_Π by wall region and trace residual (averaged over the four HJ cases)\n",
          pd.DataFrame(rows_reg).to_markdown(index=False), ""]
    (out / "table12_E2.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.parts, a.out)
