"""Merge E1 result parts and build manuscript tables (Tables 7-14) as markdown.

    python scripts/summarise.py --parts results/e1_F1_parts --out results/e1_F1
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ps.io_lm import load_lm_case  # noqa: E402

ORDER = ["ssg_published", "ssg_refit", "lrr_published", "lrr_refit", "lrr_gl_published", "lrr_gl_refit", "ridge", "rf", "lgbm", "mlp"]
NICE = {"ssg_published": "Rotta+SSG, published", "ssg_refit": "Rotta+SSG, refitted",
        "lrr_published": "Rotta+LRR-IP, published", "lrr_refit": "Rotta+LRR-IP, refitted", "lrr_gl_published": "Rotta+LRR-IP+GL wall, published", "lrr_gl_refit": "Rotta+LRR-IP+GL wall, refitted",
        "ridge": "Ridge", "rf": "Random forest", "lgbm": "Gradient-boosted trees (LightGBM)", "mlp": "MLP"}
RE = [180, 550, 1000, 2000, 5200]


def ms(s, d=3):
    m, sd = s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{m:.{d}f} ± {sd:.{d}f}" if len(s) > 1 else f"{m:.{d}f}"


def main(parts, out, data_dir):
    parts, out = Path(parts), Path(out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    frames, params = [], {}
    for p in sorted(parts.iterdir()):
        if not (p / "metrics_long.csv").exists():
            continue
        frames.append(pd.read_csv(p / "metrics_long.csv"))
        params.update(json.loads((p / "best_params.json").read_text()))
        for f in (p / "predictions").glob("*.csv"):
            shutil.copy(f, out / "predictions" / f.name)
        for f in ("validation.json", "checksums.json", "processed_table.csv"):
            if (p / f).exists():
                shutil.copy(p / f, out / f)
    df = pd.concat(frames).drop_duplicates(["model", "held_out", "seed"], keep="last")
    df.to_csv(out / "metrics_long.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(params, indent=2))
    models = [m for m in ORDER if m in set(df["model"])]
    md = []

    # Table 7: data quality
    rows = []
    for lab in ["0180", "0550", "1000", "2000", "5200"]:
        c = load_lm_case(Path(data_dir), lab)
        B = c.budgets
        tr = sum(B[k]["pressure_strain"] for k in ("uu", "vv", "ww"))
        nrm = np.sqrt(sum(B[k]["pressure_strain"] ** 2 for k in ("uu", "vv", "ww")) + 2 * B["uv"]["pressure_strain"] ** 2)
        bal = np.abs(B["k"]["balance"]) / c.eps
        rows.append({"Case": f"LM {int(lab)}", "Re_τ (actual)": f"{c.y_plus[-1] / c.y_delta[-1]:.1f}",
                     "Points (half channel, excl. wall)": len(c.y_plus) - 1,
                     "k-budget residual / ε, median": f"{np.median(bal):.1e}",
                     "k-budget residual / ε, max": f"{bal.max():.1e}",
                     "DNS trace residual, max (rel.)": f"{(np.abs(tr[1:]) / nrm[1:]).max():.0e}"})
    md += ["### Table 7 — DNS consistency\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 8: E_Pi per held-out case
    rows = []
    for m in models:
        d = df[df.model == m]
        r = {"Model": NICE[m]}
        for re in RE:
            r[str(re)] = ms(d[d.re_tau == re]["E_Pi"])
        r["Mean, interp."] = f"{d[d.regime == 'interp']['E_Pi'].mean():.3f}"
        r["Mean, extrap."] = f"{d[d.regime == 'extrap']['E_Pi'].mean():.3f}"
        rows.append(r)
    md += ["### Table 8 — E_Π on each held-out Re_τ (mean ± SD over seeds)\n",
           pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 8b: bootstrap CI
    rows = []
    for m in models:
        d = df[df.model == m]
        r = {"Model": NICE[m]}
        for re in RE:
            x = d[d.re_tau == re]
            r[str(re)] = f"[{x['E_Pi_ci_lo'].mean():.3f}, {x['E_Pi_ci_hi'].mean():.3f}]"
        rows.append(r)
    md += ["### Table 8b — 95 % block-bootstrap interval of E_Π (seed-averaged bounds)\n",
           pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 9: per component (average over folds and seeds)
    rows = []
    for m in models:
        d = df[df.model == m]
        r = {"Model": NICE[m]}
        for c in ("Pi11", "Pi22", "Pi33", "Pi12"):
            r[c + " RMSE / MAE / R²"] = f"{d[c + '_RMSE'].mean():.3f} / {d[c + '_MAE'].mean():.3f} / {d[c + '_R2'].mean():.2f}"
        r["median r (τ=0.05)"] = f"{d['r_median_f0.05'].mean():.3f}"
        r["90th pct r (τ=0.05)"] = f"{d['r_p90_f0.05'].mean():.3f}"
        rows.append(r)
    md += ["### Table 9 — Per-component errors (Π/ε units), averaged over E1 folds\n",
           pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 10: by region, split interp/extrap
    rows = []
    for m in models:
        for regime in ("interp", "extrap"):
            d = df[(df.model == m) & (df.regime == regime)]
            rows.append({"Model": NICE[m], "Regime": regime,
                         **{reg: f"{d['E_Pi_' + reg].mean():.3f}" for reg in ("viscous", "buffer", "log", "outer")}})
    md += ["### Table 10 — E_Π by wall region\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 13 (trace part)
    rows = [{"Model": NICE[m], "Mean t_n": f"{df[df.model == m]['trace_mean'].mean():.2e}",
             "Max t_n": f"{df[df.model == m]['trace_max'].max():.2e}"} for m in models]
    md += ["### Table 13 — Trace residual of predictions\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 14: cost
    rows = []
    for m in models:
        d = df[df.model == m]
        rows.append({"Model": NICE[m], "Tuning time per fold (s)": f"{d.groupby('held_out')['tune_time_s'].first().mean():.0f}",
                     "Final fit (s)": f"{d['fit_time_s'].mean():.2f}",
                     "Inference per point (µs)": f"{1e6 * d['infer_time_s_per_pt'].mean():.2f}"})
    md += ["### Table 14 — Computational cost (2-core cloud CPU)\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    # Table 11: hyperparameters / coefficients
    rows = []
    for m in models:
        r = {"Model": NICE[m]}
        for re in RE:
            v = params.get(f"{m}__LM_Channel_{re:04d}", {})
            if "fit_info" in v and "coef" in v.get("fit_info", {}):
                v = {k: round(x, 3) for k, x in v["fit_info"]["coef"].items()}
            else:
                v = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "fit_info"}
            r[str(re)] = json.dumps(v)[:200] if v else "—"
        rows.append(r)
    md += ["### Table 11 — Selected hyperparameters / refitted coefficients per outer fold\n",
           pd.DataFrame(rows).to_markdown(index=False), ""]

    (out / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/raw/lee_moser_2015")
    a = ap.parse_args()
    main(a.parts, a.out, a.data)
