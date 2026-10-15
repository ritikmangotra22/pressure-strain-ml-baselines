"""Figures 2-6 from a merged E1 results folder.

    python scripts/make_figures.py --results results/e1_F1 --best <model>

Fig 2  DNS targets Pi_ij/eps vs y+ (all cases)
Fig 3  predicted vs DNS profiles for each held-out case (best ML model, refitted SSG, published SSG)
Fig 4  E_Pi vs held-out Re_tau, all models (mean +- SD over seeds; extrapolation folds shaded)
Fig 5  E_Pi by wall region, interpolation vs extrapolation
Fig 6  trace residual t_n vs y+ for the ML models (DNS trace is zero to round-off)
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COMP = ["Pi11", "Pi22", "Pi33", "Pi12"]
LABEL = {"Pi11": r"$\Pi_{11}/\varepsilon$", "Pi22": r"$\Pi_{22}/\varepsilon$",
         "Pi33": r"$\Pi_{33}/\varepsilon$", "Pi12": r"$\Pi_{12}/\varepsilon$"}
NICE = {"ssg_published": "SSG (published)", "ssg_refit": "SSG (refitted)", "lrr_published": "LRR-IP (published)",
        "lrr_refit": "LRR-IP (refitted)", "ridge": "Ridge", "rf": "Random forest", "lgbm": "LightGBM", "mlp": "MLP"}
ORDER = ["ssg_published", "ssg_refit", "lrr_published", "lrr_refit", "ridge", "rf", "lgbm", "mlp"]
COL = {"ssg_published": "#9aa0a6", "ssg_refit": "#5f6368", "lrr_published": "#c5a3a3", "lrr_refit": "#8d5b5b",
       "ridge": "#4c78a8", "rf": "#59a14f", "lgbm": "#f28e2b", "mlp": "#b07aa1"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main(res, best):
    res = Path(res)
    fig_dir = res / "figures"
    fig_dir.mkdir(exist_ok=True)
    if (res / "WARNING_SYNTHETIC_DATA.txt").exists():
        raise SystemExit("Refusing to make manuscript figures from synthetic data.")
    tab = pd.read_csv(res / "processed_table.csv")
    met = pd.read_csv(res / "metrics_long.csv")
    preds = res / "predictions"
    models = [m for m in ORDER if m in set(met["model"])]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=220)
        plt.close(fig)

    # Fig 2
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True)
    for (case, g), shade in zip(tab.groupby("re_tau"), np.linspace(0.85, 0.1, 5)):
        for ax, c in zip(axs, COMP):
            ax.semilogx(g["y_plus"], g[c], color=str(shade) if shade < 0.8 else "#c0c0c0", lw=1.2, label=f"{case}")
    for ax, c in zip(axs, COMP):
        ax.set_xlabel(r"$y^+$"); ax.set_ylabel(LABEL[c]); ax.axhline(0, color="k", lw=0.4)
    axs[0].legend(title=r"$Re_\tau$", fontsize=7, frameon=False)
    save(fig, "fig2_dns_targets.png")

    # Fig 3: one figure per held-out case, mean prediction over seeds
    for held in sorted(tab["case"].unique()):
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True)
        ref = None
        for model, ls in [(b, "-") for b in best.split(",")] + [("ssg_refit", "--"), ("ssg_published", ":")]:
            fs = sorted(preds.glob(f"{model}__{held}__seed*.csv"))
            if not fs:
                continue
            ps = [pd.read_csv(f) for f in fs]
            ref = ps[0]
            for ax, c in zip(axs, COMP):
                y = np.mean([p[c + "_hat"] for p in ps], axis=0)
                ax.semilogx(ref["y_plus"], y, ls, color=COL[model], lw=1.4, label=NICE[model])
                if len(ps) > 1:
                    s = np.std([p[c + "_hat"] for p in ps], axis=0)
                    ax.fill_between(ref["y_plus"], y - s, y + s, color=COL[model], alpha=0.2, lw=0)
        for ax, c in zip(axs, COMP):
            ax.semilogx(ref["y_plus"], ref[c], "o", ms=2, color="k", label="DNS")
            ax.set_xlabel(r"$y^+$"); ax.set_ylabel(LABEL[c]); ax.axhline(0, color="k", lw=0.4)
        axs[0].legend(fontsize=7, frameon=False)
        re = int(held.split("_")[-1])
        fig.suptitle(rf"Held-out case $Re_\tau$ = {re} ({'interpolation' if re in (550, 1000, 2000) else 'extrapolation'})", fontsize=10)
        save(fig, f"fig3_profiles_{held}.png")

    # Fig 4
    fig, ax = plt.subplots(figsize=(6.2, 4))
    g = met.groupby(["model", "re_tau"])["E_Pi"].agg(["mean", "std"]).reset_index()
    for m in models:
        d = g[g.model == m]
        ax.errorbar(d["re_tau"], d["mean"], yerr=d["std"].fillna(0), marker="o", ms=4, capsize=2,
                    color=COL[m], ls="--" if "published" in m or "refit" in m else "-", label=NICE[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([180, 550, 1000, 2000, 5200]); ax.set_xticklabels(["180", "550", "1000", "2000", "5200"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axvspan(130, 300, color="0.93", zorder=0); ax.axvspan(3500, 7500, color="0.93", zorder=0)
    ax.text(180, ax.get_ylim()[0] * 1.05, "extrap.", ha="center", fontsize=7, color="0.4")
    ax.text(5200, ax.get_ylim()[0] * 1.05, "extrap.", ha="center", fontsize=7, color="0.4")
    ax.set_xlabel(r"held-out $Re_\tau$"); ax.set_ylabel(r"$E_\Pi$ (test)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    save(fig, "fig4_E_Pi_vs_Re.png")

    # Fig 5
    regs = ["viscous", "buffer", "log", "outer"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, regime in zip(axs, ("interp", "extrap")):
        r = met[met.regime == regime].groupby("model")[[f"E_Pi_{x}" for x in regs]].mean().reindex(models)
        x = np.arange(len(regs)); w = 0.8 / len(models)
        for i, m in enumerate(models):
            ax.bar(x + i * w, r.loc[m].values, w, color=COL[m], label=NICE[m])
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(regs); ax.set_yscale("log")
        ax.set_title("Interpolation folds (550, 1000, 2000)" if regime == "interp" else "Extrapolation folds (180, 5200)")
    axs[0].set_ylabel(r"$E_\Pi$ (test)"); axs[1].legend(fontsize=7, frameon=False, ncol=2)
    save(fig, "fig5_E_Pi_by_region.png")

    # Fig 6: trace residual of ML predictions vs y+
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m in [m for m in models if m in ("ridge", "rf", "lgbm", "mlp")]:
        for f in sorted(preds.glob(f"{m}__*__seed0.csv")):
            p = pd.read_csv(f)
            yh = p[[c + "_hat" for c in COMP]].to_numpy()
            t = np.abs(yh[:, :3].sum(1)) / np.sqrt((yh ** 2 * [1, 1, 1, 2]).sum(1))
            t = np.maximum(t, 1e-17)   # display floor: exact zeros cannot be drawn on a log axis
            ax.loglog(p["y_plus"], t, color=COL[m], lw=0.8, alpha=0.7,
                      label=NICE[m] if "0550" in f.name else None)
    ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$t_n=|\hat\Pi_{ii}|/\|\hat\Pi\|_F$ (test, seed 0; floor $10^{-17}$)")
    ax.set_title("DNS trace is zero to round-off (~1e-15)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig6_trace_residual.png")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--best", required=True,
                    help="comma-separated ML models to overlay in Fig 3, ranked by E1 INNER-validation error only")
    a = ap.parse_args()
    main(a.results, a.best)
