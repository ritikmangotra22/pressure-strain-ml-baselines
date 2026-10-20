"""Regenerate every Paper 2 manuscript table and figure from archived result files only.

Inputs (read-only): results/paper2/analysis/*.csv, results/paper2/main/*/run*/*/metrics.csv,
results/paper2/ablations/*/*/*/metrics.csv, results/paper2/audit/supplementary/*.csv.
No model is trained or evaluated here.

    python3 scripts/paper2/make_tables_figures.py --root <archive root> --out <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--supp", default=None, help="audit supplementary dir (default <root>/results/paper2/audit/supplementary)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = Path(a.root)
RES = ROOT / "results/paper2"
AN = RES / "analysis"
SUPP = Path(a.supp) if a.supp else RES / "audit/supplementary"
OUT = Path(a.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

MODELS = ["M1", "M2", "M3", "M4"]
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100"}   # validated categorical slots 1-4
MARK = {"M1": "o", "M2": "s", "M3": "^", "M4": "D"}                           # secondary encoding
CASES = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200",
         "HJ_Channel_0180", "HJ_Channel_0550", "HJ_Channel_0950", "HJ_Channel_2000"]
LAB = {c: ("LM " if c.startswith("LM") else "HJ ") + str(int(c[-4:])) for c in CASES}
REG = {"LM_Channel_0180": "extrap", "LM_Channel_5200": "extrap", "LM_Channel_0550": "interp",
       "LM_Channel_1000": "interp", "LM_Channel_2000": "interp"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dd", "grid.linewidth": 0.6, "axes.edgecolor": "#8a897f", "axes.axisbelow": True,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
INK = "#2b2a26"


def md(df, path, floatfmt=None):
    df.to_csv(path.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def f3(x):
    return f"{x:.3f}"


def save(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
mbf = pd.read_csv(AN / "model_by_fold.csv")
eff = pd.read_csv(AN / "effects.csv")
abl = pd.read_csv(AN / "ablations.csv")
man = pd.read_csv(AN / "run_manifest.csv")
met = []
for f in RES.glob("main/*/run*/*/metrics.csv"):
    d = pd.read_csv(f)
    d["model"], d["run"] = f.parts[-4], int(f.parts[-3][3:])
    met.append(d)
met = pd.concat(met, ignore_index=True)
met["regime"] = met.case.map(lambda c: REG.get(c, "E2"))

# Table 1: model by fold -----------------------------------------------------------------------
rows = []
for c in CASES:
    r = {"Held-out case": LAB[c], "Regime": REG.get(c, "E2")}
    for m in MODELS:
        x = mbf[(mbf.model == m) & (mbf.case == c)].iloc[0]
        r[m] = f"{x.primary_mean:.3f} ({x.run_min:.3f}–{x.run_max:.3f})"
    rows.append(r)
md(pd.DataFrame(rows), OUT / "tables/table1_model_by_fold")

# Table 2: tuning variability -------------------------------------------------------------------
rows = []
for m in MODELS:
    g = mbf[mbf.model == m]
    gi, g5, gh = g[g.case.isin(["LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000"])], g[g.case == "LM_Channel_5200"], g[g.case.str.startswith("HJ")]
    rows.append({"Model": m,
                 "Between-run range, interp": f"{gi.between_run_range.min():.3f}–{gi.between_run_range.max():.3f}",
                 "Between-run range, LM 5200": f3(g5.between_run_range.iloc[0]),
                 "Between-run range, HJ": f"{gh.between_run_range.min():.3f}–{gh.between_run_range.max():.3f}",
                 "Mean seed SD, interp": f"{gi.mean_seed_sd.min():.3f}–{gi.mean_seed_sd.max():.3f}",
                 "Mean seed SD, LM 5200": f3(g5.mean_seed_sd.iloc[0])})
md(pd.DataFrame(rows), OUT / "tables/table2_tuning_variability")

# Table 3: factorial effects ---------------------------------------------------------------------
def cell(r):
    flags = (" · R" if r.resolved else "") + (" · TD" if r.tuning_dominated else "")
    return f"{r.estimate:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{flags}"


rows = []
for c in CASES:
    r = {"Case": LAB[c], "Regime": REG.get(c, "E2")}
    for k, nm in [("inv", "Δ_inv"), ("con", "Δ_con"), ("int", "Δ_int")]:
        r[nm] = cell(eff[(eff.case == c) & (eff.effect == k)].iloc[0])
    rows.append(r)
md(pd.DataFrame(rows), OUT / "tables/table3_factorial_effects")
rows = []
for c in CASES:
    r = {"Case": LAB[c]}
    for k, nm in [("M1_M2", "M2−M1"), ("M3_M4", "M4−M3"), ("M1_M3", "M3−M1"), ("M2_M4", "M4−M2")]:
        r[nm] = cell(eff[(eff.case == c) & (eff.effect == k)].iloc[0])
    rows.append(r)
md(pd.DataFrame(rows), OUT / "tables/table3b_pairwise_contrasts")

# Table 4: ablations ---------------------------------------------------------------------------
order = [("A-basis", "M4_basis3"), ("A-lin", "M4_lin"), ("A-inv-5", "M2"), ("A-inv-5", "M4"), ("A-inv-4", "M2"),
         ("A-inv-4", "M4"), ("A-scale", "M1"), ("A-scale", "M2"), ("A-scale", "M3"), ("A-scale", "M4")] + \
        [("A-soft", f"M3_soft-lam{l:g}") for l in [1e-3, 1e-2, 1e-1, 1, 10, 100]] + [("A-aug", "M1")]
rows = []
for ab_, v in order:
    r = {"Ablation": ab_, "Variant": v}
    for c in CASES[:5]:
        x = abl[(abl.ablation == ab_) & (abl.variant == v) & (abl.case == c)].iloc[0]
        r[LAB[c]] = f"{x['diff']:+.3f}{' R' if x.resolved else ''}"
    rows.append(r)
md(pd.DataFrame(rows), OUT / "tables/table4_ablations")

# Table 5: exact properties --------------------------------------------------------------------
st = met.groupby("model")[["eq_max", "trace_rel_max", "iso_limit_max", "asym_max", "f32_f64_pred_rel_max"]].max()
t5 = pd.DataFrame({"Property (max over 72 jobs × 10 seeds)": ["Rotation equivariance (rel.)", "Trace (rel.)", "Isotropic limit (abs.)",
                   "Symmetry (abs.)", "float32 vs float64 prediction (rel.)"]})
for m in MODELS:
    t5[m] = [f"{st.loc[m, k]:.2g}" for k in ["eq_max", "trace_rel_max", "iso_limit_max", "asym_max", "f32_f64_pred_rel_max"]]
md(t5, OUT / "tables/table5_exact_properties")

# Table 6: diagnostics by regime ---------------------------------------------------------------
cols = {"T3_E_Pi_rotated_mean": "T3 rotated E_Π", "E_Pi": "E_Π", "trace_rel_mean": "mean trace (rel.)",
        "E_Pi_viscous": "E_Π viscous", "E_Pi_buffer": "E_Π buffer", "E_Pi_log": "E_Π log", "E_Pi_outer": "E_Π outer",
        "D3_frac_negative": "D3 fraction negative"}
g = met.groupby(["regime", "model"])[list(cols)].mean().reset_index()
g = g.set_index("regime").loc[["interp", "extrap", "E2"]].reset_index()
for k in cols:
    g[k] = g[k].map(lambda v: f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}")
md(g.rename(columns=cols).rename(columns={"regime": "Regime", "model": "Model"}), OUT / "tables/table6_diagnostics_by_regime")

# Table 7: cost --------------------------------------------------------------------------------
cost = pd.read_csv(SUPP / "model_cost.csv")
man["model"] = man.job.str.split("__").str[0]
c7 = cost.groupby("model").agg(params_median=("n_params", "median"), params_min=("n_params", "min"), params_max=("n_params", "max"),
                               refit_s=("refit_seconds_mean", "median"), infer_us=("infer_us_per_point_f64", "median")).reset_index()
c7 = c7.merge((man.groupby("model").runtime_s.sum() / 3600).rename("core_h").reset_index(), on="model")
c7 = c7.assign(**{"Parameters, median (range)": c7.apply(lambda r: f"{int(r.params_median):,} ({int(r.params_min):,}–{int(r.params_max):,})", axis=1),
                  "Refit time per seed, s (median)": c7.refit_s.map(lambda v: f"{v:.1f}"),
                  "Inference, µs per point (float64, median)": c7.infer_us.map(lambda v: f"{v:.1f}"),
                  "Tuning + refit, core-h": c7.core_h.map(lambda v: f"{v:.1f}")})
md(c7[["model", "Parameters, median (range)", "Refit time per seed, s (median)", "Inference, µs per point (float64, median)",
       "Tuning + refit, core-h"]].rename(columns={"model": "Model"}), OUT / "tables/table7_cost")

# Table 8: A-scale sublayer --------------------------------------------------------------------
asg = pd.read_csv(SUPP / "ascale_seed_signs.csv")
asr = pd.read_csv(SUPP / "ascale_regions.csv")
rows = []
for m in MODELS:
    r = {"Model": m}
    for c in CASES[:5]:
        x = asr[(asr.model == m) & (asr.outer == c) & (asr.metric == "E_Pi_viscous")].iloc[0]
        s = asg[(asg.model == m) & (asg.outer == c) & (asg.region == "E_Pi_viscous")].iloc[0]
        r[LAB[c]] = f"{x.durbin:.2f} → {x.k_over_eps:.2f} ({int(s.seeds_worse_with_k_eps)}/10)"
    rows.append(r)
md(pd.DataFrame(rows), OUT / "tables/table8_ascale_sublayer")

# Table 9: A-soft trade-off --------------------------------------------------------------------
tr = pd.read_csv(SUPP / "asoft_training_trace.csv")
am = []
for f in RES.glob("ablations/A-soft/*/*/metrics.csv"):
    d = pd.read_csv(f)
    d["lambda"] = float(f.parts[-3].split("lam")[1])
    am.append(d)
am = pd.concat(am)
t9 = am.groupby("lambda").agg(test_E=("E_Pi", "mean"), test_trace=("trace_rel_mean", "mean"), test_trace_max=("trace_rel_max", "max")).reset_index()
t9 = t9.merge(tr.groupby("lambda").train_trace_rel_mean.mean().rename("train_trace").reset_index(), on="lambda")
m3 = met[(met.model == "M3") & (met.run == 1) & met.case.isin(CASES[:5])]
t9 = pd.concat([t9, pd.DataFrame([{"lambda": "hard (M3)", "test_E": m3.E_Pi.mean(), "test_trace": m3.trace_rel_mean.mean(),
                                   "test_trace_max": m3.trace_rel_max.max(), "train_trace": 0.0}])], ignore_index=True)
t9 = t9.assign(**{"λ_tr": t9["lambda"].map(lambda v: v if isinstance(v, str) else f"{v:g}"),
                  "Test E_Π (mean of 5 folds × 10 seeds)": t9.test_E.map(f3),
                  "Mean relative trace, test": t9.test_trace.map(lambda v: f"{v:.4f}" if v > 1e-6 else f"{v:.1e}"),
                  "Max relative trace, test": t9.test_trace_max.map(lambda v: f"{v:.3f}" if v > 1e-6 else f"{v:.1e}"),
                  "Mean relative trace, training": t9.train_trace.map(lambda v: f"{v:.4f}" if v > 1e-6 else "0 (exact)")})
md(t9[["λ_tr", "Test E_Π (mean of 5 folds × 10 seeds)", "Mean relative trace, test", "Max relative trace, test",
       "Mean relative trace, training"]], OUT / "tables/table9_asoft_tradeoff")

# Figure 1: model-by-fold with tuning-run spread -----------------------------------------------
rl = met.groupby(["model", "case", "run"]).E_Pi.mean().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [3, 2, 4]}, sharey=False)
groups = [("Interpolation (E1)", CASES[1:4]), ("Extrapolation (E1)", [CASES[0], CASES[4]]), ("External HJ (E2)", CASES[5:])]
for ax, (title, cs) in zip(axes, groups):
    for i, c in enumerate(cs):
        for j, m in enumerate(MODELS):
            x = i + (j - 1.5) * 0.18
            v = rl[(rl.model == m) & (rl.case == c)].E_Pi.to_numpy()
            ax.plot([x, x], [v.min(), v.max()], color=COL[m], lw=2, solid_capstyle="round")
            ax.plot(x, v.mean(), marker=MARK[m], color=COL[m], ms=6, mec="white", mew=1.0, ls="none",
                    label=m if (i == 0 and ax is axes[0]) else None)
    ax.set_xticks(range(len(cs)), [LAB[c] for c in cs])
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Held-out E_Π")
fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
save(fig, "fig1_model_by_fold")

# Figure 2: factorial effects with intervals -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, (k, nm) in zip(axes, [("inv", "Δ_inv (representation)"), ("con", "Δ_con (hard constraints)"), ("int", "Δ_int (interaction)")]):
    for i, c in enumerate(CASES):
        r = eff[(eff.case == c) & (eff.effect == k)].iloc[0]
        y = len(CASES) - 1 - i
        col = "#2b2a26" if (r.resolved and not r.tuning_dominated) else "#8a897f"
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(r.estimate, y, "o", ms=5.5, mfc=col if r.resolved else "white", mec=col, mew=1.3)
    ax.axvline(0, color="#8a897f", lw=0.9)
    ax.set_title(nm, fontsize=9, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
fig.supxlabel("Change in held-out E_Π (positive = higher error)", fontsize=9, y=0.04)
axes[0].set_yticks(range(len(CASES)), [LAB[c] for c in CASES[::-1]])
for ax in axes:
    ax.axhspan(-0.5, 3.5, color="#f1f0eb", zorder=0, lw=0)
fig.text(0.01, -0.04, "Filled: interval excludes 0. Dark: resolved and not tuning-dominated. Shaded rows: HJ (E2).", fontsize=7.5, color="#5d5c55")
fig.tight_layout()
save(fig, "fig2_factorial_effects")

# Figure 3: rotated-frame error -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, reg in zip(axes, ["interp", "extrap", "E2"]):
    g = met[met.regime == reg].groupby("model")[["E_Pi", "T3_E_Pi_rotated_mean"]].mean()
    for j, m in enumerate(MODELS):
        ax.plot([j - 0.12], [g.loc[m, "E_Pi"]], marker=MARK[m], color=COL[m], ms=6, mfc="white", mew=1.4, ls="none")
        ax.plot([j + 0.12], [g.loc[m, "T3_E_Pi_rotated_mean"]], marker=MARK[m], color=COL[m], ms=6, mec="white", ls="none")
    ax.axhline(1.0, color="#8a897f", lw=0.9, ls="--")
    ax.set_xticks(range(4), MODELS)
    ax.set_title({"interp": "Interpolation", "extrap": "Extrapolation", "E2": "External HJ"}[reg], fontsize=9, loc="left", color=INK)
    ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("E_Π")
fig.text(0.01, -0.05, "Hollow: channel frame. Filled: mean over 20 random frames (T3). Dashed: error of a zero prediction.", fontsize=7.5, color="#5d5c55")
fig.tight_layout()
save(fig, "fig3_rotated_frame")

# Figure 4: A-soft trade-off (two panels, shared x) ----------------------------------------------
lam = t9[t9["lambda"].map(lambda v: not isinstance(v, str))]
lv = lam["lambda"].astype(float).to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
axes[0].plot(lv, lam.test_E, "-o", color=COL["M3"], ms=5, lw=2)
axes[0].axhline(m3.E_Pi.mean(), color="#2b2a26", lw=1, ls="--")
axes[0].text(lv[0], m3.E_Pi.mean(), " hard projection (M3)", va="bottom", fontsize=7.5, color="#2b2a26")
axes[0].set_ylabel("Test E_Π (mean of 5 folds)")
axes[1].plot(lv, lam.test_trace, "-o", color=COL["M3"], ms=5, lw=2, label="test folds")
axes[1].plot(lv, lam.train_trace, "--s", color="#5d5c55", ms=4.5, lw=1.5, label="training cases")
axes[1].axhline(1e-3, color="#8a897f", lw=0.9, ls=":")
axes[1].text(lv[0], 1.1e-3, " 10⁻³", fontsize=7.5, color="#5d5c55")
axes[1].set_yscale("log")
axes[1].set_ylabel("Mean relative trace")
axes[1].legend(frameon=False, fontsize=7.5)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("Penalty weight λ_tr")
fig.tight_layout()
save(fig, "fig4_asoft_tradeoff")

# Figure 5: A-scale sublayer --------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
for ax, m in zip(axes, MODELS):
    x = asr[(asr.model == m) & (asr.metric == "E_Pi_viscous")].set_index("outer").loc[CASES[:5]]
    xs = np.arange(5)
    ax.plot(xs, x.durbin, "-" + MARK[m], color=COL[m], ms=5, lw=2, label="Durbin-bounded τ")
    ax.plot(xs, x.k_over_eps, "--" + MARK[m], color=COL[m], ms=5, lw=1.5, mfc="white", label="τ = k/ε")
    ax.set_xticks(xs, [str(int(c[-4:])) for c in CASES[:5]], rotation=0, fontsize=7.5)
    ax.set_title(m, fontsize=9, loc="left", color=INK)
    ax.set_xlabel("Held-out Re_τ")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Viscous-sublayer E_Π")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
save(fig, "fig5_ascale_sublayer")

# manifest of generated outputs ---------------------------------------------------------------
files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
(OUT / "generated_files.json").write_text(json.dumps(files, indent=1))
print(f"{len(files)} files written to {OUT}")
