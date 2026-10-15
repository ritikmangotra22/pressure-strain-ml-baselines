"""Emit manuscript-ready markdown for Tables 10-14 directly from result files (no hand transcription)."""
import json
import pandas as pd

NICE = {"ssg_published": "Rotta+SSG, published", "ssg_refit": "Rotta+SSG, refitted", "lrr_published": "Rotta+LRR-IP, published",
        "lrr_refit": "Rotta+LRR-IP, refitted", "lrr_gl_published": "Rotta+LRR-IP+GL wall, published", "lrr_gl_refit": "Rotta+LRR-IP+GL wall, refitted", "ridge": "Ridge", "rf": "Random forest", "lgbm": "LightGBM", "mlp": "MLP"}
ORDER = list(NICE)
FOLDS = ["0180", "0550", "1000", "2000", "5200"]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


e1 = pd.read_csv("results/e1_F1/metrics_long.csv")
e2 = pd.read_csv("results/e2/metrics_long.csv")
P = json.load(open("results/e1_F1/best_params.json"))
T = {}

rows = []
for m in ORDER:
    for reg, lab in (("interp", "Interp."), ("extrap", "Extrap.")):
        d = e1[(e1.model == m) & (e1.regime == reg)]
        rows.append({"Model": NICE[m], "Folds": lab, **{r.capitalize(): f"{d['E_Pi_' + r].mean():.3f}" for r in ("viscous", "buffer", "log", "outer")}})
T["t10"] = md(pd.DataFrame(rows))


def hp(m, f):
    v = dict(P.get(f"{m}__LM_Channel_{f}", {}))
    if "fit_info" in v and "coef" in v["fit_info"]:
        return ", ".join(f"{k}={x:.2f}" for k, x in v["fit_info"]["coef"].items())
    v.pop("fit_info", None)
    if m == "ridge":
        return f"deg {v['degree']}, α={v['alpha']:.1e}"
    if m == "rf":
        return f"{v['n_estimators']} trees, depth {v['max_depth']}, leaf {v['min_samples_leaf']}, feat {v['max_features']}"
    if m == "lgbm":
        return f"lr {v['learning_rate']:.3f}, leaves {v['num_leaves']}, rounds {'/'.join(map(str, v['n_estimators']))}"
    if m == "mlp":
        return f"{v['n_layers']}×{v['width']} {v['activation']}, lr {v['lr']:.1e}, wd {v['weight_decay']:.1e}, {v['epochs']} ep"
    return "—"


rows = [{"Model": NICE[m], **{str(int(f)): hp(m, f) for f in FOLDS}} for m in ("ssg_refit", "lrr_refit", "lrr_gl_refit", "ridge", "rf", "lgbm", "mlp")]
T["t11"] = md(pd.DataFrame(rows))


def ms(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


cases = [("HJ_Channel_0180", "HJ 186"), ("HJ_Channel_0550", "HJ 547"), ("HJ_Channel_0950", "HJ 934"), ("HJ_Channel_2000", "HJ 2003")]
rows = []
for m in ORDER:
    d = e2[e2.model == m]
    r = {"Model": NICE[m]}
    for c, lab in cases:
        x = d[d.test_case == c]
        r[lab] = f"{ms(x['E_Pi'])} [{x['E_Pi_ci_lo'].mean():.3f}, {x['E_Pi_ci_hi'].mean():.3f}]"
    rows.append(r)
T["t12"] = md(pd.DataFrame(rows))
rows = [{"Model": NICE[m], **{r.capitalize(): f"{e2[e2.model == m]['E_Pi_' + r].mean():.3f}" for r in ("viscous", "buffer", "log", "outer")}} for m in ORDER]
T["t12c"] = md(pd.DataFrame(rows))

a2 = pd.read_csv("results/e1_F1/ablation_A2_trace.csv").groupby("model")[["E_Pi_raw", "E_Pi_reconstruct", "E_Pi_project"]].mean()
rows = []
for m in ORDER:
    d = e1[e1.model == m]
    r = {"Model": NICE[m], "Mean t_n": f"{d['trace_mean'].mean():.1e}", "Max t_n": f"{d['trace_max'].max():.1e}"}
    if m in a2.index:
        r.update({"E_Π raw": f"{a2.loc[m, 'E_Pi_raw']:.4f}", "E_Π, Π₃₃ reconstructed": f"{a2.loc[m, 'E_Pi_reconstruct']:.4f}",
                  "E_Π, trace projected": f"{a2.loc[m, 'E_Pi_project']:.4f}"})
    else:
        r.update({"E_Π raw": "—", "E_Π, Π₃₃ reconstructed": "—", "E_Π, trace projected": "—"})
    rows.append(r)
T["t13"] = md(pd.DataFrame(rows))

rows = []
for m in ORDER:
    d = e1[e1.model == m]
    n_trials = {"ridge": 100, "rf": 100, "lgbm": 100, "mlp": 100}.get(m, 0)
    rows.append({"Model": NICE[m], "Tuning trials per fold": n_trials or "—",
                 "Tuning time per fold (s)": f"{d.groupby('held_out')['tune_time_s'].first().mean():.0f}",
                 "Final fit per seed (s)": f"{d['fit_time_s'].mean():.2f}",
                 "Inference per point (µs)": f"{1e6 * d['infer_time_s_per_pt'].mean():.1f}"})
T["t14"] = md(pd.DataFrame(rows))
json.dump(T, open("results/doc_tables.json", "w"), indent=1, ensure_ascii=False)
for k, v in T.items():
    print(f"== {k}\n{v}\n")
