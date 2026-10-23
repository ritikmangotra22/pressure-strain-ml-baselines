"""Paper 3 search figures and the RQ4 cost table, from INNER objectives only (no test data).

  results/paper3/figures/hv_history.{pdf,png}       hypervolume vs evaluations, median + IQR over
                                                    optimizer seeds, Arm A and Arm B panels
  results/paper3/figures/selected_models.{pdf,png}  each job's selected model: inner f_err vs FLOPs
  results/paper3/figures/cost_table.csv             per strategy x arm: training and optimizer
                                                    overhead per evaluation, core-hours per job
Only complete jobs (200 counted evaluations) are used; the number of jobs per cell is printed.

    python scripts/paper3/figures_search.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper3"))
import analyse_search as A  # noqa: E402
import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = ROOT / "results/paper3/figures"
# reference categorical palette (dataviz skill, light mode), fixed order, plus line styles so
# identity never rests on colour alone
STYLE = {"random": ("#2a78d6", "-", "o", "Random"), "tpe": ("#eb6834", "--", "s", "TPE (f_err)"),
         "nsga2": ("#1baf7a", "-.", "^", "NSGA-II"), "qnehvi": ("#eda100", ":", "D", "qLogNEHVI"),
         "nsga2_cond": ("#e87ba4", "-", "v", "NSGA-II + f_cond"), "nsga2_nogen": ("#008300", "--", "<", "A2 no f_gen"),
         "nsga2_nophys": ("#6250d6", "-.", ">", "A4 no f_phys"), "wsum": ("#e34948", ":", "P", "A3 weighted sum")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRID_N = list(range(10, 201, 10))


def complete_jobs():
    for d in sorted(A.S.iterdir()):
        st = d / "status.json"
        if st.exists() and json.loads(st.read_text()).get("counted") == 200:
            s, arm, seed = d.name.split("__")
            yield d.name, s, arm[3:], int(seed[4:])


def style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hist, sel, cost = [], [], []
    for job, s, arm, seed in complete_jobs():
        df = A.load(job)
        P = df[A.OBJ].to_numpy()
        hist += [{"strategy": s, "arm": arm, "seed": seed, "n": n, "hv": A.hv(P[:n], arm)} for n in GRID_N]
        r = A.select(df, arm, s)
        if r is not None:
            sel.append({"strategy": s, "arm": arm, "seed": seed, "f_err": r.f_err, "f_cost": r.f_cost, "f_gen": r.f_gen})
        # the sampler proposes when Optuna creates the trial, before the objective starts, so the
        # optimizer overhead is the trial's stored duration minus the evaluation time we logged
        rows = {x["trial"]: x for x in map(json.loads, (A.S / job / "evals.jsonl").read_text().splitlines()) if "wall_seconds" in x}
        st = optuna.load_study(study_name=None, storage=f"sqlite:///{A.S / job / 'study.db'}")
        dur = {t.number: (t.datetime_complete - t.datetime_start).total_seconds() for t in st.trials
               if t.datetime_complete is not None and t.number in rows}
        tr = np.array([rows[k]["propose_to_done_s"] for k in dur])
        tot = np.array([max(dur[k], rows[k]["propose_to_done_s"]) for k in dur])
        cost.append({"strategy": s, "arm": arm, "seed": seed, "median_eval_s": float(np.median(tr)),
                     "median_overhead_s": float(np.median(tot - tr)), "overhead_share": float((tot - tr).sum() / tot.sum()),
                     "core_hours": float(tot.sum() / 3600)})
    H, S_, C = pd.DataFrame(hist), pd.DataFrame(sel), pd.DataFrame(cost)
    H.to_csv(OUT / "hv_history.csv", index=False)
    S_.to_csv(OUT / "selected_models.csv", index=False)
    ct = C.groupby(["arm", "strategy"]).agg(jobs=("seed", "size"), median_eval_s=("median_eval_s", "median"),
                                            median_overhead_s=("median_overhead_s", "median"),
                                            overhead_share=("overhead_share", "median"),
                                            core_hours_per_job=("core_hours", "median")).round(3).reset_index()
    ct.to_csv(OUT / "cost_table.csv", index=False)
    print(ct.to_string(index=False))

    # ---- Figure: hypervolume history
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for ax, arm in zip(axes, ("A", "B")):
        h = H[H.arm == arm]
        for s in STYLE:
            g = h[h.strategy == s]
            if g.empty:
                continue
            q = g.groupby("n").hv.quantile([0.25, 0.5, 0.75]).unstack()
            col, ls, _, lab = STYLE[s]
            ax.fill_between(q.index, q[0.25], q[0.75], color=col, alpha=0.12, lw=0)
            ax.plot(q.index, q[0.5], color=col, ls=ls, lw=2, label=f"{lab} (n={g.seed.nunique()})")
        style_axes(ax)
        ax.set_title(f"Arm {arm}: " + ("M4 hyperparameters" if arm == "A" else "architecture searched"),
                     fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("Evaluations", fontsize=8, color=MUTED)
    axes[0].set_ylabel("Hypervolume (M0 units)", fontsize=8, color=MUTED)
    for ax in axes:
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"hv_history.{ext}", dpi=200)
    plt.close(fig)

    # ---- Figure: selected models, accuracy vs cost
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, arm in zip(axes, ("A", "B")):
        g0 = S_[S_.arm == arm]
        for s in STYLE:
            g = g0[g0.strategy == s]
            if g.empty:
                continue
            col, _, mk, lab = STYLE[s]
            ax.scatter(g.f_cost, g.f_err, s=36, marker=mk, color=col, edgecolor="white", lw=0.8, label=lab, zorder=3)
        ax.axvline(A.M0["f_cost_flops"], color=MUTED, lw=1, ls="--")
        ax.axhline(A.F_ERR0, color=MUTED, lw=1, ls="--")
        ax.text(A.M0["f_cost_flops"], ax.get_ylim()[1] if False else A.F_ERR0, " M0", fontsize=7, color=MUTED, va="bottom")
        ax.set_xscale("log")
        style_axes(ax)
        ax.set_title(f"Arm {arm}", fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("FLOPs per evaluation", fontsize=8, color=MUTED)
    axes[0].set_ylabel("Inner mean error f_err", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"selected_models.{ext}", dpi=200)
    plt.close(fig)
    print("jobs per cell:", H.groupby(["arm", "strategy"]).seed.nunique().to_dict())


if __name__ == "__main__":
    main()
