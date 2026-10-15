"""Post hoc statistics on the saved E1/E2 test predictions (no retraining, no new model choices).

1. Unweighted E_Pi (every grid point weighted equally), alongside the primary grid-weighted E_Pi.
2. Block-length sensitivity of the 95 % moving-block bootstrap interval (blocks of 5, 10, 20 % of the
   profile), for the seed-averaged E_Pi.
3. Paired moving-block bootstrap of the difference in seed-averaged E_Pi between two models on the
   same resampled points (block 10 %, 2000 resamples).
    python scripts/posthoc_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ps.metrics import e_pi  # noqa: E402

PRED = {"E1": [Path("results/e1_F1/predictions"), Path("results/e1_F1_parts/lrr_gl/predictions")],
        "E2": [Path("results/e2/predictions"), Path("results/e2_parts/lrr_gl/predictions")]}
TABLE = {"E1": "results/e1_F1/processed_table.csv", "E2": "results/e2/processed_table_hj.csv"}
CASES = {"E1": ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"],
         "E2": ["HJ_Channel_0180", "HJ_Channel_0550", "HJ_Channel_0950", "HJ_Channel_2000"]}
MODELS = ["ssg_published", "ssg_refit", "lrr_published", "lrr_refit", "lrr_gl_published", "lrr_gl_refit",
          "ridge", "rf", "lgbm", "mlp"]
PAIRS = [("mlp", "lgbm"), ("rf", "lgbm"), ("mlp", "rf"), ("ridge", "ssg_refit"), ("lgbm", "ssg_refit"),
         ("ssg_refit", "lrr_refit"), ("lrr_gl_published", "lrr_published"), ("lrr_gl_refit", "lrr_refit")]
TGT = ["Pi11", "Pi22", "Pi33", "Pi12"]
N_BOOT = 2000


def load(exp, model, case):
    out = []
    for d in PRED[exp]:
        out += sorted(d.glob(f"{model}__{case}__seed*.csv"))
    return [pd.read_csv(f) for f in out]


def blocks(n, frac, rng):
    L = max(2, int(round(frac * n)))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(N_BOOT, nb))
    return (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(N_BOOT, -1)[:, :n]


def e_pi_rows(yh, y, w, idx):
    """Vectorised E_Pi for each bootstrap row of indices."""
    m = np.array([1, 1, 1, 2.0])
    d = ((yh[idx] - y[idx]) ** 2 * m).sum(-1)
    t = (y[idx] ** 2 * m).sum(-1)
    ww = w[idx]
    return np.sqrt((ww * d).sum(1) / (ww * t).sum(1))


rows_u, rows_b, rows_p = [], [], []
for exp in ("E1", "E2"):
    tab = pd.read_csv(TABLE[exp])
    for case in CASES[exp]:
        t = tab[tab.case == case].reset_index(drop=True)
        w = t["w"].to_numpy()
        preds = {}
        for m in MODELS:
            runs = load(exp, m, case)
            if not runs:
                continue
            for r in runs:
                assert np.allclose(r["y_plus"].to_numpy(), t["y_plus"].to_numpy()), (exp, m, case)
            preds[m] = [r[[c + "_hat" for c in TGT]].to_numpy() for r in runs]
        y = t[TGT].to_numpy()
        re = int(t["re_tau"].iloc[0])
        # 1. unweighted vs weighted
        for m, P in preds.items():
            rows_u.append({"exp": exp, "re_tau": re, "model": m,
                           "E_Pi_weighted": np.mean([e_pi(p, y, w) for p in P]),
                           "E_Pi_unweighted": np.mean([e_pi(p, y, np.ones(len(y))) for p in P])})
        # 2. block-length sensitivity
        for m, P in preds.items():
            rec = {"exp": exp, "re_tau": re, "model": m}
            for frac in (0.05, 0.10, 0.20):
                idx = blocks(len(y), frac, np.random.default_rng(0))
                v = np.mean([e_pi_rows(p, y, w, idx) for p in P], axis=0)
                rec[f"lo_{frac}"], rec[f"hi_{frac}"] = np.percentile(v, [2.5, 97.5])
            rows_b.append(rec)
        # 3. paired differences
        idx = blocks(len(y), 0.10, np.random.default_rng(1))
        for a, b in PAIRS:
            if a not in preds or b not in preds:
                continue
            va = np.mean([e_pi_rows(p, y, w, idx) for p in preds[a]], axis=0)
            vb = np.mean([e_pi_rows(p, y, w, idx) for p in preds[b]], axis=0)
            d = va - vb
            point = np.mean([e_pi(p, y, w) for p in preds[a]]) - np.mean([e_pi(p, y, w) for p in preds[b]])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows_p.append({"exp": exp, "re_tau": re, "A": a, "B": b, "dE": point, "lo": lo, "hi": hi,
                           "verdict": "A lower" if hi < 0 else ("B lower" if lo > 0 else "not resolved")})

out = Path("results/posthoc")
out.mkdir(exist_ok=True)
U, B, Pp = pd.DataFrame(rows_u), pd.DataFrame(rows_b), pd.DataFrame(rows_p)
U.to_csv(out / "unweighted.csv", index=False)
B.to_csv(out / "bootstrap_block_sensitivity.csv", index=False)
Pp.to_csv(out / "paired_bootstrap.csv", index=False)

md = ["### Unweighted vs grid-weighted E_Π (seed means)\n"]
pv = U.pivot_table(index=["model"], columns=["exp", "re_tau"], values=["E_Pi_unweighted"]).round(3)
pv2 = U.pivot_table(index=["model"], columns=["exp", "re_tau"], values=["E_Pi_weighted"]).round(3)
md += [pv.reindex(MODELS).to_markdown(), "", pv2.reindex(MODELS).to_markdown(), ""]
md += ["### Bootstrap block-length sensitivity: 95 % interval of seed-averaged E_Π\n"]
B["5 %"] = B.apply(lambda r: f"[{r['lo_0.05']:.3f}, {r['hi_0.05']:.3f}]", axis=1)
B["10 %"] = B.apply(lambda r: f"[{r['lo_0.1']:.3f}, {r['hi_0.1']:.3f}]", axis=1)
B["20 %"] = B.apply(lambda r: f"[{r['lo_0.2']:.3f}, {r['hi_0.2']:.3f}]", axis=1)
md += [B[["exp", "re_tau", "model", "5 %", "10 %", "20 %"]].to_markdown(index=False), ""]
md += ["### Paired block bootstrap of ΔE_Π = E_Π(A) − E_Π(B) (seed-averaged; block 10 %; 2000 resamples)\n"]
Pp["ΔE [95 % CI]"] = Pp.apply(lambda r: f"{r.dE:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
md += [Pp[["exp", "re_tau", "A", "B", "ΔE [95 % CI]", "verdict"]].to_markdown(index=False)]
(out / "posthoc_stats.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
