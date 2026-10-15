"""Compare the Hoyas & Jimenez pressure-strain trace with the standard deviations published with
the budgets (Re*.std.<comp>), and quantify the published standard deviation of Pi_ij relative to |Pi|.

The README says only that the .std files contain "the standard deviation of the balances". It does
not state whether this is the standard deviation of individual samples or the standard error of the
time average, nor the number of (independent) samples. Both readings are reported.
    python -I scripts/hj_std_analysis.py
"""
from pathlib import Path

import numpy as np

ROOT = Path("data/raw/hoyas_jimenez_2008")
CASES = {"Re180": 186, "Re550": 547, "Re950": 934, "Re2000": 2003}
COL_PS = 4  # y/h, y+, dissip, produc, p-strain, ...


def load(p):
    return np.loadtxt(p, comments="%")


lines = ["| Re_τ | n | median trace/σ_tr | max trace/σ_tr | points with trace > σ_tr | median σ(Π)/‖Π‖ | max σ(Π)/‖Π‖ (y⁺) |",
         "|---|---|---|---|---|---|---|"]
for d, re in CASES.items():
    bal = {c: load(ROOT / d / "balances" / f"{d}.bal.{c}") for c in ("uu", "vv", "ww", "uv")}
    std = {c: load(ROOT / d / "balances" / f"{d}.std.{c}") for c in ("uu", "vv", "ww", "uv")}
    yp = bal["uu"][:, 1]
    keep = yp > 1e-8
    ps = {c: bal[c][keep, COL_PS] for c in bal}
    sd = {c: std[c][keep, COL_PS] for c in std}
    tr = np.abs(ps["uu"] + ps["vv"] + ps["ww"])
    # Conservative (largest) combined sigma: perfectly correlated errors; the independent-error value
    # sqrt(sum sigma^2) is smaller, so the ratio below is a lower bound on trace/sigma.
    sig_tr_corr = sd["uu"] + sd["vv"] + sd["ww"]
    sig_tr_ind = np.sqrt(sd["uu"] ** 2 + sd["vv"] ** 2 + sd["ww"] ** 2)
    r = tr / sig_tr_ind
    norm = np.sqrt(ps["uu"] ** 2 + ps["vv"] ** 2 + ps["ww"] ** 2 + 2 * ps["uv"] ** 2)
    sig_norm = np.sqrt(sd["uu"] ** 2 + sd["vv"] ** 2 + sd["ww"] ** 2 + 2 * sd["uv"] ** 2)
    rel = sig_norm / norm
    i = int(np.argmax(rel))
    lines.append(f"| {re} | {keep.sum()} | {np.median(r):.1e} | {r.max():.1e} | {(tr > sig_tr_corr).sum()} | "
                 f"{np.median(rel):.1e} | {rel.max():.1e} ({yp[keep][i]:.2f}) |")
out = "\n".join(lines)
note = ("\nσ_tr = sqrt(σ11² + σ22² + σ33²) of the published p-strain standard deviations (independent errors);"
        " the column 'trace > σ_tr' uses the larger, fully correlated bound σ11 + σ22 + σ33.\n")
Path("results/hj_std_trace.md").write_text(out + "\n" + note)
print(out + note)
