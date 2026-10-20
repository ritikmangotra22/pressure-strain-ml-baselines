"""Paper 3 reference model M0: Paper 2 run-1 M4 configuration (E2 tuning, all five LM cases),
scored by the Paper 3 inner leave-one-Re_tau-out procedure with training seeds 0-4.

Outputs results/paper3/reference_m0/m0_inner.json: per-fold E_Pi (mean and per seed), the
return-to-isotropy violation and coefficient norm used to normalise f_phys, and FLOPs.
Seed 0 must reproduce Paper 2's recorded inner_val_E_Pi (0.05648547...), which checks that
src/ps3 trains exactly as src/ps2 does when dropout = 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, pipeline as PL  # noqa: E402
from ps3 import search  # noqa: E402

CASES = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"]
SEL = json.loads((ROOT / "results/paper2/main/M4/run1/E2/selected.json").read_text())
M0 = {"arch": "M4", "n_layers": SEL["n_layers"], "width": SEL["width"], "activation": SEL["activation"],
      "lr": SEL["lr"], "weight_decay": SEL["weight_decay"], "dropout": 0.0, "c_t": 6.0,
      "n_basis": 8, "inv_set": "six"}


def main():
    lm = data.load("LM")
    train = {c: PL.prepare(lm[lm.case == c]) for c in CASES}
    per_seed = {}
    for s in range(5):
        r = search.evaluate_candidate(M0, train, seeds=(s,))
        per_seed[s] = r
        print(f"seed {s}: f_err={r['f_err']:.6f} inner={np.round(r['inner_errors_seed0'], 4).tolist()}", flush=True)
    errs = np.array([per_seed[s]["inner_errors_seed0"] for s in per_seed])
    out = {"config": M0, "paper2_inner_val_E_Pi": SEL["inner_val_E_Pi"],
           "seed0_f_err": per_seed[0]["f_err"],
           "seed0_matches_paper2": abs(per_seed[0]["f_err"] - SEL["inner_val_E_Pi"]) < 1e-6,
           "fold_E_Pi_mean": dict(zip(CASES, errs.mean(0).tolist())),
           "fold_E_Pi_sd": dict(zip(CASES, errs.std(0, ddof=1).tolist())),
           "fold_E_Pi_per_seed": {str(s): dict(zip(CASES, errs[s].tolist())) for s in per_seed},
           "rti_mean": float(np.mean([per_seed[s]["rti"] for s in per_seed])),
           "coef_norm_mean": float(np.mean([per_seed[s]["coef_norm"] for s in per_seed])),
           "f_cost_flops": per_seed[0]["f_cost_flops"]}
    d = ROOT / "results/paper3/reference_m0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "m0_inner.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("seed0_f_err", "seed0_matches_paper2", "fold_E_Pi_mean", "rti_mean", "coef_norm_mean", "f_cost_flops")}, indent=1))


if __name__ == "__main__":
    main()
