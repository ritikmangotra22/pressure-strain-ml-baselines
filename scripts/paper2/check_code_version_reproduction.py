"""Stage D: show that jobs recorded under different runner code versions are reproduced bit-for-bit
by the current code (src/ps2 unchanged since 2026-10-17 16:31 UTC; only the runner CLI changed).
Refits seed 0 of the chosen jobs from their saved selected.json and compares predictions."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/paper2"))
from ps2 import pipeline as PL  # noqa
import run_protocol as RP  # noqa

JOBS = sys.argv[1:] or ["M1/run1/LM_Channel_0180", "M2/run1/LM_Channel_0180", "M4/run2/LM_Channel_1000"]
out = []
for j in JOBS:
    d = ROOT / "results/paper2/main" / j
    model, run, outer = j.split("/")
    params = json.loads((d / "selected.json").read_text())
    S = RP.splits()
    train = PL.concat([S[c] for c in RP.E1 if c != outer])
    f, info, _ = PL.fit(model, train, params, seed=0, epochs=params["epochs"])
    _, P_hat = PL.evaluate(f, S[outer])
    saved = pd.read_csv(d / "predictions" / f"{outer}__seed0.csv")
    idx = {"11": (0, 0), "22": (1, 1), "33": (2, 2), "12": (0, 1), "13": (0, 2), "23": (1, 2)}
    diff = max(float(np.abs(saved[f"Pi{k}_hat"].to_numpy() - P_hat[:, a, b]).max()) for k, (a, b) in idx.items())
    st = json.loads((d / "status.json").read_text())
    out.append({"job": j, "recorded_code_version": st.get("code_version"), "current_code_version": RP.code_version(),
                "max_abs_pred_diff_seed0": diff, "bitwise_identical": diff == 0.0})
    print(out[-1], flush=True)
p = ROOT / "results/paper2/analysis/code_version_reproduction.json"
p.write_text(json.dumps(out, indent=1))
