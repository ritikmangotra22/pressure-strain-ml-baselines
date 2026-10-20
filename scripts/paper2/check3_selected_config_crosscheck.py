"""Check 3: train the configuration selected by the Paper 2 re-tuning at Re_tau = 5200 with
Paper 1's own code. Equal results mean the 5200 deviation comes from model selection, not code.

    python scripts/paper2/check3_selected_config_crosscheck.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/paper2"))
import check3_r0 as C  # noqa: E402
from ps.models import make_model  # noqa: E402

held = "LM_Channel_5200"
p = json.loads((C.OUT / f"r0_retune_{held}_params.json").read_text())["params"]
tabs = C.tables()
tr = C.renorm(pd.concat([tabs[c] for c in C.CASES if c != held]))
ref = pd.read_csv(C.OUT / f"r0_retune_{held}.csv")
rows = []
for s in C.SEEDS:
    m = make_model("mlp", C.F1, s, p)
    m.fit(tr)
    e = C.score(m, tabs[held])
    r = float(ref[ref.seed == s].E_Pi.iloc[0])
    rows.append({"seed": s, "E_Pi_paper1_code": e, "E_Pi_paper2_code": r, "abs_diff": abs(e - r)})
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "r0_selected_config_crosscheck_5200.csv", index=False)
print(df.to_string())
print("means:", df.E_Pi_paper1_code.mean(), df.E_Pi_paper2_code.mean(), "max |diff|:", df.abs_diff.max())
