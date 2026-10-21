"""Build, validate and hash the Moser-Kim-Mansour tables (Paper 3 confirmatory data).

No model is loaded or evaluated here. Outputs in results/paper3/mkm_data/:
  processed_table_mkm.csv   Re_tau 395 and 590 (confirmatory test cases)
  processed_table_mkm180.csv  Re_tau 180, used ONLY for the convention check against Lee & Moser 180
  validation.json           Paper 1 consistency checks per case, MKM180-vs-LM180 comparison, SHA-256s
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps.features import build_table  # noqa: E402
from ps.io_hj import validate_hj  # noqa: E402
from ps.io_lm import sha256  # noqa: E402
from ps2 import data  # noqa: E402
from ps3.io_mkm import load_mkm_case  # noqa: E402

RAW = ROOT / "data/raw/moser_kim_mansour_1999"
OUT = ROOT / "results/paper3/mkm_data"
COLS = ["b11", "b22", "b33", "b12", "eta", "Re_t", "Pi11", "Pi22", "Pi33", "Pi12"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, tables = {"archive_sha256": sha256(RAW / "chandata.tar.gz")}, {}
    for re in (180, 395, 590):
        case = load_mkm_case(RAW, re)
        report[case.name] = {"checks": validate_hj(case), "raw_sha256": case.checksums}
        tables[re] = build_table(case)
        report[case.name]["n_points"] = len(tables[re])
    test = pd.concat([tables[395], tables[590]], ignore_index=True)
    test.to_csv(OUT / "processed_table_mkm.csv", index=False)
    tables[180].to_csv(OUT / "processed_table_mkm180.csv", index=False)
    report["processed_table_mkm.csv"] = {"sha256": sha256(OUT / "processed_table_mkm.csv"), "n_points": len(test)}
    report["processed_table_mkm180.csv"] = {"sha256": sha256(OUT / "processed_table_mkm180.csv"),
                                            "n_points": len(tables[180])}

    # Convention check: MKM 180 vs LM 180 (training case), interpolated onto MKM y+ inside both grids
    lm = data.load("LM")
    lm180 = lm[lm.case == "LM_Channel_0180"].sort_values("y_plus")
    m = tables[180]
    m = m[(m.y_plus >= lm180.y_plus.min()) & (m.y_plus <= lm180.y_plus.max())]
    cmp = {}
    for c in COLS:
        li = np.interp(m.y_plus, lm180.y_plus, lm180[c])
        scale = np.abs(lm180[c]).max()
        cmp[c] = {"max_abs_diff_over_max": float(np.abs(m[c] - li).max() / scale),
                  "median_abs_diff_over_max": float(np.median(np.abs(m[c] - li)) / scale)}
    report["MKM180_vs_LM180"] = cmp
    (OUT / "validation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or "checks" not in v else
                          {kk: v["checks"][kk] for kk in ("dns_trace_rel_median", "dns_trace_rel_max",
                                                          "P_uu_over_-2uvdU", "total_stress_max_dev",
                                                          "Re_tau_from_grid")} | {"n": v["n_points"]})
                      for k, v in report.items() if k != "archive_sha256"}, indent=1))


if __name__ == "__main__":
    main()
