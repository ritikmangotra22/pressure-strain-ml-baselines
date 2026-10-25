"""Cross-machine check for the a posteriori stage: compare aposteriori_recheck.jsonl (tasks recomputed
on a second machine) with the same tasks in aposteriori_lm.jsonl. Agreement criteria, fixed before
the check is run: identical solve outcome on every case, amplification within 1 % relative, and
J_R within 1e-6 where both converged.

    python scripts/paper3/compare_recheck.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/paper3/final"


def load(f):
    return {(r["model"], r["seed"], r["case"]): r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def main():
    a, b = load(D / "aposteriori_lm.jsonl"), load(D / "aposteriori_recheck.jsonl")
    bad, n = [], 0
    for k, rb in b.items():
        ra = a.get(k)
        if ra is None:
            bad.append((k, "not in main file"))
            continue
        n += 1
        if ra.get("solve_status") != rb.get("solve_status"):
            bad.append((k, f"status {ra.get('solve_status')} vs {rb.get('solve_status')}"))
        x, y = ra.get("amplification"), rb.get("amplification")
        if x is not None and y is not None and abs(x - y) > 0.01 * abs(x):
            bad.append((k, f"amplification {x:.6g} vs {y:.6g}"))
        if "J_R" in ra and "J_R" in rb and abs(ra["J_R"] - rb["J_R"]) > 1e-6:
            bad.append((k, f"J_R {ra['J_R']:.6g} vs {rb['J_R']:.6g}"))
    hosts = {r.get("host", "cloud container (before provenance field)") for r in b.values()}
    print(f"{n} case results compared; recheck host(s): {sorted(hosts)}")
    print("AGREE" if not bad else f"DISAGREE on {len(bad)}:")
    for k, why in bad:
        print(" ", k, why)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
