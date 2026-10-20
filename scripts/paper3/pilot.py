"""Paper 3 timing pilot (draft protocol, Stage 0).

Uses ONLY the five Lee & Moser training cases, scored by inner leave-one-Re_tau-out. No Hoyas &
Jimenez or Moser-Kim-Mansour data are loaded. Purpose: time one candidate evaluation without
pruning, measure seed-to-seed spread of each objective, and check that every objective is finite
and varies across configurations. Results are not used for any scientific claim.

    python scripts/paper3/pilot.py --part 0 --nparts 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, pipeline as PL  # noqa: E402
from ps3 import search  # noqa: E402

OUT = ROOT / "results/paper3/pilot"
CASES = ["LM_Channel_0180", "LM_Channel_0550", "LM_Channel_1000", "LM_Channel_2000", "LM_Channel_5200"]


def configs():
    rng = np.random.default_rng(3003)
    cs = [search.sample_space(rng, "M4") for _ in range(20)]
    for a in ("M1", "M2", "M3"):
        cs += [search.sample_space(rng, a) for _ in range(2)]
    for i, c in enumerate(cs):
        c["pilot_id"] = i
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    lm = data.load("LM")
    train = {c: PL.prepare(lm[lm.case == c]) for c in CASES}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"pilot_part{a.part}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["pilot_id"] for l in out.read_text().splitlines() if l.strip()}
    cs = [c for c in configs() if c["pilot_id"] % a.nparts == a.part and c["pilot_id"] not in done]
    if a.limit:
        cs = cs[: a.limit]
    seeds = tuple(int(s) for s in a.seeds.split(","))
    for c in cs:
        t0 = time.time()
        r = search.evaluate_candidate(c, train, seeds=seeds)
        with out.open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] id={c['pilot_id']} {c['arch']} status={r['status']} "
              f"f_err={r.get('f_err', float('nan')):.4f} f_gen={r.get('f_gen', float('nan')):.4f} "
              f"wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
