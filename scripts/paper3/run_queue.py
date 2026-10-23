"""Run the protocol_v1 primary jobs on two workers, within one session's time budget.

Each worker takes the next job that is not complete and not taken by the other worker, runs it
until it completes or the session budget is spent, then takes the next. Order: seed first, so that
every strategy has results for seed 3001 before any strategy starts seed 3002.

    python scripts/paper3/run_queue.py --workers 2 --session-minutes 100
    python scripts/paper3/run_queue.py --list
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/paper3"))
import run_search as RS  # noqa: E402

OUT = ROOT / "results/paper3/search"


def jobs():
    J = RS.PROTO["jobs"]["primary"]
    out = []
    for seed in J["seeds"]:
        for arm, strategies in (("A", J["arm_A"]), ("B", J["arm_B"])):
            for s in strategies:
                out.append((s, arm, seed))
    A = RS.PROTO["jobs"].get("ablations")            # deviation 4: run after every primary job
    if A:
        for seed in A["seeds"]:
            for s in A["strategies"]:
                out.append((s, A["arm"], seed))
    return out


def worker(wid, deadline, lock_dir, skip=()):
    while time.time() < deadline - 60:
        job = None
        for s, arm, seed in jobs():
            name = RS.job_name(s, arm, seed)
            if s in skip or (OUT / name / "status.json").exists():
                continue
            lock = lock_dir / f"{name}.lock"
            try:
                lock.touch(exist_ok=False)
            except FileExistsError:
                continue
            job = (s, arm, seed, lock)
            break
        if job is None:
            return
        s, arm, seed, lock = job
        try:
            mins = (deadline - time.time()) / 60
            res = RS.run(s, arm, seed, out=OUT, max_minutes=mins)
            print(f"[worker {wid}] {RS.job_name(s, arm, seed)} -> {res}", flush=True)
        finally:
            lock.unlink(missing_ok=True)
        if res == "paused":
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--session-minutes", type=float, default=100.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated strategies to leave for later (order only)")
    a = ap.parse_args()
    if a.list:
        for s, arm, seed in jobs():
            n = RS.job_name(s, arm, seed)
            st = "complete" if (OUT / n / "status.json").exists() else ("started" if (OUT / n).exists() else "-")
            print(f"{n:28s} {st}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock_dir = OUT / ".locks"
    lock_dir.mkdir(exist_ok=True)
    for p in lock_dir.glob("*.lock"):          # locks from a killed session
        p.unlink()
    deadline = time.time() + a.session_minutes * 60
    ps = [mp.Process(target=worker, args=(i, deadline, lock_dir, tuple(x for x in a.skip.split(",") if x))) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    done = sum((OUT / RS.job_name(*j) / "status.json").exists() for j in jobs())
    print(json.dumps({"session_end": time.strftime("%Y-%m-%dT%H:%M:%S"), "jobs_complete": done, "jobs_total": len(jobs())}))


if __name__ == "__main__":
    main()
