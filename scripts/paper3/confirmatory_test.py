"""Paper 3 single confirmatory test (protocol_v1 data.confirmatory_test). RUN ONCE.

Evaluates every refitted selected model (10 seeds), M0 and the classical references on
  * Moser, Kim & Mansour Re_tau 395 and 590 (confirmatory; never seen by any model), and
  * Hoyas & Jimenez (comparability; previously inspected in Paper 2),
a priori (Paper 2 metrics: E_Pi, Paper 1 component metrics, regions, structural diagnostics) and,
on the two MKM cases, a posteriori (conditioning and frozen solve, as scripts/paper3/aposteriori.py).

Refuses to run unless ALL of the following hold (checked before the confirmatory table is opened):
  1. HEAD is the commit tagged paper3-selection-v1 and src/, scripts/, configs/ have no changes;
  2. protocol_v1.json approvals.supervisor.status starts with "APPROVED" and has a date;
  3. results/paper3/final/selected.json, every refit checkpoint and classical.json exist;
  4. results/paper3/confirmatory/RUN_LOCK.json does not exist (the test has never been started);
  5. the MKM and HJ tables match their frozen SHA-256 hashes.
The lock file is written before the first model is evaluated, so an interrupted run cannot be
silently repeated; a restart after an interruption must be logged as a deviation.

    python scripts/paper3/confirmatory_test.py --check     # preconditions only, reads no test data
    python scripts/paper3/confirmatory_test.py              # the confirmatory test
    python scripts/paper3/confirmatory_test.py --dry-run    # code path on LM 550/2000 stand-ins
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import final_common as F
import aposteriori as AP
from ps import classical as C, metrics as MET
from ps2 import data, pipeline as PL
from ps3 import rsm1d
from ps3.io_mkm import load_mkm_case

TAG = "paper3-selection-v1"
MKM_TABLE = F.ROOT / "results/paper3/mkm_data/processed_table_mkm.csv"
MKM_RAW = F.ROOT / "data/raw/moser_kim_mansour_1999"
OUT = F.ROOT / "results/paper3/confirmatory"
Y = ["Pi11", "Pi22", "Pi33", "Pi12"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preconditions(final: Path) -> list:
    bad = []
    tag_commit = F.git("rev-list", "-n", "1", TAG)
    if not tag_commit:
        bad.append(f"tag {TAG} does not exist")
    elif F.git("rev-parse", "HEAD") != tag_commit:
        bad.append(f"HEAD is not the {TAG} commit")
    if F.git("status", "--porcelain", "--", "src", "scripts", "configs"):
        bad.append("uncommitted changes in src/, scripts/ or configs/")
    sup = F.PROTO["approvals"]["supervisor"]
    if not (str(sup.get("status", "")).upper().startswith("APPROVED") and sup.get("date")):
        bad.append("supervisor approval not recorded in protocol_v1.json")
    if not (final / "selected.json").exists() or not (final / "classical.json").exists():
        bad.append("selected.json or classical.json missing")
    else:
        sel = json.loads((final / "selected.json").read_text())
        if len(sel) < F.N_JOBS + 1:
            bad.append(f"selected.json has {len(sel)} models, expected {F.N_JOBS + 1}")
        miss = [(m, s) for m in sel if sel[m]["config"] is not None for s in F.REFIT_SEEDS
                if not F.ckpt_path(final, m, s).exists()]
        if miss:
            bad.append(f"{len(miss)} refit checkpoints missing, e.g. {miss[:3]}")
    if (OUT / "RUN_LOCK.json").exists():
        bad.append("RUN_LOCK.json exists: the confirmatory test has already been started")
    if sha(MKM_TABLE) != F.PROTO["data"]["confirmatory_test"]["sha256"]:
        bad.append("MKM table hash differs from the frozen value")
    if sha(data.TABLES["HJ"]) != F.PROTO["data"]["comparability_test"]["sha256"]:
        bad.append("HJ table hash differs from the frozen value")
    return bad


def test_sets(dry: bool):
    """{case: (Split builder by c_t, raw ChannelCase or None)}."""
    if dry:
        lm = data.load("LM")
        tab = {c: lm[lm.case == c] for c in ("LM_Channel_0550", "LM_Channel_2000")}
        from ps3 import search
        raw = {c: search._channel_case(c) for c in tab}
        return tab, raw
    mkm = pd.read_csv(MKM_TABLE)
    hj = data.load("HJ")
    tab = {c: mkm[mkm.case == c] for c in sorted(mkm.case.unique())}
    tab.update({c: hj[hj.case == c] for c in sorted(hj.case.unique())})
    raw = {}
    for re in (395, 590):
        cs = load_mkm_case(MKM_RAW, re)
        raw[cs.name] = rsm1d.case_from_lm(cs)
    assert set(raw) <= set(tab), (sorted(raw), sorted(tab))
    return tab, raw


def predictions_frame(df, P_hat4, model, seed):
    pr = df[["case", "y_plus", "y_delta", "region", "w"] + Y].reset_index(drop=True).copy()
    for j, c in enumerate(Y):
        pr[c + "_hat"] = P_hat4[:, j]
    pr.insert(0, "seed", seed)
    pr.insert(0, "model", model)
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--final", type=Path, default=F.FINAL, help="refit directory (dry runs only)")
    a = ap.parse_args()
    if a.dry_run:
        out = a.final / "confirmatory_dryrun"
    else:
        if a.final != F.FINAL:
            sys.exit("--final may only be changed for a dry run")
        bad = preconditions(a.final)
        if bad or a.check:
            print("preconditions:", "all met" if not bad else "", *[f"\n  - {b}" for b in bad])
            sys.exit(1 if bad else 0)
        out = OUT
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((a.final / "selected.json").read_text())
    lock = {"started": datetime.datetime.now().isoformat(), "head": F.git("rev-parse", "HEAD"), "tag": TAG,
            "dry_run": a.dry_run, "mkm_sha256": sha(MKM_TABLE), "hj_sha256": sha(data.TABLES["HJ"]),
            "n_models": len(sel)}
    (out / "RUN_LOCK.json").write_text(json.dumps(lock, indent=1))
    tab, raw = test_sets(a.dry_run)
    print("test cases:", sorted(tab), "| a posteriori on", sorted(raw), flush=True)
    rows, ap_rows, preds = [], [], []
    four = lambda A: np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1]], 1)
    for mid, spec in sel.items():
        if spec["config"] is None:
            continue
        c_t = spec["config"]["c_t"]
        train = F.lm_split(c_t)
        splits = {k: PL.prepare(v, "durbin", c_t) for k, v in tab.items()}
        for s in F.REFIT_SEEDS:
            if a.dry_run and not F.ckpt_path(a.final, mid, s).exists():
                continue                      # dry runs use a partial set of refits
            fitted, ck = F.load_fitted(F.ckpt_path(a.final, mid, s), train)
            for case, te in splits.items():
                met, P_hat = PL.evaluate(fitted, te)
                rows.append({"model": mid, "seed": s, "case": case, **met})
                preds.append(predictions_frame(te.df, four(P_hat), mid, s))
            for case, cc in raw.items():
                ap_rows.append({"model": mid, "seed": s, "case": case,
                                **AP.assess(cc, rsm1d.learned(fitted, c_t), s in AP.SOLVE_SEEDS)})
        if not any(r["model"] == mid for r in rows):
            continue
        print(f"{mid}: mean E_Pi over seeds", {k: round(float(np.mean([r['E_Pi'] for r in rows if r['model'] == mid and r['case'] == k])), 4)
                                                for k in splits}, flush=True)
    for name, v in json.loads((a.final / "classical.json").read_text()).items():
        cl = C.ClassicalClosure(v["kind"], "published")
        cl.coef_ = np.array(list(v["coefs"].values()), float)
        for case, df in tab.items():
            y_hat, y, w = cl.predict(df), df[Y].to_numpy(), df.w.to_numpy()
            p1 = MET.all_metrics(y_hat, y, w, df.region.to_numpy())
            rows.append({"model": f"classical:{name}", "seed": -1, "case": case, "E_Pi": MET.e_pi(y_hat, y, w),
                         **{f"p1_{k}": x for k, x in p1.items()}})
            preds.append(predictions_frame(df, y_hat, f"classical:{name}", -1))
        for case, cc in raw.items():
            ap_rows.append({"model": f"classical:{name}", "seed": -1, "case": case,
                            **AP.assess(cc, rsm1d.classical(v["kind"], v["coefs"]), True)})
    for case, cc in raw.items():
        ap_rows.append({"model": "anchored_oracle", "seed": -1, "case": case, **AP.assess(cc, rsm1d.anchored_oracle(cc), True)})
    pd.DataFrame(rows).to_csv(out / "a_priori_metrics.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv.gz", index=False)
    (out / "a_posteriori.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in ap_rows))
    lock["completed"] = datetime.datetime.now().isoformat()
    (out / "RUN_LOCK.json").write_text(json.dumps(lock, indent=1))
    print("done:", out)


if __name__ == "__main__":
    main()
