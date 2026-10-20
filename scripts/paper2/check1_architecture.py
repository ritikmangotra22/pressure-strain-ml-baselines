"""Section H check 1: architectural properties of the untrained Paper 2 models M1-M4.

Question: do the four architectures have, by construction, exactly the properties the 2 x 2
factorial design assigns to them, and lack the ones it does not?

  property            M1    M2    M3    M4      registered criterion (Paper 2 doc, Section H item 1)
  O(3) equivariance   no    yes   no    yes     M2, M4: max relative error <= 1e-10; M1 must fail
  zero trace          no    no    yes   yes     M3, M4: exact (we use max |tr|/||Pi|| <= 1e-12)
  isotropic limit     no    no    yes   yes     M3, M4: exact (we use max ||Pi|| <= 1e-12)

"M1 must fail" is operationalised before running as: max relative equivariance error > 1e-3.
The M2 and M3 columns marked "no" are not acceptance criteria; they are reported because the
factorial design is only clean if they also hold.

Inputs are the real DNS states of the released Paper 1 tables (1691 LM + 684 HJ points) with
the Durbin-bounded time scale (C_T = 6, provisional). Networks are untrained (random weights);
equivariance, trace and the isotropic limit are architectural, so they must hold for ANY
weights, which is why several seeds, depths and activations are tested.

    python scripts/paper2/check1_architecture.py --config configs/paper2/check1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, models  # noqa: E402
from ps2 import tensors as T  # noqa: E402


def evaluate(model, b, G, re_t, Qs):
    """Return dict of property diagnostics for one model."""
    with torch.no_grad():
        P = model(b, G, re_t)
        nP = T.frob(P)
        floor = 1e-12 * float(nP.max())
        out = {"min ||Pi||": float(nP.min()), "max ||Pi||": float(nP.max())}
        eq = {}
        for name, Q in Qs.items():
            if Q.dim() == 2:
                Q = Q.expand(len(b), 3, 3)
                Pq = model(T.rotate(Q, b), T.rotate(Q, G), re_t)
                err = T.frob(Pq - T.rotate(Q, P)) / torch.clamp(nP, min=floor)
                eq[name] = (float(err.max()), float(err.mean()))
            else:  # batch of random rotations: apply each to all points
                worst, tot, cnt = 0.0, 0.0, 0
                for q in Q:
                    qq = q.expand(len(b), 3, 3)
                    Pq = model(T.rotate(qq, b), T.rotate(qq, G), re_t)
                    err = T.frob(Pq - T.rotate(qq, P)) / torch.clamp(nP, min=floor)
                    worst, tot, cnt = max(worst, float(err.max())), tot + float(err.sum()), cnt + len(err)
                eq[name] = (worst, tot / cnt)
        out["eq_max"] = max(v[0] for v in eq.values())
        out.update({f"eq_max[{k}]": v[0] for k, v in eq.items()})
        out.update({f"eq_mean[{k}]": v[1] for k, v in eq.items()})
        out["trace_rel_max"] = float((T.tr(P).abs() / torch.clamp(nP, min=floor)).max())
        out["asym_max"] = float((P - P.transpose(1, 2)).abs().max())
        rt = torch.logspace(np.log10(float(re_t.min()) + 1e-12), np.log10(float(re_t.max())), 100, dtype=T.DT)
        z = torch.zeros((len(rt), 3, 3), dtype=T.DT)
        out["iso_limit_max"] = float(T.frob(model(z, z, rt)).max())
    return out


def main(cfg):
    t0 = time.time()
    torch.set_num_threads(1)
    frames = []
    for w in ("LM", "HJ"):
        d = data.load(w)
        d["db"] = w
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    b, G = T.channel_tensors(df, cfg["time_scale"], cfg["C_T"])
    re_t = torch.tensor(df["Re_t"].to_numpy(float), dtype=T.DT)
    Qs = {"random_SO3": T.random_rotations(cfg["n_random_rotations"], cfg["rotation_seed"]),
          "reflect_y": T.REFLECT_Y}
    for deg in cfg["z_rotations_deg"]:
        Qs[f"rot_z_{deg}"] = T.rot_about_z(deg)

    # invariants must be unchanged by every transformation (sanity check of the inputs)
    S, W = T.sym_skew(G)
    I0 = T.invariants(b, S, W, T.INVARIANT_SETS["E4+bW2"])
    inv_err = 0.0
    for name, Q in Qs.items():
        for q in (Q if Q.dim() == 3 else [Q])[:50]:
            qq = q.expand(len(b), 3, 3)
            Sq, Wq = T.sym_skew(T.rotate(qq, G))
            Iq = T.invariants(T.rotate(qq, b), Sq, Wq, T.INVARIANT_SETS["E4+bW2"])
            inv_err = max(inv_err, float(((Iq - I0).abs() / I0.abs().clamp(min=1e-12)).max()))

    rows = []
    for name in ("M1", "M2", "M3", "M4"):
        for arch in cfg["architectures"]:
            for seed in cfg["seeds"]:
                m = models.build(name, b, G, re_t, seed=seed, **arch).eval()
                r = {"model": name, "seed": seed, **arch, **evaluate(m, b, G, re_t, Qs)}
                rows.append(r)
                print(f"{name} seed={seed} {arch}: eq_max={r['eq_max']:.2e} trace={r['trace_rel_max']:.2e} "
                      f"iso={r['iso_limit_max']:.2e} asym={r['asym_max']:.1e}", flush=True)
    res = pd.DataFrame(rows)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "check1_architecture.csv", index=False)

    c = cfg["criteria"]
    agg = res.groupby("model")[["eq_max", "trace_rel_max", "iso_limit_max", "asym_max"]].max()
    mins = res.groupby("model")[["eq_max", "trace_rel_max", "iso_limit_max"]].min()
    verdict = {
        "M2 equivariant (<= %g)" % c["eq_tol"]: bool(agg.loc["M2", "eq_max"] <= c["eq_tol"]),
        "M4 equivariant (<= %g)" % c["eq_tol"]: bool(agg.loc["M4", "eq_max"] <= c["eq_tol"]),
        "M3 trace-free (<= %g)" % c["trace_tol"]: bool(agg.loc["M3", "trace_rel_max"] <= c["trace_tol"]),
        "M4 trace-free (<= %g)" % c["trace_tol"]: bool(agg.loc["M4", "trace_rel_max"] <= c["trace_tol"]),
        "M3 isotropic limit (<= %g)" % c["iso_tol"]: bool(agg.loc["M3", "iso_limit_max"] <= c["iso_tol"]),
        "M4 isotropic limit (<= %g)" % c["iso_tol"]: bool(agg.loc["M4", "iso_limit_max"] <= c["iso_tol"]),
        "M1 fails equivariance in every run (> %g)" % c["m1_fail_eq"]: bool(mins.loc["M1", "eq_max"] > c["m1_fail_eq"]),
        "all outputs symmetric (<= 1e-14)": bool(agg["asym_max"].max() <= 1e-14),
    }
    complementary = {
        "M2 NOT trace-free in every run": bool(mins.loc["M2", "trace_rel_max"] > c["m1_fail_eq"]),
        "M2 NOT isotropic-limit exact in every run": bool(mins.loc["M2", "iso_limit_max"] > c["m1_fail_eq"]),
        "M3 NOT equivariant in every run": bool(mins.loc["M3", "eq_max"] > c["m1_fail_eq"]),
        "M1 NOT trace-free / NOT iso-limit in every run": bool(min(mins.loc["M1", "trace_rel_max"], mins.loc["M1", "iso_limit_max"]) > c["m1_fail_eq"]),
    }
    summary = {"n_points": len(df), "n_runs_per_model": int(len(res) / 4),
               "max_relative_invariant_change_under_transforms": inv_err,
               "max_over_runs": agg.to_dict(), "min_over_runs": mins.to_dict(),
               "acceptance": verdict, "complementary": complementary,
               "PASS": all(verdict.values()), "runtime_s": round(time.time() - t0, 1), "config": cfg}
    (out / "check1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("acceptance", "complementary", "PASS",
                                               "max_relative_invariant_change_under_transforms")}, indent=2))
    print(agg.to_string(float_format=lambda v: f"{v:.2e}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(json.loads(Path(ap.parse_args().config).read_text()))
