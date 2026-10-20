"""Section H check 2: re-run the basis-adequacy and invariant-uniqueness diagnostics of
Section E (E.2, E.3, E.4) inside the Paper 2 code, save them, and compare every number with
the value printed in the Paper 2 first deliverable.

Part A  basis adequacy and conditioning (E.2 table; E.3 rank/span claims), LM and HJ,
        k/eps and Durbin-bounded time scale, plus sensitivity to the provisional C_T.
Part B  invariant uniqueness (E.4): analytic swap identity on every DNS state, the twin search
        of E.4 on synthetic states (as originally done) and on real DNS states, and a check of
        the degenerate set where the E.4 set loses uniqueness.

    python scripts/paper2/check2_basis_invariants.py --config configs/paper2/check2.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ps2 import data, diagnostics as D  # noqa: E402
from ps2 import tensors as T  # noqa: E402

REGIONS = ["viscous", "buffer", "log", "outer"]


def part_a(cfg, out):
    rows, per_point = [], []
    runs = [(db, sc, cfg["C_T"]) for db in ("LM", "HJ") for sc in ("k/eps", "durbin")]
    runs += [("LM", "durbin", c) for c in cfg["C_T_sensitivity"] if c != cfg["C_T"]]
    runs += [("HJ", "durbin", c) for c in cfg["C_T_sensitivity"] if c != cfg["C_T"]]
    for db, sc, ct in runs:
        df = data.load(db)
        r = D.adequacy(df, sc, ct)
        pp = df[["case", "y_plus", "region", "w"]].copy()
        for k, v in r.items():
            pp[k] = v
        pp["db"], pp["scheme"], pp["C_T"] = db, sc, ct
        per_point.append(pp)
        for reg in REGIONS + ["all"]:
            g = pp if reg == "all" else pp[pp.region == reg]
            i = g.g_norm.idxmax()
            rows.append({"db": db, "scheme": sc, "C_T": ct if sc == "durbin" else np.nan, "region": reg,
                         "n": len(g), "min_rank": int(g["rank"].min()), "max_res_free": g.res_free.max(),
                         "median_g": g.g_norm.median(), "max_g": g.g_norm.max(),
                         "max_g_case": g.loc[i, "case"], "max_g_yplus": g.loc[i, "y_plus"],
                         "n_g_gt_100": int((g.g_norm > 100).sum()),
                         "yplus_range_g_gt_100": (float(g[g.g_norm > 100].y_plus.min()), float(g[g.g_norm > 100].y_plus.max()))
                         if (g.g_norm > 100).any() else None,
                         "wmean_res_box10": float(np.average(g.res_box10, weights=g.w)),
                         "wmean_res_box100": float(np.average(g.res_box100, weights=g.w)),
                         "max_res_box10": g.res_box10.max(), "min_s3_rel": g.s3.min(),
                         "max|res_free - trace_rel/sqrt3|": float((g.res_free - g.trace_rel / np.sqrt(3)).abs().max())})
        print(f"A: {db} {sc} C_T={ct} done", flush=True)
    pd.concat(per_point).to_csv(out / "check2_adequacy_points.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "check2_adequacy_summary.csv", index=False)

    # rank-3 subsets at every DNS point (E.3: "39 of 56 ... at generic states")
    subsets = {db: [list(s) for s in D.spanning_subsets(data.load(db), "durbin", cfg["C_T"])] for db in ("LM", "HJ")}
    return summ, subsets


def part_b(cfg, out):
    import torch
    res = {}
    names4, names5, names6 = (T.INVARIANT_SETS[k] for k in ("four", "E4", "E4+bW2"))
    # B1 analytic swap identity on every DNS state: (b11, b22) -> (b22, b11), same b12, eta
    sw = []
    for db in ("LM", "HJ"):
        df = data.load(db)
        b, G = T.channel_tensors(df, "durbin", cfg["C_T"])
        P = torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=T.DT).expand(len(b), 3, 3)
        b_sw = T.rotate(P, b)                       # swaps b11 and b22, keeps b12, b33
        S, W = T.sym_skew(G)
        Sp, Wp = T.rotate(P, S), T.rotate(P, W)     # what the same reflection does to S and W
        I = {k: T.invariants(b, S, W, T.INVARIANT_SETS[k]).numpy() for k in ("four", "E4", "E4+bW2")}
        Isw = {k: T.invariants(b_sw, S, W, T.INVARIANT_SETS[k]).numpy() for k in ("four", "E4", "E4+bW2")}
        d = np.abs(df.b11 - df.b22).to_numpy()
        sw.append({"db": db, "n": len(df),
                   "P S P^T == S (max diff)": float((Sp - S).abs().max()),
                   "P W P^T == -W (max diff)": float((Wp + W).abs().max()),
                   "four invariants: max |I(swap) - I| / max|I|": float(np.abs(Isw["four"] - I["four"]).max() / np.abs(I["four"]).max()),
                   "tr(bSW): min |I(swap)-I| / |I| where |b11-b22|>1e-3": float(
                       (np.abs(Isw["E4"][:, 4] - I["E4"][:, 4]) / np.maximum(np.abs(I["E4"][:, 4]), 1e-300))[(d > 1e-3) & (df.eta.to_numpy() > 0)].min()),
                   "swapped states realizable": int(sum(D.realizable([r.b22, r.b11, r.b12, r.eta]) for r in df.itertuples())),
                   "states with |b11-b22| <= 1e-3": int((d <= 1e-3).sum())})
    res["B1_swap_identity"] = sw

    # B2 twin search: synthetic states (as in E.4) and real DNS states
    rng = np.random.default_rng(cfg["twin_seed"])
    lm = data.load("LM")
    synth, dns = [], []
    while len(synth) < cfg["n_twin_states"]:
        x = np.array([rng.uniform(0.05, 0.6), rng.uniform(-0.33, -0.05), rng.uniform(-0.17, -0.01), rng.uniform(0.5, 20)])
        if D.realizable(x):
            synth.append(x)
    idx = rng.choice(len(lm), cfg["n_twin_states"], replace=False)
    dns = [lm.loc[i, ["b11", "b22", "b12", "eta"]].to_numpy(float) for i in idx]
    twins = []
    for label, states in (("synthetic", synth), ("DNS_LM", dns)):
        for nm, names in (("four", names4), ("E4", names5), ("E4+bW2", names6)):
            r2 = np.random.default_rng(cfg["twin_seed"] + 1)
            found, swapped = 0, 0
            for x in states:
                z = D.find_twin(x, names, r2, n_starts=cfg["twin_starts"])
                if z is not None:
                    found += 1
                    swapped += int(np.allclose(z[:3], [x[1], x[0], x[2]], atol=1e-6))
            twins.append({"states": label, "invariant_set": nm, "n_states": len(states),
                          "twins_found": found, "of_which_b11_b22_swap": swapped})
            print(f"B2: {label} {nm}: {found}/{len(states)} twins ({swapped} are the swap)", flush=True)
    res["B2_twin_search"] = twins

    # B3 degenerate set of the E4 set: tr b^3 = (3/4) s (d^2 + 4 b12^2 - s^2), s = b11+b22, d = b22-b11
    deg = []
    for db in ("LM", "HJ"):
        df = data.load(db)
        s, dd, b12 = (df.b11 + df.b22).to_numpy(), (df.b22 - df.b11).to_numpy(), df.b12.to_numpy()
        formula = 0.75 * s * (dd ** 2 + 4 * b12 ** 2 - s ** 2)
        q = (dd ** 2 + 4 * b12 ** 2 - s ** 2) / (dd ** 2 + 4 * b12 ** 2 + s ** 2)
        sign_changes = int(sum((np.diff(np.sign(q[(df.case == c).to_numpy()])) != 0).sum() for c in df.case.unique()))
        deg.append({"db": db, "max|III_b - formula|": float(np.abs(df.III_b.to_numpy() - formula).max()),
                    "min |q|": float(np.abs(q).min()), "q range": [float(q.min()), float(q.max())],
                    "sign changes of q along profiles": sign_changes,
                    "min |s|": float(np.abs(s).min())})
    res["B3_degenerate_set"] = deg
    # explicit twin on the degenerate set (E4 not unique there; tr(bW^2) separates)
    d0, b12_0 = 0.1, -0.05
    s0 = np.sqrt(d0 ** 2 + 4 * b12_0 ** 2)
    xa = np.array([(s0 - d0) / 2, (s0 + d0) / 2, b12_0, 5.0])
    xb = np.array([(-s0 - d0) / 2, (-s0 + d0) / 2, b12_0, 5.0])
    res["B3_example"] = {"state_a": xa.tolist(), "state_b": xb.tolist(),
                         "realizable": [bool(D.realizable(xa)), bool(D.realizable(xb))],
                         "E4 invariants equal": bool(np.allclose(D.channel_state_invariants(xa, names5), D.channel_state_invariants(xb, names5), atol=1e-14)),
                         "tr(bW^2) a, b": [float(D.channel_state_invariants(xa, ["tr_bW2"])[0]), float(D.channel_state_invariants(xb, ["tr_bW2"])[0])]}
    return res


def main(cfg):
    t0 = time.time()
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    summ, subsets = part_a(cfg, out)
    b = part_b(cfg, out)
    result = {"part_A_subsets_rank3_everywhere": {k: {"count": len(v), "subsets": v} for k, v in subsets.items()},
              **b, "runtime_s": round(time.time() - t0, 1), "config": cfg}
    (out / "check2_results.json").write_text(json.dumps(result, indent=2, default=float))
    cols = ["db", "scheme", "C_T", "region", "n", "min_rank", "max_res_free", "max_g", "max_g_case", "max_g_yplus",
            "n_g_gt_100", "yplus_range_g_gt_100", "wmean_res_box10", "wmean_res_box100"]
    print(summ[cols].to_string(float_format=lambda v: f"{v:.3g}"))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(json.loads(Path(ap.parse_args().config).read_text()))
