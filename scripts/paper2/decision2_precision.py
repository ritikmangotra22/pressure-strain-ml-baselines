"""Design decisions 2 and 4: initialisation protocol and numerical precision (no model trained).

  1  initialisation correspondence: float32-built-and-cast models give identical initial weights
     across repeated builds and across precisions; native float64 builds do not match them
  2  precision of the property tests: the same weights evaluated in float32 and float64, giving
     the equivariance, trace and isotropic-limit errors each precision can resolve, and the
     float32-float64 prediction difference
  3  cost: wall time of forward + backward passes per full-batch epoch (no optimiser step, so
     no parameter is changed), for M1-M4 and the R0 network, in float32 and float64

    python scripts/paper2/decision2_precision.py --config configs/paper2/decision2.json
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
from ps2.train import F1, ChannelInputNet  # noqa: E402

NAMES = ["M1", "M2", "M3", "M4"]


def params_of(m):
    return torch.cat([p.detach().double().reshape(-1) for p in m.parameters()])


def props(m, b, G, re_t, Qs):
    with torch.no_grad():
        P = m(b, G, re_t)
        nP = T.frob(P).clamp(min=1e-30)
        eq = 0.0
        for Q in Qs:
            Qn = Q.to(b.dtype).expand(len(b), 3, 3)
            Pq = m(T.rotate(Qn, b), T.rotate(Qn, G), re_t)
            eq = max(eq, float((T.frob(Pq - T.rotate(Qn, P)) / nP).max()))
        z = torch.zeros((50, 3, 3), dtype=b.dtype)
        iso = float(T.frob(m(z, z, torch.logspace(0, 4, 50, dtype=b.dtype))).max())
        trace = float((T.tr(P).abs() / nP).max())
    return {"eq_max": eq, "trace_rel_max": trace, "iso_max": iso}, P


def main(cfg):
    torch.set_num_threads(1)
    df = pd.concat([data.load("LM"), data.load("HJ")], ignore_index=True)
    b64, G64 = T.channel_tensors(df, "durbin", cfg["C_T"])
    r64 = torch.tensor(df.Re_t.to_numpy(float), dtype=T.DT)
    b32, G32, r32 = b64.float(), G64.float(), r64.float()
    Qs = list(T.random_rotations(20, 3)) + [T.REFLECT_Y, T.rot_about_z(45)]
    out = {"init": [], "precision": [], "timing": []}

    # 1 + 2
    for name in NAMES:
        for arch in cfg["architectures"]:
            for seed in cfg["seeds"]:
                kw = dict(seed=seed, inv_set=cfg["inv_set"], **arch)
                a = models.build(name, b64, G64, r64, init="float32_cast", **kw)
                a2 = models.build(name, b64, G64, r64, init="float32_cast", **kw)
                nat = models.build(name, b64, G64, r64, init="native", **kw)
                torch.manual_seed(seed)
                ref32 = models.mlp(a.net[0].in_features, a.net[-1].out_features, init="float32_cast",
                                   **{k: v for k, v in arch.items()})
                out["init"].append({
                    "model": name, "seed": seed, **arch,
                    "repeat_build_identical": bool(torch.equal(params_of(a), params_of(a2))),
                    "equals_float32_build": bool(torch.equal(params_of(a.net), params_of(ref32))),
                    "native_float64_max_param_diff": float((params_of(a) - params_of(nat)).abs().max())})
                m64 = a.eval()
                p64, P64 = props(m64, b64, G64, r64, Qs)
                m32 = models.build(name, b64, G64, r64, init="float32_cast", dtype=torch.float32, **kw).eval()
                p32, P32 = props(m32, b32, G32, r32, Qs)
                diff = float((T.frob(P32.double() - P64) / T.frob(P64).clamp(min=1e-30)).max())
                out["precision"].append({"model": name, "seed": seed, **arch,
                                         **{f"f64_{k}": v for k, v in p64.items()},
                                         **{f"f32_{k}": v for k, v in p32.items()},
                                         "f32_vs_f64_pred_rel_max": diff})
        print(f"{name}: init and precision done", flush=True)

    # 3 timing on the largest E1 training set (held-out Re_tau = 180): forward + backward only
    lm = data.load("LM")
    tr = lm[lm.case != "LM_Channel_0180"].reset_index(drop=True)
    bt, Gt = T.channel_tensors(tr, "durbin", cfg["C_T"])
    rt = torch.tensor(tr.Re_t.to_numpy(float), dtype=T.DT)
    target = T.target_tensor(tr)
    w = torch.tensor(tr.w.to_numpy(float) / tr.w.sum(), dtype=T.DT)
    X5 = torch.tensor(((tr[F1] - tr[F1].mean()) / tr[F1].std()).to_numpy(float), dtype=T.DT)
    n_ep = cfg["timing_epochs"]
    for arch in cfg["architectures"]:
        for dt in (torch.float32, torch.float64):
            for name in NAMES + ["R0"]:
                if name == "R0":
                    m = ChannelInputNet(5, dtype=dt, init="float32_cast", **arch)
                    args = (X5.to(dt),)
                else:
                    m = models.build(name, bt, Gt, rt, seed=0, inv_set=cfg["inv_set"], init="float32_cast", dtype=dt, **arch)
                    args = (bt.to(dt), Gt.to(dt), rt.to(dt))
                tg, ww = target.to(dt), w.to(dt)
                times = []
                for rep in range(cfg["timing_repeats"]):
                    t0 = time.perf_counter()
                    for _ in range(n_ep):
                        m.zero_grad(set_to_none=True)
                        loss = (ww * ((m(*args) - tg) ** 2).sum(dim=(1, 2))).sum()
                        loss.backward()
                    times.append((time.perf_counter() - t0) / n_ep)
                out["timing"].append({"model": name, "dtype": str(dt).split(".")[-1], **arch,
                                      "n_points": len(tr), "ms_per_epoch_median": 1e3 * float(np.median(times))})
                print(out["timing"][-1], flush=True)

    res = Path(ROOT / cfg["out_dir"])
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out["init"]).to_csv(res / "init_correspondence.csv", index=False)
    prec = pd.DataFrame(out["precision"])
    prec.to_csv(res / "precision_properties.csv", index=False)
    tim = pd.DataFrame(out["timing"])
    tim.to_csv(res / "timing_forward_backward.csv", index=False)
    summary = {
        "init_all_repeat_identical": bool(pd.DataFrame(out["init"]).repeat_build_identical.all()),
        "init_all_equal_float32_build": bool(pd.DataFrame(out["init"]).equals_float32_build.all()),
        "native_float64_min_param_diff": float(pd.DataFrame(out["init"]).native_float64_max_param_diff.min()),
        "precision_max_by_model": prec.groupby("model")[[c for c in prec.columns if c.startswith(("f64_", "f32_"))]].max().to_dict(),
        "timing_ratio_f64_over_f32": (tim.pivot_table(index=["model", "n_layers", "width"], columns="dtype", values="ms_per_epoch_median")
                                      .assign(ratio=lambda x: x.float64 / x.float32)["ratio"].round(2).reset_index().to_dict("records")),
        "timing_ratio_to_R0_float32": (tim[tim.dtype == "float32"].assign(
            ratio=lambda x: x.ms_per_epoch_median / x.groupby(["n_layers", "width"]).ms_per_epoch_median.transform(
                lambda s: float(s[tim.loc[s.index, "model"] == "R0"].iloc[0])))[["model", "n_layers", "width", "ratio"]]
                                       .round(2).to_dict("records")),
        "config": cfg}
    (res / "decision2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(json.loads(Path(ap.parse_args().config).read_text()))
