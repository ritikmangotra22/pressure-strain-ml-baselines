"""Paper 3 classical reference closures (training data only).

SSG, LRR-IP and LRR-IP + Gibson-Launder wall reflection (src/ps/classical.py, verified in Paper 1),
each with (i) published coefficients and (ii) coefficients refitted by weighted least squares on all
five Lee & Moser cases (each case weighted equally, Frobenius multiplicities, as the networks are
trained). For the refitted forms the same inner leave-one-Re_tau-out procedure as the search gives
f_err and f_gen, so they can be placed on the search's objective scale.

    python scripts/paper3/classical_refs.py      ->  results/paper3/final/classical.json
"""
import json

import numpy as np

import final_common as F
from ps import classical as C, metrics as MET
from ps2 import data

KINDS = ("ssg", "lrr_ip", "lrr_ip_gl")
Y = ["Pi11", "Pi22", "Pi33", "Pi12"]
M0 = json.loads((F.ROOT / "results/paper3/reference_m0/m0_inner.json").read_text())


def fit(kind, df):
    w = df.groupby("case")["w"].transform(lambda x: x / x.sum()).to_numpy()   # each case weighted equally
    return C.ClassicalClosure(kind, "refit").fit(df, df[Y].to_numpy(), w, MET.M)


def scores(cl, lm):
    return {c: MET.e_pi(cl.predict(lm[lm.case == c]), lm[lm.case == c][Y].to_numpy(), lm[lm.case == c].w.to_numpy())
            for c in F.CASES_LM}


def main():
    lm = data.load("LM")
    lm = lm[lm.case.isin(F.CASES_LM)].reset_index(drop=True)
    out = {}
    for kind in KINDS:
        pub = C.ClassicalClosure(kind, "published")
        ref = fit(kind, lm)
        inner = {}
        for vc in F.CASES_LM:
            cl = fit(kind, lm[lm.case != vc])
            v = lm[lm.case == vc]
            inner[vc] = MET.e_pi(cl.predict(v), v[Y].to_numpy(), v.w.to_numpy())
        rel = [inner[c] / M0["fold_E_Pi_mean"][c] for c in F.CASES_LM]
        for mode, cl in (("published", pub), ("refit", ref)):
            out[f"{kind}_{mode}"] = {"kind": kind, "mode": mode,
                                     "coefs": dict(zip(cl.coef_names_, map(float, cl.coef_))),
                                     "in_sample_E_Pi": scores(cl, lm)}
        out[f"{kind}_refit"]["inner_fold_E_Pi"] = inner
        out[f"{kind}_refit"]["f_err"] = float(np.mean(list(inner.values())))
        out[f"{kind}_refit"]["f_gen"] = float(max(rel))
        out[f"{kind}_published"]["f_err_on_LM"] = float(np.mean(list(out[f"{kind}_published"]["in_sample_E_Pi"].values())))
    F.FINAL.mkdir(parents=True, exist_ok=True)
    (F.FINAL / "classical.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {c: round(x, 3) for c, x in v["coefs"].items()},
              "f_err" if "f_err" in v else "mean E_Pi(LM)", round(v.get("f_err", v.get("f_err_on_LM", np.nan)), 4))


if __name__ == "__main__":
    main()
