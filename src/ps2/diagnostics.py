"""Data diagnostics for the tensor basis and the invariant set (Section E.2-E.4, check 2).

Nothing here fits a model. The basis-adequacy test projects each DNS tensor Pi/eps onto the
span of the basis tensors evaluated at the same point; the residual is a lower bound on the
error of ANY model that uses that basis, and the size of the coefficients needed says how hard
the representation is for a network with bounded outputs.

Tensors are flattened to 9 entries, so Euclidean norms equal Frobenius norms (the 12 and 21
entries are both counted, as in Paper 1's E_Pi).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from . import tensors as T


def basis_matrices(df, scheme, c_t=6.0):
    """A: (n, 9, 8) basis matrices; p: (n, 9) flattened DNS Pi/eps."""
    b, G = T.channel_tensors(df, scheme, c_t)
    S, W = T.sym_skew(G)
    B = T.basis_dev(b, S, W)
    A = np.stack([x.reshape(-1, 9).numpy() for x in B], axis=2)
    p = T.target_tensor(df).reshape(-1, 9).numpy()
    return A, p


def adequacy(df, scheme, c_t=6.0, bounds=(10.0, 100.0)):
    """Per-point projection diagnostics. Returns a dict of arrays (one entry per point)."""
    A, p = basis_matrices(df, scheme, c_t)
    n = len(p)
    out = {k: np.empty(n) for k in ["rank", "s3", "res_free", "g_norm"] + [f"res_box{int(B)}" for B in bounds]}
    for i in range(n):
        Ai, pi = A[i], p[i]
        sv = np.linalg.svd(Ai, compute_uv=False)
        tol = max(Ai.shape) * np.finfo(float).eps * sv[0]
        out["rank"][i] = int((sv > tol).sum())
        out["s3"][i] = sv[2] / sv[0]
        g, *_ = np.linalg.lstsq(Ai, pi, rcond=None)          # minimum-norm solution
        pn = np.linalg.norm(pi)
        out["res_free"][i] = np.linalg.norm(Ai @ g - pi) / pn
        out["g_norm"][i] = np.linalg.norm(g)
        for B in bounds:
            sol = lsq_linear(Ai, pi, bounds=(-B, B), method="bvls")
            out[f"res_box{int(B)}"][i] = np.linalg.norm(Ai @ sol.x - pi) / pn
    out["trace_rel"] = np.abs(p[:, 0] + p[:, 4] + p[:, 8]) / np.linalg.norm(p, axis=1)
    return out


def spanning_subsets(df, scheme, c_t=6.0, k=3):
    """Which k-element subsets of the 8-element basis have rank 3 at EVERY point of df."""
    A, _ = basis_matrices(df, scheme, c_t)
    good = []
    for c in combinations(range(8), k):
        ok = True
        for i in range(len(A)):
            sv = np.linalg.svd(A[i][:, list(c)], compute_uv=False)
            if sv[-1] <= 1e-10 * max(sv[0], 1e-300):
                ok = False
                break
        if ok:
            good.append(tuple(j + 1 for j in c))
    return good


# ---------------------------------------------------------------------------------------------
# Invariant uniqueness
# ---------------------------------------------------------------------------------------------
def channel_state_invariants(x, names, tau_ratio=1.0):
    """Invariants of the channel state x = (b11, b22, b12, eta) (time-scale ratio folded in)."""
    import torch
    b11, b22, b12, eta = x
    b = torch.tensor([[[b11, b12, 0.0], [b12, b22, 0.0], [0.0, 0.0, -b11 - b22]]], dtype=T.DT)
    G = torch.zeros((1, 3, 3), dtype=T.DT)
    G[0, 0, 1] = eta * tau_ratio
    S, W = T.sym_skew(G)
    return T.invariants(b, S, W, names).numpy()[0]


def realizable(x):
    b11, b22, b12, _ = x
    b = np.array([[b11, b12, 0], [b12, b22, 0], [0, 0, -b11 - b22]])
    ev = np.linalg.eigvalsh(b)
    return ev.min() >= -1 / 3 - 1e-9 and ev.max() <= 2 / 3 + 1e-9


def find_twin(x0, names, rng, n_starts=25, sep=1e-3, tol=1e-11):
    """Search for a realizable state with the same eta and the same invariants but different b.

    Returns the twin (array) or None. A twin related to x0 only by a symmetry that preserves the
    whole input (b, S, W) would not count, but none exists here: the only orthogonal maps that
    fix G = g e1 (x) e2 are the identity and the z-reflection, which leave a channel b unchanged.
    """
    tgt = channel_state_invariants(x0, names)
    scale = np.maximum(np.abs(tgt), 1e-12)
    for _ in range(n_starts):
        xs = np.array([rng.uniform(-1 / 3, 2 / 3), rng.uniform(-1 / 3, 2 / 3), rng.uniform(-0.5, 0.5), x0[3]])
        f = lambda z: (channel_state_invariants(np.r_[z, x0[3]], names) - tgt) / scale
        r = least_squares(f, xs[:3], xtol=1e-15, ftol=1e-15, gtol=1e-15)
        z = np.r_[r.x, x0[3]]
        if np.abs(f(r.x)).max() < tol and np.abs(z[:3] - x0[:3]).max() > sep and realizable(z):
            return z
    return None
