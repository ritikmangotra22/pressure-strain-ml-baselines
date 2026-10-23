"""A posteriori conditioning diagnostic for a pressure-strain closure (Paper 3, Stage 2 finding).

Frozen-k anisotropy problem on a DNS channel case: k, U', eps, the dissipation tensor and the
turbulent + pressure transport are taken from the DNS; the unknowns are b11, b22, b12 (b33 = -b11
- b22) on the DNS grid; the equations are the deviatoric parts of the four stress-transport
equations, with R = 2k(b + I/3). At the DNS state the residual equals the closure's a priori error
(plus the small DNS budget imbalance). The amplification factor

    A = (||delta b|| / ||b_DNS||) / rms(residual / eps),   delta b = -J^{-1} F(b_DNS),

measures how strongly the closure's a priori error is magnified into an a posteriori anisotropy
error by the first Newton step. Small A (anchored oracle: 0.6-0.9) means a well-conditioned
closure; A >> 1 means the closure fixes Pi on the DNS states but not its response to anisotropy
deviations, so small errors produce large solution changes (cf. Wu et al. 2019, JFM 869).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import rsm1d as S


def _setup(cc):
    S.CENTRE[0] = cc.re_tau
    y = cc.y
    m = len(y) - 1
    idx = np.arange(1, len(y))
    kD = 0.5 * cc.R_dns[1:, :3].sum(1)
    De, Do = S._diffusion_matrix(y)
    D = sp.block_diag([De, De, De, Do], format="csr")
    src = -cc.eps_ij[idx] + cc.T_ij[idx]
    bD = np.stack([cc.R_dns[1:, 0] / (2 * kD) - 1 / 3, cc.R_dns[1:, 1] / (2 * kD) - 1 / 3, cc.R_dns[1:, 3] / (2 * kD)], 1)
    return m, idx, kD, D, src, bD


def residual_fn(cc, closure):
    m, idx, kD, D, src, bD = _setup(cc)

    def Rof(b):
        return np.stack([2 * kD * (b[:, 0] + 1 / 3), 2 * kD * (b[:, 1] + 1 / 3),
                         2 * kD * (-b[:, 0] - b[:, 1] + 1 / 3), 2 * kD * b[:, 2]], 1)

    def F(b):
        Rm = Rof(b)
        P = np.zeros_like(Rm)
        P[:, 0] = -2 * Rm[:, 3] * cc.dUdy[idx]
        P[:, 3] = -Rm[:, 1] * cc.dUdy[idx]
        G = P + closure(S.closure_table(Rm, cc, idx)) * cc.eps[idx][:, None] + src + (D @ Rm.T.ravel()).reshape(4, m).T
        tr = G[:, :3].sum(1) / 3
        return np.stack([G[:, 0] - tr, G[:, 1] - tr, G[:, 3]], 1) / cc.eps[idx][:, None]
    return F, bD, Rof, m


def jacobian(F, b, m, h=1e-7):
    """Block-tridiagonal Jacobian by coloured finite differences (pointwise closure + 3-point diffusion)."""
    f = F(b)
    rows, cols, vals = [], [], []
    for c in range(3):
        for s in range(3):
            bp = b.copy()
            bp[s::3, c] += h
            df = (F(bp) - f) / h
            for j in range(s, m, 3):
                for jj in (j - 1, j, j + 1):
                    if 0 <= jj < m:
                        for r in range(3):
                            rows.append(jj * 3 + r); cols.append(j * 3 + c); vals.append(df[jj, r])
    return f, sp.csr_matrix((vals, (rows, cols)), shape=(3 * m, 3 * m))


def amplification(cc, closure) -> dict:
    F, bD, Rof, m = residual_fn(cc, closure)
    f, J = jacobian(F, bD, m)
    d = spla.spsolve(J.tocsc(), -f.ravel()).reshape(m, 3)
    rms = float(np.sqrt(np.mean(f ** 2)))
    step = float(np.linalg.norm(d) / np.linalg.norm(bD))
    return {"a_priori_rms": rms, "first_step_rel": step, "amplification": step / max(rms, 1e-30)}
