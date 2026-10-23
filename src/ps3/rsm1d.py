"""One-dimensional Reynolds-stress solve for fully developed channel flow (Paper 3, Stage 2).

FROZEN test (protocol_v1 Stage 2, first design point): the four Reynolds-stress transport
equations are solved for R = (uu, vv, ww, uv) on the DNS grid, with everything except the
pressure-strain term taken from the DNS:

    0 = P_ij(R) + Pi_ij(R) - eps_ij^DNS + T_ij^DNS + D_ij^nu(R),

    P_11 = -2 uv U',  P_22 = P_33 = 0,  P_12 = -vv U'        (U' = dU+/dy+ from the DNS)
    T_ij^DNS = turbulent transport + pressure transport       (DNS budget columns)
    eps_ij^DNS = component dissipation                       (DNS budget column, subtracted)
    D_ij^nu = d^2 R_ij / dy+^2                                (viscous diffusion, 2nd-order FD)
    Pi_ij(R) = eps * f(b, G*, Re_t) from the closure, with b = R/(2k) - I/3, k = tr R / 2,
               eps the DNS scalar dissipation, eta = k |U'| / eps, Re_t = k^2 / eps.

So every error in the solved stresses comes from the closure and its feedback through
production and its own inputs. Boundary conditions: R = 0 at the wall; symmetry at the
centreline through a mirror node (normal stresses even, uv odd about y/delta = 1).
Wall units throughout (nu = 1).

Solver: pseudo-transient continuation with damped Newton steps; the Jacobian is the exact
tridiagonal diffusion part plus a 4x4-per-point block for production and closure, the latter
by finite differences (the closure is pointwise). Converged when the residual, scaled by the
DNS eps at each point, falls below `tol` everywhere.

Outcome classes (protocol_v1: a failed solve is infeasible, never given a finite score):
  converged | diverged (non-finite or blow-up) | not_converged (iteration limit) |
  non_realizable (converged but a negative eigenvalue of R beyond -1e-8 k)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

COMP = ("uu", "vv", "ww", "uv")


@dataclass
class ChannelCase:
    name: str
    re_tau: float
    y: np.ndarray            # y+ including the wall node y=0
    dUdy: np.ndarray
    eps: np.ndarray          # scalar dissipation (k budget), > 0
    eps_ij: np.ndarray       # (n, 4) component dissipation, as subtracted in the balance
    T_ij: np.ndarray         # (n, 4) turbulent + pressure transport
    Pi_dns: np.ndarray       # (n, 4) DNS pressure-strain (for the oracle test and errors)
    R_dns: np.ndarray        # (n, 4) DNS stresses
    y_delta: np.ndarray = field(default=None)


def case_from_lm(case) -> ChannelCase:
    """From a ps.io_lm Case (raw Lee & Moser files, wall node included)."""
    B = case.budgets
    g = lambda t: np.stack([B[c][t] for c in COMP], 1)
    return ChannelCase(case.name, case.y_plus[-1] / case.y_delta[-1], case.y_plus.copy(), case.dUdy.copy(),
                       np.abs(B["k"]["visc_dissipation"]), g("visc_dissipation"),
                       g("turb_transport") + g("pressure_transport"), g("pressure_strain"),
                       np.stack([case.uu, case.vv, case.ww, case.uv], 1), case.y_delta.copy())


# ------------------------------------------------------------------ discretisation
def _diffusion_matrix(y: np.ndarray):
    """Second-derivative operator on interior nodes 1..n-1 (node 0 is the wall, R = 0), with a
    mirror node beyond the last point at y_g = 2 Re_tau - y_last (y in wall units, centreline
    at Re_tau). Returns (D_even, D_odd): sparse (m, m) for m = n-1 unknown nodes."""
    n = len(y)
    re = None
    m = n - 1
    rows, cols, ve, vo = [], [], [], []
    for j in range(1, n):
        i = j - 1
        ym = y[j - 1]
        if j < n - 1:
            yp, ghost = y[j + 1], False
        else:
            re = y[-1] / 1.0
            yp, ghost = None, True
        if not ghost:
            hm, hp = y[j] - ym, yp - y[j]
            a, c, b = 2 / (hm * (hm + hp)), 2 / (hp * (hm + hp)), -2 / (hm * hp)
            for col, v in ((i - 1, a), (i, b), (i + 1, c)):
                if col >= 0:
                    rows.append(i); cols.append(col); ve.append(v); vo.append(v)
        elif abs(y[j] - CENTRE[0]) <= 1e-9 * CENTRE[0]:
            # last node ON the centreline (MKM grids): ghost at y_{n-2} mirrored, value R_{n-2} (even)
            # or -R_{n-2} (odd); LM grids end below the centreline and take the branch below
            h = y[j] - ym
            rows += [i, i]; cols += [i - 1, i]
            ve += [2 / h ** 2, -2 / h ** 2]; vo += [0.0, -2 / h ** 2]
        else:
            # mirror node: centreline at y_c; ghost at y_g = 2 y_c - y_last, value R_last (even) or -R_last (odd)
            hm = y[j] - ym
            hp = 2 * (CENTRE[0] - y[j])
            a, c, b = 2 / (hm * (hm + hp)), 2 / (hp * (hm + hp)), -2 / (hm * hp)
            rows += [i, i]; cols += [i - 1, i]
            ve += [a, b + c]; vo += [a, b - c]
    De = sp.csr_matrix((ve, (rows, cols)), shape=(m, m))
    Do = sp.csr_matrix((vo, (rows, cols)), shape=(m, m))
    return De, Do


CENTRE = [None]


# ------------------------------------------------------------------ closure inputs
def closure_table(R: np.ndarray, cc: ChannelCase, idx: np.ndarray) -> pd.DataFrame:
    """The Paper 1 table columns that the classical and learned closures read, from solved R."""
    uu, vv, ww, uv = R.T
    k = 0.5 * (uu + vv + ww)
    eps, dU, yp = cc.eps[idx], cc.dUdy[idx], cc.y[idx]
    return pd.DataFrame({"b11": uu / (2 * k) - 1 / 3, "b22": vv / (2 * k) - 1 / 3, "b33": ww / (2 * k) - 1 / 3,
                         "b12": uv / (2 * k), "eta": k * np.abs(dU) / eps, "Re_t": k ** 2 / eps,
                         "P_over_eps": -uv * dU / eps, "dUdy": dU, "k": k, "eps": eps, "y_plus": yp,
                         "y_delta": cc.y_delta[idx] if cc.y_delta is not None else yp / cc.re_tau,
                         "Pi11": 0.0, "Pi22": 0.0, "Pi33": 0.0, "Pi12": 0.0, "w": 1.0, "case": cc.name,
                         "region": "solve"})


@dataclass
class Result:
    status: str
    R: np.ndarray
    iterations: int
    residual: float
    history: list


def solve(cc: ChannelCase, closure, R0: np.ndarray | None = None, tol=1e-8, max_iter=400,
          dt0=1e-2, dt_max=1e6, fd_eps=1e-6, verbose=False) -> Result:
    """closure(df) -> (n, 4) Pi/eps for the rows of df (components 11, 22, 33, 12)."""
    y = cc.y
    n = len(y)
    CENTRE[0] = cc.re_tau
    idx = np.arange(1, n)
    m = n - 1
    De, Do = _diffusion_matrix(y)
    Dblk = sp.block_diag([De, De, De, Do], format="csr")           # unknown order: comp-major
    R = (cc.R_dns[1:] if R0 is None else R0[1:]).copy()
    scale = np.repeat(cc.eps[idx][None, :], 4, 0).ravel()          # residual scaled by eps
    src = (-cc.eps_ij[idx] + cc.T_ij[idx]).T.ravel()

    def pi_of(Rm):
        return closure(closure_table(Rm, cc, idx)) * cc.eps[idx][:, None]

    def F(Rm):
        P = np.zeros_like(Rm)
        P[:, 0] = -2 * Rm[:, 3] * cc.dUdy[idx]
        P[:, 3] = -Rm[:, 1] * cc.dUdy[idx]
        return (P + pi_of(Rm)).T.ravel() + src + Dblk @ Rm.T.ravel()

    def local_jac(Rm, f0):
        """d(P + Pi)/dR, 4x4 per point, by finite differences of the pointwise terms."""
        J = np.zeros((m, 4, 4))
        base = f0 - Dblk @ Rm.T.ravel() - src
        for c in range(4):
            h = fd_eps * np.maximum(np.abs(Rm[:, c]), 1e-12 + 1e-6 * np.abs(Rm).max())
            Rp = Rm.copy()
            Rp[:, c] += h
            fp = F(Rp) - Dblk @ Rp.T.ravel() - src
            J[:, :, c] = ((fp - base).reshape(4, m).T) / h[:, None]
        rows = np.repeat((np.arange(4)[:, None] * m + np.arange(m)[None, :]).T[:, :, None], 4, 2)
        cols = np.repeat((np.arange(4)[:, None] * m + np.arange(m)[None, :]).T[:, None, :], 4, 1)
        return sp.csr_matrix((J.ravel(), (rows.ravel(), cols.ravel())), shape=(4 * m, 4 * m))

    # Newton first (dt = inf); pseudo-time steps only after a failed Newton step (dt then grows back)
    dt, hist = np.inf, []
    for it in range(1, max_iter + 1):
        f = F(R)
        if not np.all(np.isfinite(f)):
            return Result("diverged", _full(R), it, np.inf, hist)
        res = float(np.max(np.abs(f) / scale))
        hist.append(res)
        if verbose and it % 10 == 0:
            print(it, res, dt)
        if res < tol:
            st = "converged" if realizable(_full(R)) else "non_realizable"
            return Result(st, _full(R), it, res, hist)
        J = Dblk + local_jac(R, f)
        A = (-J) if np.isinf(dt) else (sp.identity(4 * m) / dt - J)
        try:
            dx = spla.spsolve(A.tocsc(), f)
        except Exception:
            dx = np.full(4 * m, np.nan)
        d = dx.reshape(4, m).T
        if not np.all(np.isfinite(d)):
            dt = dt0 if np.isinf(dt) else dt * 0.25
            if dt < 1e-14:
                return Result("diverged", _full(R), it, res, hist)
            continue
        # only constraint on the step: keep k > 0 (shorten the step, do not reject it)
        k, dk = 0.5 * R[:, :3].sum(1), 0.5 * d[:, :3].sum(1)
        alpha = 1.0
        if np.any(dk < 0):
            alpha = min(1.0, 0.9 * float((k[dk < 0] / -dk[dk < 0]).min()))
        l2 = lambda v: float(np.sqrt(np.mean((v / scale) ** 2)))
        # backtracking line search along the (Newton or pseudo-time) direction: the Newton direction
        # is a descent direction of the scaled residual norm, so a short enough step always helps
        accepted = False
        while alpha >= 1e-6:
            Rn = R + alpha * d
            fn = F(Rn)
            if np.all(np.isfinite(fn)) and l2(fn) < (1 - 1e-4 * alpha) * l2(f):
                accepted = True
                break
            alpha *= 0.5
        if accepted:
            R = Rn
            if not np.isinf(dt):
                dt = dt * 4
                if dt > dt_max:
                    dt = np.inf
        else:                                   # line search failed: fall back to pseudo-time steps
            dt = dt0 if np.isinf(dt) else dt * 0.25
            if dt < 1e-14:
                return Result("diverged", _full(R), it, res, hist)
        if np.abs(R).max() > 1e4:
            return Result("diverged", _full(R), it, res, hist)
    return Result("not_converged", _full(R), max_iter, hist[-1] if hist else np.inf, hist)


def _full(Rm):
    return np.vstack([np.zeros((1, 4)), Rm])


def realizable(R: np.ndarray, tol=1e-8) -> bool:
    for uu, vv, ww, uv in R[1:]:
        k = 0.5 * (uu + vv + ww)
        ev = np.linalg.eigvalsh(np.array([[uu, uv, 0], [uv, vv, 0], [0, 0, ww]]))
        if ev.min() < -tol * max(k, 1e-30):
            return False
    return True


def stress_error(R: np.ndarray, cc: ChannelCase) -> dict:
    """Grid-weighted relative errors of the solved stresses against the DNS (J_R), per component and
    for the Frobenius norm of the full tensor (uv counted twice)."""
    w = np.empty_like(cc.y)
    w[1:-1] = 0.5 * (cc.y[2:] - cc.y[:-2]); w[0] = 0.5 * (cc.y[1] - cc.y[0]); w[-1] = 0.5 * (cc.y[-1] - cc.y[-2])
    w /= w.sum()
    m = np.array([1, 1, 1, 2.0])
    d, r = R - cc.R_dns, cc.R_dns
    out = {"J_R": float(np.sqrt((w[:, None] * m * d ** 2).sum() / (w[:, None] * m * r ** 2).sum()))}
    for i, c in enumerate(COMP):
        out[f"err_{c}"] = float(np.sqrt((w * d[:, i] ** 2).sum() / (w * r[:, i] ** 2).sum()))
    return out


# ------------------------------------------------------------------ closures
def oracle(cc: ChannelCase):
    """Returns the DNS Pi/eps at every interior node, independent of R (verification only)."""
    pe = cc.Pi_dns[1:] / cc.eps[1:, None]
    return lambda df: pe


def classical(name: str, coefs: dict):
    from ps import classical as C
    basis = {"ssg": C.ssg_basis, "lrr_ip": C.lrr_ip_basis, "lrr_ip_gl": C.lrr_ip_gl_basis}[name]
    v = np.array(list(coefs.values()), float)
    return lambda df: basis(df) @ v


def learned(fitted, c_t: float):
    """A Paper 2/3 network (ps2.pipeline.Fitted) as a closure; inputs built exactly as in training."""
    from ps2 import pipeline as PL
    import torch

    m64 = fitted.m64()

    def f(df):
        sp_ = PL.prepare(df, "durbin", c_t)
        with torch.no_grad():
            P = m64(sp_.b, sp_.G, sp_.re).numpy() * fitted.s
        return np.stack([P[:, 0, 0], P[:, 1, 1], P[:, 2, 2], P[:, 0, 1]], 1)
    return f


def anchored_oracle(cc: ChannelCase, C: float = 3.6):
    """Verification closure: Pi/eps = Pi_DNS/eps - C (b - b_DNS). Equal to the DNS term at the DNS
    state, with a Rotta-like restoring term (C = 3.6 = 2 x Rotta's 1.8). Needed because with the DNS
    dissipation and transport imposed, a closure independent of R leaves uu and uv governed only by
    viscous diffusion (ill-posed away from the wall)."""
    pe = cc.Pi_dns[1:] / cc.eps[1:, None]
    uu, vv, ww, uv = cc.R_dns[1:].T
    k = 0.5 * (uu + vv + ww)
    bd = np.stack([uu / (2 * k) - 1 / 3, vv / (2 * k) - 1 / 3, ww / (2 * k) - 1 / 3, uv / (2 * k)], 1)
    return lambda df: pe - C * (df[["b11", "b22", "b33", "b12"]].to_numpy() - bd)
