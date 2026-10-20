"""Tensor algebra for Paper 2 (Section E of the Paper 2 first deliverable).

Conventions (identical to Paper 1):
  * axes: x1 streamwise, x2 wall-normal, x3 spanwise; lower half-channel, dU/dy >= 0;
  * b_ij = R_ij / (2k) - delta_ij / 3;  G_ij = dU_i/dx_j, so in channel flow G_12 = dU/dy;
  * S = (G + G^T)/2, W = (G - G^T)/2, both multiplied by a time scale tau (S* = tau S, W* = tau W);
  * all quantities in wall units, so nu = 1, Re_t = k^2/eps and sqrt(nu/eps) = (k/eps)/sqrt(Re_t);
  * target Pi_ij/eps, stored as the four non-zero channel components (11, 22, 33, 12).

All functions take and return torch float64 tensors with a leading batch dimension, so the
same code is used by the models (autograd) and by the diagnostics.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DT = torch.float64
EYE = torch.eye(3, dtype=DT)
# Frobenius multiplicity of the stored channel components (11, 22, 33, 12).
FROB_M4 = np.array([1.0, 1.0, 1.0, 2.0])


# ----------------------------------------------------------------------------------------------
# Building tensors from the Paper 1 processed tables
# ----------------------------------------------------------------------------------------------
def time_scale_ratio(re_t, scheme: str = "durbin", c_t: float = 6.0):
    """tau / (k/eps).  'k/eps' -> 1;  'durbin' -> max(1, C_T / sqrt(Re_t)).

    tau = max(k/eps, C_T sqrt(nu/eps)) and sqrt(nu/eps) = (k/eps)/sqrt(Re_t) in wall units.
    C_T = 6 is the provisional value of the Paper 2 plan; it is still [TO CONFIRM] against
    Durbin (1991), so every diagnostic that uses it also reports a sensitivity to C_T.
    """
    re_t = np.asarray(re_t, float)
    if scheme == "k/eps":
        return np.ones_like(re_t)
    if scheme == "durbin":
        return np.maximum(1.0, c_t / np.sqrt(np.maximum(re_t, 1e-300)))
    raise ValueError(scheme)


def channel_tensors(df, scheme: str = "durbin", c_t: float = 6.0):
    """Return (b, G*) as (n,3,3) float64 tensors, where G* = tau * grad U (dimensionless).

    eta = (k/eps)|dU/dy| is stored in the table; the sign of dU/dy is kept (it is >= 0 in the
    lower half-channel used by Paper 1).
    """
    n = len(df)
    b = np.zeros((n, 3, 3))
    b[:, 0, 0], b[:, 1, 1], b[:, 2, 2] = df["b11"], df["b22"], df["b33"]
    b[:, 0, 1] = b[:, 1, 0] = df["b12"]
    sgn = np.sign(df["dUdy"].to_numpy(float))
    g = df["eta"].to_numpy(float) * sgn * time_scale_ratio(df["Re_t"], scheme, c_t)
    G = np.zeros((n, 3, 3))
    G[:, 0, 1] = g
    return torch.tensor(b, dtype=DT), torch.tensor(G, dtype=DT)


def target_tensor(df) -> torch.Tensor:
    P = np.zeros((len(df), 3, 3))
    P[:, 0, 0], P[:, 1, 1], P[:, 2, 2] = df["Pi11"], df["Pi22"], df["Pi33"]
    P[:, 0, 1] = P[:, 1, 0] = df["Pi12"]
    return torch.tensor(P, dtype=DT)


def to4(T: torch.Tensor) -> torch.Tensor:
    """(n,3,3) -> (n,4) channel components (11, 22, 33, 12)."""
    return torch.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1]], dim=1)


def sym_from6(v: torch.Tensor) -> torch.Tensor:
    """(n,6) in order (11,22,33,12,13,23) -> symmetric (n,3,3)."""
    n = v.shape[0]
    T = torch.zeros((n, 3, 3), dtype=v.dtype)
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = v[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = v[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = v[:, 5]
    return T


def to6(T: torch.Tensor) -> torch.Tensor:
    return torch.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 0, 2], T[:, 1, 2]], 1)


# ----------------------------------------------------------------------------------------------
# Basic operations
# ----------------------------------------------------------------------------------------------
def tr(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def dev(A):
    return A - tr(A)[:, None, None] * EYE.to(A.dtype) / 3.0


def sym_skew(G):
    return 0.5 * (G + G.transpose(1, 2)), 0.5 * (G - G.transpose(1, 2))


def frob(A):
    return torch.sqrt((A * A).sum(dim=(-2, -1)))


def rotate(Q, A):
    """Q A Q^T for a single (3,3) Q or a batch of Q (n,3,3)."""
    return Q @ A @ Q.transpose(-1, -2)


# ----------------------------------------------------------------------------------------------
# Tensor basis (Gerolymos, Lo & Vallet 2012; 8 polynomial elements)
# ----------------------------------------------------------------------------------------------
def basis_raw(b, S, W):
    """The 8 products WITHOUT the deviatoric correction (used, with the identity, by M2)."""
    b2 = b @ b
    return [b, b2, S, b @ S + S @ b, b @ W - W @ b, b2 @ S + S @ b2, b2 @ W - W @ b2,
            b2 @ W @ b - b @ W @ b2]


def basis_dev(b, S, W):
    """The 8-element deviatoric basis T(1..8) of Section E.3 (used by M4)."""
    return [dev(T) for T in basis_raw(b, S, W)]


def basis_m2(b, S, W):
    """Identity + raw products: 9 elements whose span contains trace-carrying tensors."""
    n = b.shape[0]
    return [EYE.to(b.dtype).expand(n, 3, 3)] + basis_raw(b, S, W)


# ----------------------------------------------------------------------------------------------
# Scalar invariants
# ----------------------------------------------------------------------------------------------
INVARIANT_SETS = {
    # Section E.4 set (5 tensor invariants; Re_t is appended separately)
    "E4": ["tr_b2", "tr_b3", "tr_S2", "tr_bS", "tr_bSW"],
    # the four-invariant set shown in E.4 to be insufficient
    "four": ["tr_b2", "tr_b3", "tr_S2", "tr_bS"],
    # E.4 set plus tr(b W^2) (proposed after check 2, see results/paper2/check2)
    "E4+bW2": ["tr_b2", "tr_b3", "tr_S2", "tr_bS", "tr_bSW", "tr_bW2"],
    # protocol v1 primary set (decision 1): the same six invariants, named for the protocol
    "P1": ["tr_b2", "tr_b3", "tr_S2", "tr_bS", "tr_bSW", "tr_bW2"],
}


def invariants(b, S, W, names):
    b2 = b @ b
    fn = {
        "tr_b2": lambda: tr(b2),
        "tr_b3": lambda: tr(b2 @ b),
        "tr_S2": lambda: tr(S @ S),
        "tr_W2": lambda: tr(W @ W),
        "tr_bS": lambda: tr(b @ S),
        "tr_b2S": lambda: tr(b2 @ S),
        "tr_bSW": lambda: tr(b @ S @ W),
        "tr_b2SW": lambda: tr(b2 @ S @ W),
        "tr_bW2": lambda: tr(b @ W @ W),
        "tr_b2W2": lambda: tr(b2 @ W @ W),
    }
    return torch.stack([fn[k]() for k in names], dim=1)


# ----------------------------------------------------------------------------------------------
# Transformations used in the tests
# ----------------------------------------------------------------------------------------------
def random_rotations(n, seed):
    """n Haar-distributed proper rotations (det = +1), float64."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out[i] = q
    return torch.tensor(out, dtype=DT)


def rot_about_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)


REFLECT_Y = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=DT))
