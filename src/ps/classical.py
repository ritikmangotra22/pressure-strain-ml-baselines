"""Classical pressure-strain closures specialised to fully developed channel flow.

Only dU/dy is non-zero, so with S12 = Omega12 = dU/dy / 2 and
Omega_ij = (dU_i/dx_j - dU_j/dx_i)/2, every term reduces to functions of
b11, b22, b33, b12 and eta = k |dU/dy| / eps. Everything is normalised by eps.

Verification (2026-10-15): coefficient values and the full tensor form, including the
sign of the rotation term, agree with the independent implementation in OpenFOAM v12
(src/MomentumTransportModels/.../SSG.C and LRR.C, which cite Speziale, Sarkar & Gatski
1991 and Launder, Reece & Rodi 1975); see tests/test_ssg_matches_openfoam_convention...
LRR-IP (C_R = 1.8, C_P = 0.6) also agrees with the ANSYS CFX coefficient table (LRR-IP:
Cs1 = 1.8, Cr2 = 0.8, Cr4 = Cr5 = 0.6 in the a = 2b normalisation). The LRR-IP rapid term is
built directly from the production tensor, so it carries no separate rotation-sign choice.

Both closures are linear in their coefficients, so "refitting" is a weighted
linear least-squares problem (closed form).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SSG_PUBLISHED = {"C1": 3.4, "C1s": 1.8, "C2": 4.2, "C3": 0.8, "C3s": 1.3, "C4": 1.25, "C5": 0.4}
LRR_IP_PUBLISHED = {"CR": 1.8, "CP": 0.6}


def _b(df: pd.DataFrame):
    return (df["b11"].to_numpy(), df["b22"].to_numpy(), df["b33"].to_numpy(), df["b12"].to_numpy())


def ssg_basis(df: pd.DataFrame) -> np.ndarray:
    """Return array (n, 4 components, 7 coefficients) so that Pi/eps = basis @ C.

    Coefficient order: C1, C1*, C2, C3, C3*, C4, C5 (SSG 1991 form):
      Pi = -(C1 eps + C1* P) b + C2 eps (b.b - II/3 I) + (C3 - C3* sqrt(II)) k S
           + C4 k (b.S + S.b - 2/3 (b:S) I) + C5 k (b.W^T + W.b^T ...)  [see docstring]
    """
    b11, b22, b33, b12 = _b(df)
    eta = df["eta"].to_numpy()                 # k|U'|/eps, U' > 0 on the lower half
    p_eps = df["P_over_eps"].to_numpy()        # = -2 b12 eta
    ii = b11**2 + b22**2 + b33**2 + 2 * b12**2
    n = len(df)
    B = np.zeros((n, 4, 7))
    b = np.stack([b11, b22, b33, b12], axis=1)
    # T1: -b and -(P/eps) b
    B[:, :, 0] = -b
    B[:, :, 1] = -(p_eps[:, None] * b)
    # T2: b_ik b_kj - II/3 delta_ij
    bb = np.stack([b11**2 + b12**2, b12**2 + b22**2, b33**2, b12 * (b11 + b22)], axis=1)
    B[:, :, 2] = bb - (ii[:, None] / 3) * np.array([1, 1, 1, 0])
    # T3: (k/eps) S_ij  -> only (1,2): eta/2 ; and -sqrt(II) times it
    s = np.stack([0 * eta, 0 * eta, 0 * eta, eta / 2], axis=1)
    B[:, :, 3] = s
    B[:, :, 4] = -np.sqrt(ii)[:, None] * s
    # T4: (k/eps)(b_ik S_jk + b_jk S_ik - 2/3 b_mn S_mn delta_ij)
    B[:, :, 5] = np.stack([b12 * eta / 3, b12 * eta / 3, -2 * b12 * eta / 3, (b11 + b22) * eta / 2], axis=1)
    # T5: (k/eps)(b_ik W_jk + b_jk W_ik), W_ij = (dU_i/dx_j - dU_j/dx_i)/2. Sign checked against
    # OpenFOAM SSG.C and, independently, by the identity LRR-IP == SSG-form with
    # (C1, C1*, C2, C3, C3*, C4, C5) = (3.6, 0, 0, 0.8, 0, 1.2, 1.2) (tests/test_core.py).
    B[:, :, 6] = np.stack([b12 * eta, -b12 * eta, 0 * eta, (b22 - b11) * eta / 2], axis=1)
    return B


def lrr_ip_basis(df: pd.DataFrame) -> np.ndarray:
    """Pi/eps = -C_R * 2b - C_P (P_ij - P_kk/3 delta_ij)/eps. Order: C_R, C_P."""
    b11, b22, b33, b12 = _b(df)
    eta = df["eta"].to_numpy()
    p11 = -4 * b12 * eta                  # P11/eps = -2 R12 U'/eps
    p12 = -2 * (b22 + 1 / 3) * eta        # P12/eps = -R22 U'/eps
    n = len(df)
    B = np.zeros((n, 4, 2))
    B[:, :, 0] = -2 * np.stack([b11, b22, b33, b12], axis=1)
    B[:, :, 1] = -np.stack([2 * p11 / 3, -p11 / 3, -p11 / 3, p12], axis=1)
    return B


# Gibson & Launder (1978) wall-reflection ("wall-echo") correction added to LRR-IP, in the form
# implemented in OpenFOAM (LRR.C, wallReflection yes; Cref1 = 0.5, Cref2 = 0.3, kappa = 0.41,
# Cmu = 0.09):
#   phi^w_ij = f (Xkm nk nm d_ij - 3/2 Xik nk nj - 3/2 Xjk nk ni),
#   X = Cref1 (eps/k) R + Cref2 phi_2,  phi_2 = -C_P dev(P),  f = Cmu^(3/4) k^(3/2) / (kappa eps y).
# With n = e_y (lower wall), O(X) = (X22, -2 X22, X22, -3/2 X12) for components (11, 22, 33, 12).
# The coefficients enter linearly as (C_R, C_P, Cref1, D = Cref2 * C_P), so the refit stays closed form.
GL_KAPPA, GL_CMU = 0.41, 0.09
LRR_IP_GL_PUBLISHED = {"CR": 1.8, "CP": 0.6, "Cref1": 0.5, "D_Cref2xCP": 0.3 * 0.6}


def gl_wall_function(df: pd.DataFrame) -> np.ndarray:
    """f = Cmu^(3/4) k^(3/2) / (kappa eps y), all in wall units (nu = 1); y = distance to the nearest wall."""
    k, eps, yp = df["k"].to_numpy(), df["eps"].to_numpy(), df["y_plus"].to_numpy()
    return GL_CMU ** 0.75 * k ** 1.5 / (GL_KAPPA * eps * yp)


def lrr_ip_gl_basis(df: pd.DataFrame) -> np.ndarray:
    """LRR-IP + Gibson-Launder wall reflection. Order: C_R, C_P, Cref1, D = Cref2*C_P."""
    b11, b22, b33, b12 = _b(df)
    eta = df["eta"].to_numpy()
    f = gl_wall_function(df)
    base = lrr_ip_basis(df)
    p11 = -4 * b12 * eta
    p12 = -2 * (b22 + 1 / 3) * eta
    # X1 = (eps/k) R / eps = R/k = 2(b + I/3)
    x1_22, x1_12 = 2 * (b22 + 1 / 3), 2 * b12
    # X2 = phi_2 / (eps * C_P) = -dev(P)/eps  ->  22: +p11/3, 12: -p12
    x2_22, x2_12 = p11 / 3, -p12

    def op(x22, x12):
        return np.stack([x22, -2 * x22, x22, -1.5 * x12], axis=1)

    B = np.zeros((len(df), 4, 4))
    B[:, :, :2] = base
    B[:, :, 2] = f[:, None] * op(x1_22, x1_12)
    B[:, :, 3] = f[:, None] * op(x2_22, x2_12)
    return B


BASES = {"ssg": (ssg_basis, SSG_PUBLISHED), "lrr_ip": (lrr_ip_basis, LRR_IP_PUBLISHED),
         "lrr_ip_gl": (lrr_ip_gl_basis, LRR_IP_GL_PUBLISHED)}


class ClassicalClosure:
    """sklearn-like wrapper. mode='published' uses literature coefficients;
    mode='refit' solves weighted least squares with Frobenius multiplicities."""

    def __init__(self, kind: str = "ssg", mode: str = "published"):
        self.kind, self.mode = kind, mode
        basis_fn, pub = BASES[kind]
        self.basis_fn = basis_fn
        self.coef_ = np.array(list(pub.values()), dtype=float)
        self.coef_names_ = list(pub.keys())

    def fit(self, df: pd.DataFrame, y: np.ndarray, w: np.ndarray, m: np.ndarray):
        if self.mode == "published":
            return self
        B = self.basis_fn(df)                         # (n, 4, p)
        sw = np.sqrt(w)[:, None] * np.sqrt(m)[None, :]  # (n, 4)
        A = (B * sw[:, :, None]).reshape(-1, B.shape[2])
        rhs = (y * sw).reshape(-1)
        self.coef_, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.basis_fn(df) @ self.coef_
