"""Paper 2 model architectures M1-M4 (Section E.6 of the Paper 2 first deliverable).

Every model maps (b, G*, Re_t) -> Pi/eps as a symmetric (n,3,3) tensor, where G* = tau grad U.
They differ only in how the output is represented:

  M1  conventional MLP on the 16 Cartesian inputs (6 of b, 9 of G*, Re_t) -> 6 components
  M2  tensor basis: identity + 8 raw products, coefficients from scalar invariants (no dev)
  M3  M1 followed by  dev[ f(x) - f(x_iso) ],  x_iso = (b = 0, G* = 0, same Re_t)
  M4  deviatoric 8-element basis, coefficients from scalar invariants

No model is trained in this module. Input scalers are passed in (fitted elsewhere on training
data only); the unit tests use scalers fitted on the test states themselves, which is
harmless because no prediction is scored against data in those tests.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import tensors as T

DT = T.DT


def mlp(n_in, n_out, n_layers=3, width=64, activation="tanh", init="native"):
    """init='native': layers created directly in float64 (used by check 1, kept for reproduction).
    init='float32_cast': layers created in float32 and cast to float64, so a given seed yields the
    same initial weights whatever precision is later used (protocol v1, decision 2)."""
    act = nn.Tanh if activation == "tanh" else nn.SiLU
    build = torch.float32 if init == "float32_cast" else DT
    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, width, dtype=build), act()]
        d = width
    layers.append(nn.Linear(d, n_out, dtype=build))
    return nn.Sequential(*layers).to(DT)


class Scaler(nn.Module):
    """Standardisation with frozen statistics. Zero-variance inputs get sd = 1 (they are constant
    in the channel frame, e.g. b13, G_21; dividing by ~0 would make rotated inputs explode)."""

    def __init__(self, mu, sd):
        super().__init__()
        sd = np.where(np.asarray(sd) > 1e-12, sd, 1.0)
        self.register_buffer("mu", torch.tensor(np.asarray(mu), dtype=DT))
        self.register_buffer("sd", torch.tensor(sd, dtype=DT))

    def forward(self, x):
        return (x - self.mu) / self.sd


def cartesian_inputs(b, G, re_t):
    """16 frame-dependent inputs of M1/M3: b (6 independent), G* (9), Re_t."""
    return torch.cat([T.to6(b), G.reshape(-1, 9), re_t[:, None]], dim=1)


def fit_scaler(x: torch.Tensor) -> Scaler:
    x = x.detach().numpy()
    return Scaler(x.mean(0), x.std(0))


class M1(nn.Module):
    name = "M1"

    def __init__(self, scaler, **kw):
        super().__init__()
        self.scaler, self.net = scaler, mlp(16, 6, **kw)

    def raw(self, b, G, re_t):
        return T.sym_from6(self.net(self.scaler(cartesian_inputs(b, G, re_t))))

    def forward(self, b, G, re_t):
        return self.raw(b, G, re_t)


class M3(M1):
    """Hard constraints on a conventional MLP: zero trace and the isotropic limit."""
    name = "M3"

    def forward(self, b, G, re_t):
        z_b, z_G = torch.zeros_like(b), torch.zeros_like(G)
        return T.dev(self.raw(b, G, re_t) - self.raw(z_b, z_G, re_t))


class _Basis(nn.Module):
    inv_set = "E4"

    def __init__(self, inv_scaler, n_coef, **kw):
        super().__init__()
        self.scaler, self.net = inv_scaler, mlp(len(T.INVARIANT_SETS[self.inv_set]) + 1, n_coef, **kw)

    def features(self, b, G, re_t):
        S, W = T.sym_skew(G)
        return torch.cat([T.invariants(b, S, W, T.INVARIANT_SETS[self.inv_set]), re_t[:, None]], 1)

    def coefficients(self, b, G, re_t):
        return self.net(self.scaler(self.features(b, G, re_t)))

    def forward(self, b, G, re_t):
        S, W = T.sym_skew(G)
        g = self.coefficients(b, G, re_t)
        basis = self.basis(b, S, W)
        return sum(g[:, i, None, None] * B for i, B in enumerate(basis))

    # Protocol v1 precision policy: invariants and basis tensors are computed once in float64
    # from the inputs and cast to the training precision (prepare), then reused every epoch.
    def prepare(self, b, G, re_t, dtype):
        S, W = T.sym_skew(G)
        feats = self.features(b, G, re_t)
        basis = torch.stack(self.basis(b, S, W), dim=1)
        return feats.to(dtype), basis.to(dtype)

    def forward_from(self, feats, basis):
        g = self.net(self.scaler(feats))
        return (g[:, :, None, None] * basis).sum(dim=1)


class M2(_Basis):
    name = "M2"
    basis = staticmethod(T.basis_m2)

    def __init__(self, inv_scaler, **kw):
        super().__init__(inv_scaler, 9, **kw)


class M4(_Basis):
    name = "M4"
    basis = staticmethod(T.basis_dev)

    def __init__(self, inv_scaler, **kw):
        super().__init__(inv_scaler, 8, **kw)


def build(name, b, G, re_t, seed=0, inv_set="E4", dtype=None, **kw):
    """Construct an untrained model with scalers fitted on the given states.

    dtype=None keeps float64 (check 1). dtype=torch.float32 casts the whole model, scalers
    included, for float32 training; inputs must then be passed in float32 too. For the
    protocol, call with init='float32_cast' and inv_set='P1' (see configs/paper2/protocol_v1.json).
    """
    m = _build(name, b, G, re_t, seed, inv_set, **kw)
    return m if dtype is None else m.to(dtype)


def _build(name, b, G, re_t, seed=0, inv_set="E4", **kw):
    torch.manual_seed(seed)
    if name in ("M1", "M3"):
        sc = fit_scaler(cartesian_inputs(b, G, re_t))
        return (M1 if name == "M1" else M3)(sc, **kw)
    cls = M2 if name == "M2" else M4
    cls = type(cls.__name__, (cls,), {"inv_set": inv_set})
    S, W = T.sym_skew(G)
    feats = torch.cat([T.invariants(b, S, W, T.INVARIANT_SETS[inv_set]), re_t[:, None]], 1)
    return cls(fit_scaler(feats), **kw)


# ---------------------------------------------------------------------------------------------
# Ablation variants of protocol v1 (configs/paper2/protocol_v1.json, "ablations").
# The four core classes above are unchanged.
# ---------------------------------------------------------------------------------------------
class M3Soft(M1):
    """A-soft: M3 with the dev(.) projection removed; the trace is only penalised in the loss.
    The isotropic-limit subtraction is kept, so the ablation changes the trace mechanism only."""
    name = "M3_soft"

    def forward(self, b, G, re_t):
        z_b, z_G = torch.zeros_like(b), torch.zeros_like(G)
        return self.raw(b, G, re_t) - self.raw(z_b, z_G, re_t)


class M4Basis3(_Basis):
    """A-basis: M4 restricted to T(1), T(2), T(3) = b, dev(b^2), S* (fixed in protocol v1)."""
    name = "M4_basis3"

    @staticmethod
    def basis(b, S, W):
        return T.basis_dev(b, S, W)[:3]

    def __init__(self, inv_scaler, **kw):
        super().__init__(inv_scaler, 3, **kw)


class M4Lin(_Basis):
    """A-lin: rapid part linear in grad U. Coefficients depend only on (tr b^2, tr b^3, Re_t);
    T(1), T(2) do not depend on grad U and T(3)-T(8) are linear in S*, W*, so the output is
    affine in grad U at fixed b and Re_t."""
    name = "M4_lin"
    inv_set = "b_only"
    basis = staticmethod(T.basis_dev)

    def __init__(self, inv_scaler, **kw):
        nn.Module.__init__(self)
        self.scaler, self.net = inv_scaler, mlp(3, 8, **kw)

    def features(self, b, G, re_t):
        b2 = b @ b
        return torch.stack([T.tr(b2), T.tr(b2 @ b), re_t], 1)


VARIANTS = {"M3_soft": M3Soft, "M4_basis3": M4Basis3, "M4_lin": M4Lin}


def build_any(name, b, G, re_t, seed=0, inv_set="P1", dtype=None, **kw):
    """Core models (M1-M4) or ablation variants, built the same way (scalers on the given states)."""
    if name not in VARIANTS:
        return build(name, b, G, re_t, seed=seed, inv_set=inv_set, dtype=dtype, **kw)
    torch.manual_seed(seed)
    cls = VARIANTS[name]
    if cls is M3Soft:
        m = cls(fit_scaler(cartesian_inputs(b, G, re_t)), **kw)
    else:
        if cls is M4Basis3:
            cls = type(cls.__name__, (cls,), {"inv_set": inv_set})
        proto = cls.__new__(cls)
        feats = cls.features(proto, b, G, re_t) if cls is M4Lin else \
            torch.cat([T.invariants(b, *T.sym_skew(G), T.INVARIANT_SETS[inv_set]), re_t[:, None]], 1)
        m = cls(fit_scaler(feats), **kw)
    return m if dtype is None else m.to(dtype)
