"""Paper 2 trainer (tensor form) and the channel-input network used by reproduction check R0.

The trainer works on full (n,3,3) tensors, because M2-M4 produce tensors directly. Its loss is
the grid-weighted squared Frobenius norm of the error in units of a scalar target scale:

    L = sum_n w_n || Pi_hat_n - Pi_n ||_F^2 / s^2,   w_n normalised to sum to one over the batch,
    s^2 = sum_n w_n ||Pi_n||_F^2 / 4      (the scale Paper 1 used for its MLP targets).

For a network whose head outputs o_c = sqrt(m_c) Pi_c / s for the four channel components
(m = 1, 1, 1, 2), this loss equals Paper 1's MLP loss term by term, so "M1 with Paper 1's
inputs" (check R0) is the Paper 1 MLP expressed through the Paper 2 code path. Optimiser,
full-batch updates, early stopping (patience 200, max 5000 epochs) and the refit rule (median of
the inner early-stopping epochs) follow Paper 1 Sections 7.6 and 9.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import tensors as T

SQM4 = np.sqrt(T.FROB_M4)
F1 = ["b11", "b22", "b12", "eta", "Re_t"]


class ChannelInputNet(nn.Module):
    """MLP on Paper 1's five channel inputs with a four-component channel head (R0 only)."""

    def __init__(self, n_in=5, n_layers=3, width=64, activation="tanh", dtype=torch.float32,
                 init="native"):
        """init='native': layers are created directly in `dtype` (PyTorch then draws a different
        random initial-weight stream in float64 than in float32, so seed s is NOT the same network).
        init='float32_cast': layers are created in float32 and cast to `dtype`, so seed s gives
        the same initial weights as Paper 1 (to float32 precision). Found during check 3."""
        super().__init__()
        build_dtype = torch.float32 if init == "float32_cast" else dtype
        act = nn.Tanh if activation == "tanh" else nn.SiLU
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, width, dtype=build_dtype), act()]
            d = width
        layers.append(nn.Linear(d, 4, dtype=build_dtype))
        self.net = nn.Sequential(*layers).to(dtype)
        self.register_buffer("sqm", torch.tensor(SQM4, dtype=dtype))

    def forward(self, x):
        """Return Pi/(eps s) as an (n,3,3) tensor."""
        o = self.net(x) / self.sqm
        P = torch.zeros((x.shape[0], 3, 3), dtype=x.dtype)
        P[:, 0, 0], P[:, 1, 1], P[:, 2, 2] = o[:, 0], o[:, 1], o[:, 2]
        P[:, 0, 1] = P[:, 1, 0] = o[:, 3]
        return P


def renorm(df):
    df = df.copy()
    df["w"] = df.groupby("case")["w"].transform(lambda s: s / s.sum())
    return df


def target_scale(P, w):
    w = w / w.sum()
    return float(np.sqrt((w * (P ** 2).sum(axis=(1, 2))).sum() / 4.0)) + 1e-12


def e_pi_tensor(P_hat, P, w):
    """Relative Frobenius error (Paper 1 Eq. for E_Pi), on (n,3,3) numpy arrays."""
    w = w / w.sum()
    num = (w * ((P_hat - P) ** 2).sum(axis=(1, 2))).sum()
    return float(np.sqrt(num / (w * (P ** 2).sum(axis=(1, 2))).sum()))


class R0Model:
    """Fit/predict wrapper: ChannelInputNet trained by the Paper 2 tensor-form trainer."""

    def __init__(self, seed=0, n_layers=3, width=64, activation="tanh", lr=1e-3, weight_decay=1e-5,
                 epochs=None, max_epochs=5000, patience=200, dtype="float32", init="native", **_):
        self.seed, self.dtype, self.init = seed, getattr(torch, dtype), init
        self.arch = dict(n_layers=n_layers, width=width, activation=activation)
        self.lr, self.wd, self.epochs = lr, weight_decay, epochs
        self.max_epochs, self.patience = max_epochs, patience

    def _x(self, df):
        X = (df[F1].to_numpy(float) - self.mu) / self.sd
        return torch.tensor(X, dtype=self.dtype)

    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        X = train[F1].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        P = T.target_tensor(train).numpy()
        w = train["w"].to_numpy(float)
        self.s = target_scale(P, w)
        Xt = self._x(train)
        Pt = torch.tensor(P / self.s, dtype=self.dtype)
        wt = torch.tensor(w / w.sum(), dtype=self.dtype)
        self.net = ChannelInputNet(len(F1), dtype=self.dtype, init=self.init, **self.arch)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = lambda Ph, Pr, ww: (ww * ((Ph - Pr) ** 2).sum(dim=(1, 2))).sum()
        if val is not None:
            Xv = self._x(val)
            Pv = torch.tensor(T.target_tensor(val).numpy() / self.s, dtype=self.dtype)
            wv = val["w"].to_numpy(float)
            wvt = torch.tensor(wv / wv.sum(), dtype=self.dtype)
        n_ep = self.epochs or self.max_epochs
        best, best_ep, best_state, since = np.inf, 0, None, 0
        for ep in range(1, n_ep + 1):
            opt.zero_grad()
            loss = loss_fn(self.net(Xt), Pt, wt)
            loss.backward()
            opt.step()
            if val is not None:
                with torch.no_grad():
                    vl = float(loss_fn(self.net(Xv), Pv, wvt))
                if vl < best - 1e-9:
                    best, best_ep, since = vl, ep, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    since += 1
                    if since >= self.patience:
                        break
        if val is not None and best_state is not None:
            self.net.load_state_dict(best_state)
            return {"epochs": best_ep}
        return {"epochs": n_ep}

    def predict_tensor(self, df):
        with torch.no_grad():
            return self.net(self._x(df)).double().numpy() * self.s

    def predict(self, df):
        """(n,4) channel components, Paper 1 layout."""
        P = self.predict_tensor(df)
        return np.stack([P[:, 0, 0], P[:, 1, 1], P[:, 2, 2], P[:, 0, 1]], 1)


SPACE = {  # identical to Paper 1 MLPModel.space
    "n_layers": ("int", 2, 5), "width": ("cat", [32, 64, 128]), "activation": ("cat", ["tanh", "silu"]),
    "lr": ("logf", 1e-4, 3e-3), "weight_decay": ("logf", 1e-6, 1e-2)}


def suggest(trial):
    return {"n_layers": trial.suggest_int("n_layers", 2, 5),
            "width": trial.suggest_categorical("width", [32, 64, 128]),
            "activation": trial.suggest_categorical("activation", ["tanh", "silu"]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
