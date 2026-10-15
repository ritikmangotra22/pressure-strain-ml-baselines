"""Model wrappers with a common interface (manuscript Table 5).

    m = make_model(name, params, features, seed)
    info = m.fit(train_df, val_df=None)   # val_df only for early stopping during tuning
    y_hat = m.predict(df)                 # (n, 4) array of Pi/eps

Frobenius weighting: multi-output models are trained on y * sqrt(m) with
m = (1, 1, 1, 2), so their squared loss equals the tensor Frobenius loss.
All scalers are fitted on the training data passed to fit() only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .classical import ClassicalClosure
from .features import FROB_M, TARGETS

SQM = np.sqrt(FROB_M)


def _Xyw(df, features):
    return df[features].to_numpy(float), df[TARGETS].to_numpy(float), df["w"].to_numpy(float)


class Classical:
    tunable = False

    def __init__(self, kind, mode, **_):
        self.c = ClassicalClosure(kind, mode)

    def fit(self, train, val=None):
        self.c.fit(train, train[TARGETS].to_numpy(), train["w"].to_numpy(), FROB_M)
        return {"coef": dict(zip(self.c.coef_names_, map(float, self.c.coef_)))}

    def predict(self, df):
        return self.c.predict(df)


class RidgeModel:
    tunable = True

    def __init__(self, features, alpha=1.0, degree=1, **_):
        self.features = features
        self.pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                                  StandardScaler(), Ridge(alpha=alpha))

    @staticmethod
    def space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-6, 1e2, log=True),
                "degree": trial.suggest_int("degree", 1, 2)}

    def fit(self, train, val=None):
        X, y, w = _Xyw(train, self.features)
        self.pipe.fit(X, y * SQM, ridge__sample_weight=w)
        return {}

    def predict(self, df):
        return self.pipe.predict(df[self.features].to_numpy(float)) / SQM


class RFModel:
    tunable = True

    def __init__(self, features, seed=0, n_estimators=300, max_depth=None, min_samples_leaf=1,
                 max_features=1.0, n_jobs=1, **_):
        self.features = features
        self.m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf, max_features=max_features,
                                       random_state=seed, n_jobs=n_jobs)

    @staticmethod
    def space(trial):
        md = trial.suggest_categorical("max_depth", [8, 16, 32, "none"])
        return {"n_estimators": trial.suggest_categorical("n_estimators", [300, 1000]),
                "max_depth": None if md == "none" else md,
                "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10]),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0])}

    def fit(self, train, val=None):
        X, y, w = _Xyw(train, self.features)
        self.m.fit(X, y * SQM, sample_weight=w * len(w))
        return {}

    def predict(self, df):
        return self.m.predict(df[self.features].to_numpy(float)) / SQM


class LGBMModel:
    tunable = True

    def __init__(self, features, seed=0, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=1.0, reg_lambda=0.0, n_estimators=None, max_rounds=5000, **_):
        self.features, self.seed = features, seed
        self.kw = dict(learning_rate=learning_rate, num_leaves=num_leaves,
                       min_child_samples=min_child_samples, subsample=subsample,
                       subsample_freq=1 if subsample < 1 else 0, reg_lambda=reg_lambda,
                       random_state=seed, verbose=-1, n_jobs=1)
        self.n_estimators = n_estimators  # list of 4 ints for the final refit, or None
        self.max_rounds = max_rounds

    @staticmethod
    def space(trial):
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}

    def fit(self, train, val=None):
        import lightgbm as lgb
        X, y, w = _Xyw(train, self.features)
        self.models, best = [], []
        for j in range(4):
            if val is not None:
                Xv, yv, wv = _Xyw(val, self.features)
                m = lgb.LGBMRegressor(n_estimators=self.max_rounds, **self.kw)
                m.fit(X, y[:, j], sample_weight=w * len(w), eval_X=(Xv,), eval_y=(yv[:, j],),
                      eval_sample_weight=[wv * len(wv)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                best.append(int(m.best_iteration_ or self.max_rounds))
            else:
                n = self.n_estimators[j] if self.n_estimators else 500
                m = lgb.LGBMRegressor(n_estimators=n, **self.kw)
                m.fit(X, y[:, j], sample_weight=w * len(w))
                best.append(n)
            self.models.append(m)
        return {"n_estimators": best}

    def predict(self, df):
        X = df[self.features].to_numpy(float)
        return np.stack([m.predict(X) for m in self.models], axis=1)


class MLPModel:
    tunable = True

    def __init__(self, features, seed=0, n_layers=3, width=64, activation="tanh", lr=1e-3,
                 weight_decay=1e-5, epochs=None, max_epochs=5000, patience=200, **_):
        self.features, self.seed = features, seed
        self.n_layers, self.width, self.activation = n_layers, width, activation
        self.lr, self.wd = lr, weight_decay
        self.epochs, self.max_epochs, self.patience = epochs, max_epochs, patience

    @staticmethod
    def space(trial):
        return {"n_layers": trial.suggest_int("n_layers", 2, 5),
                "width": trial.suggest_categorical("width", [32, 64, 128]),
                "activation": trial.suggest_categorical("activation", ["tanh", "silu"]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}

    def _net(self, p):
        import torch.nn as nn
        act = nn.Tanh if self.activation == "tanh" else nn.SiLU
        layers, d = [], p
        for _ in range(self.n_layers):
            layers += [nn.Linear(d, self.width), act()]
            d = self.width
        layers.append(nn.Linear(d, 4))
        return nn.Sequential(*layers)

    def fit(self, train, val=None):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        X, y, w = _Xyw(train, self.features)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        ys = y * SQM
        self.yscale = float(np.sqrt((w[:, None] * ys ** 2).sum() / w.sum() / 4)) + 1e-12
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        Xt, yt, wt = T((X - self.mu) / self.sd), T(ys / self.yscale), T(w / w.sum())
        self.net = self._net(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = lambda p, t, ww: (ww[:, None] * (p - t) ** 2).sum()
        if val is not None:
            Xv, yv, wv = _Xyw(val, self.features)
            Xvt, yvt, wvt = T((Xv - self.mu) / self.sd), T(yv * SQM / self.yscale), T(wv / wv.sum())
        n_ep = self.epochs or self.max_epochs
        best, best_ep, best_state, since = np.inf, 0, None, 0
        for ep in range(1, n_ep + 1):
            opt.zero_grad()
            loss = loss_fn(self.net(Xt), yt, wt)
            loss.backward()
            opt.step()
            if val is not None:
                with torch.no_grad():
                    vl = float(loss_fn(self.net(Xvt), yvt, wvt))
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

    def predict(self, df):
        import torch
        X = (df[self.features].to_numpy(float) - self.mu) / self.sd
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32)).numpy()
        return out * self.yscale / SQM


REGISTRY = {
    "ssg_published": lambda f, s, p: Classical("ssg", "published"),
    "ssg_refit": lambda f, s, p: Classical("ssg", "refit"),
    "lrr_published": lambda f, s, p: Classical("lrr_ip", "published"),
    "lrr_refit": lambda f, s, p: Classical("lrr_ip", "refit"),
    "lrr_gl_published": lambda f, s, p: Classical("lrr_ip_gl", "published"),
    "lrr_gl_refit": lambda f, s, p: Classical("lrr_ip_gl", "refit"),
    "ridge": lambda f, s, p: RidgeModel(f, **p),
    "rf": lambda f, s, p: RFModel(f, seed=s, **p),
    "lgbm": lambda f, s, p: LGBMModel(f, seed=s, **p),
    "mlp": lambda f, s, p: MLPModel(f, seed=s, **p),
}
SPACES = {"ridge": RidgeModel.space, "rf": RFModel.space, "lgbm": LGBMModel.space, "mlp": MLPModel.space}
STOCHASTIC = {"rf", "lgbm", "mlp"}


def make_model(name, features, seed=0, params=None):
    return REGISTRY[name](features, seed, params or {})
