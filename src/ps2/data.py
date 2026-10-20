"""Loading the Paper 1 processed DNS tables for Paper 2.

The raw DNS files are not redistributed and could not be downloaded in the environment where
the Section H checks were run, so Paper 2 reads the tables that Paper 1 released:

  results/e1_F1/processed_table.csv                          Lee & Moser, 1691 points
  results/e2_parts/classical_ridge_rf_lgbm/processed_table_hj.csv   Hoyas & Jimenez, 684 points

They were produced by ps.features.build_table from checksum-verified raw files. The SHA-256 of
each table is recorded here so that any later change to them is detected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = {
    "LM": ROOT / "results/e1_F1/processed_table.csv",
    "HJ": ROOT / "results/e2_parts/classical_ridge_rf_lgbm/processed_table_hj.csv",
}
SHA256 = {
    "LM": "aa7e316d68aacf5906a8045f627f2e6705c0c89b19fd242155064b3bf6780f11",
    "HJ": "dc97f3350f4fe13a4cdbe437e5e1d3bab767826e75892bd8865dfdf9c54f30ed",
}
N_POINTS = {"LM": 1691, "HJ": 684}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(which: str, verify: bool = True) -> pd.DataFrame:
    path = TABLES[which]
    if verify and sha256(path) != SHA256[which]:
        raise ValueError(f"{path} does not match the recorded Paper 1 checksum")
    df = pd.read_csv(path)
    if len(df) != N_POINTS[which]:
        raise ValueError(f"{which}: expected {N_POINTS[which]} points, found {len(df)}")
    return df


def consistency_report(df: pd.DataFrame) -> dict:
    """Identities that the processed tables must satisfy (Paper 1 definitions)."""
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    return {
        "max|b11+b22+b33|": float((df.b11 + df.b22 + df.b33).abs().max()),
        "max rel |eta - k|dU|/eps|": rel(df.eta.to_numpy(), (df.k * df.dUdy.abs() / df.eps).to_numpy()),
        "max rel |Re_t - k^2/eps|": rel(df.Re_t.to_numpy(), (df.k ** 2 / df.eps).to_numpy()),
        "max|P/eps + 2 b12 eta|": float((df.P_over_eps + 2 * df.b12 * df.eta).abs().max()),
        "max|Pi_ii| (Pi/eps)": float((df.Pi11 + df.Pi22 + df.Pi33).abs().max()),
        "min dU/dy": float(df.dUdy.min()),
        "max b12": float(df.b12.max()),
        "points with eta = 0": int((df.eta == 0).sum()),
    }
