"""Missingness summaries. Nothing is imputed here."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_by_column(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for name in frame.columns:
        missing = frame[name].isna()
        rows.append({"column": str(name), "n_missing": int(missing.sum()), "share_missing": float(missing.mean())})
    return rows


def complete_case_mask(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].notna().all(axis=1).to_numpy()
