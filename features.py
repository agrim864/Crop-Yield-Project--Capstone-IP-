from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def add_lagged_yield(
    panel: pd.DataFrame, lags: Iterable[int] = (1, 2)
) -> pd.DataFrame:
    """Add lagged yield features per (region, crop)."""
    panel = panel.sort_values(["region", "crop", "year"]).copy()
    for lag in lags:
        lag_col = f"yield_lag{lag}"
        panel[lag_col] = (
            panel.groupby(["region", "crop"])["yield"].shift(lag)
        )
    logger.info("Added lagged yield columns: %s", [f"yield_lag{l}" for l in lags])
    return panel


def add_yield_classes(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous yield into 3 classes (0,1,2) using crop-specific tertiles.
    0 = low, 1 = medium, 2 = high.
    """
    panel = panel.copy()
    panel["yield_class"] = np.nan

    for crop in panel["crop"].unique():
        mask = panel["crop"] == crop
        y = panel.loc[mask, "yield"]
        if y.notna().sum() < 3:
            median = y.median()
            panel.loc[mask & (y < median), "yield_class"] = 0
            panel.loc[mask & (y >= median), "yield_class"] = 1
            continue

        q1 = y.quantile(1 / 3)
        q2 = y.quantile(2 / 3)
        panel.loc[mask & (y < q1), "yield_class"] = 0
        panel.loc[mask & (y >= q1) & (y < q2), "yield_class"] = 1
        panel.loc[mask & (y >= q2), "yield_class"] = 2

    panel["yield_class"] = panel["yield_class"].astype(int)
    logger.info("Added yield_class column (3-tertile classes per crop).")
    return panel


def is_aerosol_col(name: str) -> bool:
    n = name.lower()
    aerosol_tokens = [
        "aod",
        "ch4",
        "pm2.5",
        "ozone",
        "o3 ",
        " o3",
        "so2",
        "co ",
        "co_",
        "co sfc",
        "co day",
        "co night",
        "co avg",
        "co ppbv",
        "net downward shortwave",
        "net dw sw",
        "net sw land",
        "shortwave radiation",
        " radiation",
    ]
    return any(tok in n for tok in aerosol_tokens)


def is_met_col(name: str) -> bool:
    n = name.lower()
    if any(tok in n for tok in ["temp", "skin", "precip", "mm/day"]):
        return True
    if n.strip() in ["day", "night", "avg"]:
        return True
    return False


def split_feature_groups(panel: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Return aerosol-only, met-only, and combined feature column names."""
    ignore = {"year", "yield", "sheet", "region", "crop", "yield_class"}
    feature_cols = [c for c in panel.columns if c not in ignore]

    aerosol_cols = [c for c in feature_cols if isinstance(c, str) and is_aerosol_col(c)]
    met_cols = [c for c in feature_cols if isinstance(c, str) and is_met_col(c)]
    combined_cols = sorted(set(aerosol_cols + met_cols))

    logger.info("Identified %d aerosol features, %d meteorology features.",
                len(aerosol_cols), len(met_cols))
    return aerosol_cols, met_cols, combined_cols


def drop_all_missing_features(
    df: pd.DataFrame, cols: Iterable[str], label: str = ""
) -> List[str]:
    """Drop columns that are entirely NaN in df; return the remaining list."""
    good = [c for c in cols if df[c].notna().any()]
    dropped = sorted(set(cols) - set(good))
    if dropped:
        logger.warning("Dropping %d all-missing features from %s: %s",
                       len(dropped), label, dropped)
    return good
