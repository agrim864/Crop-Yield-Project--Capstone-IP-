from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config import CONFIG

logger = logging.getLogger(__name__)


def forecast_next_10_years(
    panel: pd.DataFrame, reg_model: Pipeline, feature_cols: List[str]
) -> pd.DataFrame:
    """
    For each region × crop, fit a linear trend on each feature over time and
    extrapolate 10 years ahead, then predict yields with the trained regressor.
    """
    last_year = int(panel["year"].max())
    future_years = np.arange(
        last_year + 1, last_year + CONFIG.scenario.years_ahead + 1
    )

    rows = []
    for (region, crop), group in panel.groupby(["region", "crop"]):
        for year in future_years:
            row = {"region": region, "crop": crop, "year": year}
            for feat in feature_cols:
                temp = group[["year", feat]].copy()
                temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                temp[feat] = pd.to_numeric(temp[feat], errors="coerce")
                temp = temp.dropna(subset=["year", feat])

                if temp.empty:
                    row[feat] = np.nan
                    continue

                if temp[feat].nunique() == 1 or len(temp) == 1:
                    row[feat] = float(temp[feat].iloc[0])
                    continue

                years_num = temp["year"].values.astype(float)
                feat_num = temp[feat].values.astype(float)
                coeffs = np.polyfit(years_num, feat_num, 1)
                row[feat] = float(coeffs[0] * year + coeffs[1])

            rows.append(row)

    future_df = pd.DataFrame(rows)
    X_future_num = future_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_future_cat = future_df[["region", "crop"]]
    X_future = pd.concat([X_future_num, X_future_cat], axis=1)

    future_df["yield_pred"] = reg_model.predict(X_future)
    logger.info("Generated %d future rows for 10-year forecast.", len(future_df))
    return future_df


def forecast_scenarios(
    baseline_df: pd.DataFrame,
    reg_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Build three 10-year scenarios for each region/crop:
      - baseline: as-is (uses baseline_df's yield_pred)
      - clean_air: AOD/PM2.5 reduced by 20%
      - polluted: AOD/PM2.5 increased by 20%
    """
    scenarios = []
    baseline = baseline_df.copy()
    baseline["scenario"] = "baseline"

    aod_cols = CONFIG.scenario.aod_cols
    pm_cols = CONFIG.scenario.pm_cols

    for scenario_name, scale in [
        ("clean_air", CONFIG.scenario.clean_scale),
        ("polluted", CONFIG.scenario.polluted_scale),
    ]:
        df_s = baseline.copy()
        for col in list(aod_cols) + list(pm_cols):
            if col in df_s.columns:
                df_s[col] = df_s[col] * scale

        X_num_s = df_s[feature_cols].apply(pd.to_numeric, errors="coerce")
        X_cat_s = df_s[["region", "crop"]]
        X_s = pd.concat([X_num_s, X_cat_s], axis=1)
        df_s["yield_pred"] = reg_model.predict(X_s)
        df_s["scenario"] = scenario_name
        scenarios.append(df_s)

    all_scenarios = pd.concat([baseline] + scenarios, ignore_index=True)
    logger.info(
        "Built scenario forecasts with %d rows (baseline + clean_air + polluted).",
        len(all_scenarios),
    )
    return all_scenarios


def backtest_forecasting(
    panel: pd.DataFrame,
    reg_model: Pipeline,
    feature_cols: List[str],
    n_test_years: int = 3,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Simple backtest: for the last n_test_years of each region×crop,
    simulate "forecasting" those years using linear extrapolation from
    earlier years, then compare to the true yields.
    """
    records = []

    for (region, crop), group in panel.groupby(["region", "crop"]):
        group = group.sort_values("year")
        unique_years = sorted(group["year"].unique())
        if len(unique_years) <= n_test_years + 1:
            continue

        test_years = unique_years[-n_test_years:]

        for target_year in test_years:
            hist = group[group["year"] < target_year]
            true_row = group[group["year"] == target_year]
            if hist.empty or true_row.empty:
                continue

            row = {"region": region, "crop": crop, "year": target_year}
            for feat in feature_cols:
                temp = hist[["year", feat]].copy()
                temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                temp[feat] = pd.to_numeric(temp[feat], errors="coerce")
                temp = temp.dropna(subset=["year", feat])

                if temp.empty:
                    row[feat] = np.nan
                    continue

                if temp[feat].nunique() == 1 or len(temp) == 1:
                    row[feat] = float(temp[feat].iloc[0])
                    continue

                years_num = temp["year"].values.astype(float)
                feat_num = temp[feat].values.astype(float)
                coeffs = np.polyfit(years_num, feat_num, 1)
                row[feat] = float(coeffs[0] * target_year + coeffs[1])

            row["yield_true"] = float(true_row["yield"].iloc[0])
            records.append(row)

    if not records:
        logger.warning("No records for backtesting; returning empty results.")
        return pd.DataFrame(), float("nan"), float("nan")

    backtest_df = pd.DataFrame(records)
    X_num = backtest_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_cat = backtest_df[["region", "crop"]]
    X = pd.concat([X_num, X_cat], axis=1)

    backtest_df["yield_pred"] = reg_model.predict(X)
    rmse = np.sqrt(mean_squared_error(backtest_df["yield_true"], backtest_df["yield_pred"]))
    r2 = r2_score(backtest_df["yield_true"], backtest_df["yield_pred"])

    logger.info("Backtest forecasting RMSE: %.3f, R^2: %.3f", rmse, r2)
    return backtest_df, float(rmse), float(r2)
