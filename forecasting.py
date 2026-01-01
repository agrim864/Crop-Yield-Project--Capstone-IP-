from __future__ import annotations

from typing import List, Tuple, Dict, Any
import re
import unicodedata
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from config import CONFIG

logger = logging.getLogger(__name__)

_LAG_RE = re.compile(r"^yield_lag_?(\d+)$")


# ----------------------------
# Column name normalization (must match data_loading.py behavior)
# ----------------------------
def _norm_col(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    t = str(name)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
    t = t.replace("μ", "u").replace("µ", "u")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _make_norm_map(cols: List[str]) -> Dict[str, str]:
    """Map normalized_name -> actual column name (first occurrence wins)."""
    mp: Dict[str, str] = {}
    for c in cols:
        nc = _norm_col(c).lower()
        if nc and nc not in mp:
            mp[nc] = c
    return mp


def _lag_features(feature_cols: List[str]) -> List[tuple[int, str]]:
    """Return [(k, colname)] for yield_lag{k} columns present in feature_cols."""
    out: List[tuple[int, str]] = []
    for c in feature_cols:
        if isinstance(c, str):
            m = _LAG_RE.match(c)
            if m:
                out.append((int(m.group(1)), c))
    out.sort(key=lambda x: x[0])
    return out


def _build_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Build the exact input schema expected by the tuned sklearn Pipeline:
      numeric: feature_cols + ['year']
      categorical: ['region', 'crop']

    Robustness:
      - if some feature_cols are missing in df, create them as NaN
      - force year numeric
    """
    df2 = df.copy()

    # ensure region/crop exist
    if "region" not in df2.columns:
        df2["region"] = np.nan
    if "crop" not in df2.columns:
        df2["crop"] = np.nan

    # ensure year exists
    if "year" not in df2.columns:
        df2["year"] = np.nan

    # ensure all features exist
    for c in feature_cols:
        if c not in df2.columns:
            df2[c] = np.nan

    X_num = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_cat = df2[["region", "crop"]]
    X_year = pd.to_numeric(df2["year"], errors="coerce").to_frame("year")

    return pd.concat([X_num, X_cat, X_year], axis=1)


def forecast_next_10_years(
    panel: pd.DataFrame, reg_model: Pipeline, feature_cols: List[str]
) -> pd.DataFrame:
    """
    For each region × crop, fit a linear trend on each non-lag feature over time and
    extrapolate N years ahead, then predict yields with the trained regressor.

    If yield_lag{k} features are present, this automatically switches to the recursive
    forecaster so lags are updated year-by-year.
    """
    if _lag_features(feature_cols):
        logger.info("Lag features detected; using recursive forecaster.")
        return forecast_next_10_years_recursive(panel, reg_model, feature_cols)

    last_year = int(pd.to_numeric(panel["year"], errors="coerce").max())
    future_years = np.arange(last_year + 1, last_year + CONFIG.scenario.years_ahead + 1)

    rows = []
    for (region, crop), group in panel.groupby(["region", "crop"]):
        group = group.sort_values("year")
        for year in future_years:
            row = {"region": region, "crop": crop, "year": int(year)}

            for feat in feature_cols:
                if feat not in group.columns:
                    row[feat] = np.nan
                    continue

                temp = group[["year", feat]].copy()
                temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                temp[feat] = pd.to_numeric(temp[feat], errors="coerce")
                temp = temp.dropna(subset=["year", feat])

                if temp.empty:
                    row[feat] = np.nan
                    continue

                if temp[feat].nunique(dropna=True) <= 1 or len(temp) == 1:
                    row[feat] = float(temp[feat].iloc[0])
                    continue

                years_num = temp["year"].values.astype(float)
                feat_num = temp[feat].values.astype(float)
                a, b = np.polyfit(years_num, feat_num, 1)
                row[feat] = float(a * float(year) + b)

            rows.append(row)

    future_df = pd.DataFrame(rows)
    X_future = _build_X(future_df, feature_cols)
    future_df["yield_pred"] = reg_model.predict(X_future)

    logger.info(
        "Generated %d future rows for %d-year forecast.",
        len(future_df),
        CONFIG.scenario.years_ahead,
    )
    return future_df


def forecast_scenarios(
    baseline_df: pd.DataFrame,
    reg_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Build three scenarios for each region/crop:
      - baseline: as-is
      - clean_air: AOD/PM2.5 reduced
      - polluted: AOD/PM2.5 increased

    Important:
      Column matching is robust to normalization differences (spaces, μ->u, case).
    """
    baseline = baseline_df.copy()
    baseline["scenario"] = "baseline"

    # Build normalized column lookup for the baseline df
    norm_map = _make_norm_map(list(baseline.columns))

    # Normalize config columns too
    aod_cfg = [_norm_col(c).lower() for c in CONFIG.scenario.aod_cols]
    pm_cfg = [_norm_col(c).lower() for c in CONFIG.scenario.pm_cols]

    # Resolve to actual columns present in df
    aod_cols_actual = [norm_map[c] for c in aod_cfg if c in norm_map]
    pm_cols_actual = [norm_map[c] for c in pm_cfg if c in norm_map]

    if not aod_cols_actual and not pm_cols_actual:
        logger.warning(
            "Scenario scaling: none of the configured AOD/PM columns were found. "
            "Check ScenarioConfig vs normalized column names."
        )

    scenarios = []

    for scenario_name, scale in [
        ("clean_air", CONFIG.scenario.clean_scale),
        ("polluted", CONFIG.scenario.polluted_scale),
    ]:
        df_s = baseline.copy()

        for col in aod_cols_actual + pm_cols_actual:
            # multiply only numeric-like values; coerce errors -> NaN (safe)
            df_s[col] = pd.to_numeric(df_s[col], errors="coerce") * float(scale)

        X_s = _build_X(df_s, feature_cols)
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
    Leak-free walk-forward backtest:

    For each target_year (last n_test_years globally):
      1) Fit clone(reg_model) on ALL rows with year < target_year
      2) For each (region,crop) that has a true row at target_year, create a
         forecast row by extrapolating non-lag features from prior years
      3) For yield_lag{k} features, use TRUE historical yields from hist
      4) Predict yield for that row using the model trained only on past years
    """
    panel2 = panel.copy()
    panel2["year"] = pd.to_numeric(panel2["year"], errors="coerce")
    panel2 = panel2.dropna(subset=["year"])
    panel2["year"] = panel2["year"].astype(int)

    if "yield" not in panel2.columns:
        logger.warning("Panel missing yield; returning empty results.")
        return pd.DataFrame(), float("nan"), float("nan")

    unique_years = sorted(panel2["year"].unique())
    if len(unique_years) < n_test_years + 2:
        logger.warning("Not enough years for backtesting; returning empty results.")
        return pd.DataFrame(), float("nan"), float("nan")

    test_years = unique_years[-n_test_years:]
    records: List[dict] = []

    lag_cols = _lag_features(feature_cols)
    max_lag = max([k for k, _ in lag_cols], default=0)

    for target_year in test_years:
        train_df = panel2[panel2["year"] < int(target_year)].copy()
        if train_df.empty:
            continue

        # Training mask: must have yield, region/crop, and at least one numeric feature present
        X_num = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        nonempty = ~X_num.isna().all(axis=1)
        cat_ok = (
            train_df["region"].notna()
            & train_df["crop"].notna()
            & train_df["year"].notna()
        )
        y_ok = train_df["yield"].notna()
        train_mask = nonempty & cat_ok & y_ok
        train_df = train_df.loc[train_mask].copy()

        if len(train_df) < 10:
            logger.warning(
                "Skipping target_year=%s: too few training rows (%d).",
                target_year,
                len(train_df),
            )
            continue

        X_train = _build_X(train_df, feature_cols)
        y_train = train_df["yield"].astype(float).values

        model_y = clone(reg_model)
        model_y.fit(X_train, y_train)

        for (region, crop), group in panel2.groupby(["region", "crop"]):
            group = group.sort_values("year")
            hist = group[group["year"] < int(target_year)]
            true_row = group[group["year"] == int(target_year)]
            if hist.empty or true_row.empty:
                continue

            row = {"region": region, "crop": crop, "year": int(target_year)}

            # 1) set lag features from TRUE history
            if max_lag > 0:
                hist_y = (
                    pd.to_numeric(hist["yield"], errors="coerce")
                    .dropna()
                    .values.astype(float)
                )
                if len(hist_y) == 0:
                    for _, col in lag_cols:
                        row[col] = np.nan
                else:
                    for k, col in lag_cols:
                        if len(hist_y) >= k:
                            row[col] = float(hist_y[-k])
                        else:
                            row[col] = float(hist_y[-1])

            # 2) extrapolate other numeric features
            for feat in feature_cols:
                if isinstance(feat, str) and _LAG_RE.match(feat):
                    continue

                if feat not in hist.columns:
                    row[feat] = np.nan
                    continue

                temp = hist[["year", feat]].copy()
                temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                temp[feat] = pd.to_numeric(temp[feat], errors="coerce")
                temp = temp.dropna(subset=["year", feat])

                if temp.empty:
                    row[feat] = np.nan
                elif temp[feat].nunique(dropna=True) <= 1 or len(temp) == 1:
                    row[feat] = float(temp[feat].iloc[0])
                else:
                    years_num = temp["year"].values.astype(float)
                    feat_num = temp[feat].values.astype(float)
                    a, b = np.polyfit(years_num, feat_num, 1)
                    row[feat] = float(a * float(target_year) + b)

            row["yield_true"] = float(pd.to_numeric(true_row["yield"].iloc[0], errors="coerce"))

            X_pred = _build_X(pd.DataFrame([row]), feature_cols)
            row["yield_pred"] = float(model_y.predict(X_pred)[0])

            records.append(row)

    if not records:
        logger.warning("No records for leak-free backtesting; returning empty results.")
        return pd.DataFrame(), float("nan"), float("nan")

    backtest_df = pd.DataFrame(records)
    rmse = float(np.sqrt(mean_squared_error(backtest_df["yield_true"], backtest_df["yield_pred"])))
    r2 = float(r2_score(backtest_df["yield_true"], backtest_df["yield_pred"]))

    logger.info("Leak-free backtest RMSE: %.3f, R^2: %.3f", rmse, r2)
    return backtest_df, rmse, r2


def forecast_next_10_years_recursive(
    panel: pd.DataFrame,
    reg_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Recursive N-year forecast that supports yield_lag{k} for any k present in feature_cols:
      - Predict year t+1 -> use it as lag1 for t+2, etc.
      - Non-lag features are extrapolated with a linear trend per group.
    """
    last_year_global = int(pd.to_numeric(panel["year"], errors="coerce").max())
    future_years = list(
        range(last_year_global + 1, last_year_global + CONFIG.scenario.years_ahead + 1)
    )

    lag_cols = _lag_features(feature_cols)
    max_lag = max([k for k, _ in lag_cols], default=0)

    rows = []
    for (region, crop), group in panel.groupby(["region", "crop"]):
        group = group.sort_values("year").copy()

        # history yields for seeding lags
        hist_y = pd.to_numeric(group.get("yield"), errors="coerce").dropna().values.astype(float)
        if len(hist_y) == 0:
            continue

        # keep last max_lag yields (pad if needed)
        if max_lag <= 0:
            last_yields: list[float] = []
        else:
            last_yields = list(hist_y[-max_lag:])
            if len(last_yields) < max_lag:
                last_yields = [last_yields[0]] * (max_lag - len(last_yields)) + last_yields

        for year in future_years:
            row = {"region": region, "crop": crop, "year": int(year)}

            # extrapolate non-lag features
            for feat in feature_cols:
                if isinstance(feat, str) and _LAG_RE.match(feat):
                    continue

                if feat not in group.columns:
                    row[feat] = np.nan
                    continue

                temp = group[["year", feat]].copy()
                temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                temp[feat] = pd.to_numeric(temp[feat], errors="coerce")
                temp = temp.dropna(subset=["year", feat])

                if temp.empty:
                    row[feat] = np.nan
                elif temp[feat].nunique(dropna=True) <= 1 or len(temp) == 1:
                    row[feat] = float(temp[feat].iloc[0])
                else:
                    years_num = temp["year"].values.astype(float)
                    feat_num = temp[feat].values.astype(float)
                    a, b = np.polyfit(years_num, feat_num, 1)
                    row[feat] = float(a * float(year) + b)

            # set lag features from last_yields
            if max_lag > 0:
                for k, col in lag_cols:
                    idx = -k
                    if len(last_yields) >= k:
                        row[col] = float(last_yields[idx])
                    else:
                        row[col] = float(last_yields[-1])

            X = _build_X(pd.DataFrame([row]), feature_cols)
            y_hat = float(reg_model.predict(X)[0])
            row["yield_pred"] = y_hat

            rows.append(row)

            # update lag buffer
            if max_lag > 0:
                last_yields.append(y_hat)
                last_yields = last_yields[-max_lag:]

    future_df = pd.DataFrame(rows)
    logger.info("Recursive forecast generated %d rows.", len(future_df))
    return future_df


# ----------------------------
# Time-based metrics (by year) for backtests
# ----------------------------
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "n": 0}

    yt = yt[mask]
    yp = yp[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))

    denom = np.abs(yt)
    m_mask = denom > 1e-12
    if bool(m_mask.any()):
        mape = float(np.mean(np.abs((yt[m_mask] - yp[m_mask]) / yt[m_mask])) * 100.0)
    else:
        mape = np.nan

    # r2_score needs at least two unique true values
    r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "n": int(len(yt))}


def compute_backtest_metrics_overall(
    backtest_df: pd.DataFrame,
    year_col: str = "year",
    true_col: str = "yield_true",
    pred_col: str = "yield_pred",
) -> Dict[str, Any]:
    if backtest_df is None or backtest_df.empty:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "n": 0}

    yt = pd.to_numeric(backtest_df.get(true_col), errors="coerce").values
    yp = pd.to_numeric(backtest_df.get(pred_col), errors="coerce").values
    return _regression_metrics(yt, yp)


def compute_backtest_metrics_by_year(
    backtest_df: pd.DataFrame,
    year_col: str = "year",
    true_col: str = "yield_true",
    pred_col: str = "yield_pred",
) -> pd.DataFrame:
    cols = ["year", "rmse", "mae", "mape", "r2", "n"]
    if backtest_df is None or backtest_df.empty or year_col not in backtest_df.columns:
        return pd.DataFrame(columns=cols)

    df = backtest_df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df[year_col] = df[year_col].astype(int)

    rows: list[dict] = []
    for yr, g in df.groupby(year_col):
        yt = pd.to_numeric(g.get(true_col), errors="coerce").values
        yp = pd.to_numeric(g.get(pred_col), errors="coerce").values
        m = _regression_metrics(yt, yp)
        rows.append({"year": int(yr), **m})

    out = pd.DataFrame(rows, columns=cols).sort_values("year").reset_index(drop=True)
    return out


# ----------------------------
# Future yield prediction using past data only (carry-forward features + recursive lags)
# ----------------------------
def forecast_next_10_years_past_only(
    panel: pd.DataFrame,
    reg_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Forecast future yields using only past data (no feature trend extrapolation):

    For each region × crop:
      - Non-lag features are carried forward from the last available (non-null) value
      - If yield_lag{k} features exist, predicted yields are fed back recursively to update lags

    Output columns: region, crop, year, <features>, yield_pred, forecast_method
    """
    last_year_global = int(pd.to_numeric(panel["year"], errors="coerce").max())
    future_years = list(
        range(last_year_global + 1, last_year_global + CONFIG.scenario.years_ahead + 1)
    )

    lag_cols = _lag_features(feature_cols)
    max_lag = max([k for k, _ in lag_cols], default=0)

    rows: list[dict] = []
    non_lag_feats = [f for f in feature_cols if not (isinstance(f, str) and _LAG_RE.match(f))]

    for (region, crop), group in panel.groupby(["region", "crop"]):
        group = group.sort_values("year").copy()

        # Seed yield history for lags (from TRUE data)
        hist_y = pd.to_numeric(group.get("yield"), errors="coerce").dropna().values.astype(float)
        if len(hist_y) == 0:
            continue

        if max_lag <= 0:
            last_yields: list[float] = []
        else:
            last_yields = list(hist_y[-max_lag:])
            if len(last_yields) < max_lag:
                last_yields = [last_yields[0]] * (max_lag - len(last_yields)) + last_yields

        # Carry-forward values for non-lag features
        carry: Dict[str, float] = {}
        for feat in non_lag_feats:
            if feat not in group.columns:
                carry[feat] = np.nan
                continue
            s = pd.to_numeric(group[feat], errors="coerce").dropna()
            carry[feat] = float(s.iloc[-1]) if not s.empty else np.nan

        for year in future_years:
            row: dict = {"region": region, "crop": crop, "year": int(year), "forecast_method": "past_only"}

            for feat in non_lag_feats:
                row[feat] = carry.get(feat, np.nan)

            if max_lag > 0:
                for k, col in lag_cols:
                    if len(last_yields) >= k:
                        row[col] = float(last_yields[-k])
                    else:
                        row[col] = float(last_yields[-1])

            X = _build_X(pd.DataFrame([row]), feature_cols)
            y_hat = float(reg_model.predict(X)[0])
            row["yield_pred"] = y_hat
            rows.append(row)

            if max_lag > 0:
                last_yields.append(y_hat)
                last_yields = last_yields[-max_lag:]

    future_df = pd.DataFrame(rows)
    logger.info("Past-only forecast generated %d rows.", len(future_df))
    return future_df
