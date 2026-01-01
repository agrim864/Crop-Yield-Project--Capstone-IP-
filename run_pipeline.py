# run_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import numpy as np
import pandas as pd

from config import CONFIG, load_yaml_config, setup_logging, ensure_output_dirs
from data_loading import build_panel_dataset
from features import add_lagged_yield, add_yield_classes, split_feature_groups
from modeling import (
    compute_baseline_metrics,
    tune_classifier,
    tune_regressor,
    compute_permutation_importance,
)
from forecasting import (
    forecast_next_10_years,
    forecast_next_10_years_recursive,
    forecast_next_10_years_past_only,
    forecast_scenarios,
    backtest_forecasting,
    compute_backtest_metrics_overall,
    compute_backtest_metrics_by_year,
)
from report_generation import write_summary


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_timing_summary(timings: List[Tuple[str, float]], out_dir: Path) -> None:
    if not timings:
        return
    df = pd.DataFrame(timings, columns=["stage", "seconds"])
    _safe_write_csv(df, out_dir / "timing_summary.csv")


def _write_metrics_summary(
    baseline: Dict[str, float],
    clf_info: Dict[str, Any],
    reg_info: Dict[str, Any],
    bt_rmse: float,
    bt_r2: float,
    bt_overall: Dict[str, Any],
    out_dir: Path,
) -> None:
    rows: list[dict] = []

    rows.append(
        {
            "model": "Baseline",
            "baseline_clf_macro_f1": baseline.get("baseline_clf_macro_f1"),
            "baseline_reg_rmse": baseline.get("baseline_reg_rmse"),
            "baseline_reg_r2": baseline.get("baseline_reg_r2"),
        }
    )

    rows.append(
        {
            "model": "BestClassifier",
            "best_model_name": clf_info.get("best_model") or clf_info.get("best_model_name"),
            "cv_f1_macro": clf_info.get("best_score_cv_mean_f1_macro") or clf_info.get("best_score"),
            "train_f1_macro": clf_info.get("train_f1_macro"),
            "train_accuracy": clf_info.get("train_accuracy"),
            "oof_f1_macro": clf_info.get("oof_f1_macro"),
            "oof_accuracy": clf_info.get("oof_accuracy"),
            "oof_n": clf_info.get("oof_n"),
        }
    )

    rows.append(
        {
            "model": "BestRegressor",
            "best_model_name": reg_info.get("best_model") or reg_info.get("best_model_name"),
            "cv_r2": reg_info.get("best_score_cv_mean_r2"),
            "train_rmse": reg_info.get("train_rmse"),
            "train_r2": reg_info.get("train_r2"),
            "oof_rmse": reg_info.get("oof_rmse"),
            "oof_r2": reg_info.get("oof_r2"),
            "oof_n": reg_info.get("oof_n"),
        }
    )

    rows.append(
        {
            "model": "Backtest",
            "rmse": bt_rmse,
            "r2": bt_r2,
            "mae": bt_overall.get("mae"),
            "mape": bt_overall.get("mape"),
            "n": bt_overall.get("n"),
        }
    )

    _safe_write_csv(pd.DataFrame(rows), out_dir / "metrics_summary.csv")


def main() -> None:
    load_yaml_config()
    setup_logging()
    ensure_output_dirs()

    out_dir = CONFIG.paths.outputs_dir
    data_path = CONFIG.paths.data

    timings: List[Tuple[str, float]] = []
    t0 = time.perf_counter()
    t_prev = t0

    def _mark(stage: str) -> None:
        nonlocal t_prev
        t_now = time.perf_counter()
        timings.append((stage, float(t_now - t_prev)))
        t_prev = t_now

    # 1) Load + clean
    panel = build_panel_dataset(data_path)
    _mark("load_and_clean_data")

    # 2) Feature engineering
    panel = add_lagged_yield(panel, lags=(1, 2))
    panel = add_yield_classes(panel)
    _mark("feature_engineering")

    # 3) Pick feature columns
    aerosol_cols, met_cols, combined_cols = split_feature_groups(panel)
    combined_with_lags = combined_cols + ["yield_lag1", "yield_lag2"]
    _mark("select_features")

    # 4) Baselines + tuning
    baseline = compute_baseline_metrics(panel)
    best_clf, clf_info = tune_classifier(panel, combined_with_lags, out_dir)
    best_reg, reg_info = tune_regressor(panel, combined_with_lags, out_dir)
    _mark("tuning_models")

    # 5) Backtest (leak-free)
    backtest_df, backtest_rmse, backtest_r2 = backtest_forecasting(
        panel, best_reg, combined_with_lags, n_test_years=3
    )
    bt_overall = compute_backtest_metrics_overall(backtest_df)
    bt_by_year = compute_backtest_metrics_by_year(backtest_df)

    _safe_write_csv(backtest_df if not backtest_df.empty else pd.DataFrame(), out_dir / "backtest_forecasts.csv")
    _safe_write_csv(bt_by_year, out_dir / "backtest_metrics_by_year.csv")
    _mark("backtest")

    # 6) Forecasts (trend-based, extrapolating features)
    forecast_10y = forecast_next_10_years(panel, best_reg, combined_with_lags)
    scenarios = forecast_scenarios(forecast_10y, best_reg, combined_with_lags)

    _safe_write_csv(forecast_10y, out_dir / "yield_forecast_10_years.csv")
    _safe_write_csv(scenarios, out_dir / "yield_forecast_scenarios.csv")
    _mark("forecast_trend_based")

    # 7) Forecasts (past-only: carry-forward features + recursive lags)
    forecast_past = forecast_next_10_years_past_only(panel, best_reg, combined_with_lags)
    scenarios_past = forecast_scenarios(forecast_past, best_reg, combined_with_lags)

    _safe_write_csv(forecast_past, out_dir / "yield_forecast_past_only.csv")
    _safe_write_csv(scenarios_past, out_dir / "yield_forecast_scenarios_past_only.csv")
    _mark("forecast_past_only")

    # 8) Permutation importance (on fitted pipeline)
    perm_imp = compute_permutation_importance(
        panel=panel,
        feature_cols=combined_with_lags,
        model=best_reg,
        n_repeats=10,
        random_state=CONFIG.random_state,
        n_jobs=1,
    )
    if perm_imp is None:
        perm_imp = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    _safe_write_csv(perm_imp, out_dir / "permutation_importance.csv")
    # Backward-compatible name (older app/report expects this)
    _safe_write_csv(perm_imp, out_dir / "gb_feature_importance.csv")
    _mark("feature_importance")

    # 9) Write outputs expected by app.py / report
    _safe_write_csv(panel, out_dir / "panel_dataset_cleaned.csv")

    _write_metrics_summary(
        baseline=baseline,
        clf_info=clf_info,
        reg_info=reg_info,
        bt_rmse=float(backtest_rmse) if np.isfinite(backtest_rmse) else np.nan,
        bt_r2=float(backtest_r2) if np.isfinite(backtest_r2) else np.nan,
        bt_overall=bt_overall,
        out_dir=out_dir,
    )
    _mark("write_outputs")

    # 10) Timing summary
    _write_timing_summary(timings, out_dir)

    # 11) Human-readable summary/report
    write_summary(out_dir)

    # Total time (best-effort)
    total = float(time.perf_counter() - t0)
    timings.append(("total", total))
    _safe_write_csv(pd.DataFrame(timings, columns=["stage", "seconds"]), out_dir / "timing_summary.csv")


if __name__ == "__main__":
    main()
