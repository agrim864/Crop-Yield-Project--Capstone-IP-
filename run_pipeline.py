from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import CONFIG, setup_logging, ensure_output_dirs
from data_loading import build_panel_dataset
from features import (
    add_lagged_yield,
    add_yield_classes,
    split_feature_groups,
    drop_all_missing_features,
)
from modeling import (
    compute_baseline_metrics,
    train_classifiers,
    train_combined_classifier,
    train_yield_regressor,
    train_yield_regressor_gb,
    compute_gb_feature_importance,
)
from forecasting import (
    forecast_next_10_years,
    forecast_scenarios,
    backtest_forecasting,
)


logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ensure_output_dirs()
    outputs = CONFIG.paths.outputs_dir

    panel = build_panel_dataset(CONFIG.paths.data)
    panel = add_yield_classes(panel)
    panel = add_lagged_yield(panel, lags=(1, 2))

    aerosol_cols, met_cols, combined_cols = split_feature_groups(panel)

    lag_cols = [
        c for c in panel.columns if isinstance(c, str) and c.startswith("yield_lag")
    ]

    aerosol_cols = drop_all_missing_features(panel, aerosol_cols, label="aerosols")
    met_cols = drop_all_missing_features(panel, met_cols, label="meteorology")
    combined_cols = drop_all_missing_features(panel, combined_cols, label="combined")
    lag_cols = drop_all_missing_features(panel, lag_cols, label="lags")

    combined_with_lags = combined_cols + lag_cols

    logger.info("Aerosol feature columns: %s", aerosol_cols)
    logger.info("Meteorological feature columns: %s", met_cols)
    logger.info("Lagged yield columns: %s", lag_cols)

    baseline_metrics = compute_baseline_metrics(panel)

    clf_models, clf_metrics = train_classifiers(panel, aerosol_cols, met_cols)

    combined_clf, combined_clf_metrics = train_combined_classifier(
        panel, combined_with_lags
    )

    ridge_model, ridge_metrics = train_yield_regressor(panel, combined_cols)
    gb_model, gb_metrics = train_yield_regressor_gb(panel, combined_cols)

    gb_importance = compute_gb_feature_importance(panel, combined_cols, gb_model)
    if gb_importance is not None:
        gb_importance.to_csv(outputs / "gb_feature_importance.csv", index=False)

    future_forecast = forecast_next_10_years(panel, gb_model, combined_cols)
    future_forecast.to_csv(outputs / "yield_forecast_10_years.csv", index=False)

    scenario_forecasts = forecast_scenarios(future_forecast, gb_model, combined_cols)
    scenario_forecasts.to_csv(outputs / "yield_forecast_scenarios.csv", index=False)

    backtest_df, backtest_rmse, backtest_r2 = backtest_forecasting(
        panel, gb_model, combined_cols
    )
    if not backtest_df.empty:
        backtest_df.to_csv(outputs / "backtest_forecasts.csv", index=False)

    panel.to_csv(outputs / "panel_dataset_cleaned.csv", index=False)

    all_metrics = []

    base_row = {"model": "Baseline"}
    base_row.update(baseline_metrics)
    all_metrics.append(base_row)

    all_metrics.extend(clf_metrics)
    all_metrics.append(combined_clf_metrics)

    all_metrics.append(ridge_metrics)
    all_metrics.append(gb_metrics)

    all_metrics.append(
        {
            "model": "GB_backtest",
            "setting": "backtest",
            "rmse": backtest_rmse,
            "r2": backtest_r2,
        }
    )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(outputs / "metrics_summary.csv", index=False)

    logger.info("Pipeline completed. Outputs written to %s", outputs.resolve())


if __name__ == "__main__":
    main()
