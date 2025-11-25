from __future__ import annotations

import logging
from typing import Dict, List, Tuple



import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

from config import CONFIG

logger = logging.getLogger(__name__)


def _time_series_split() -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=CONFIG.cv.n_splits)


def compute_baseline_metrics(panel: pd.DataFrame) -> Dict[str, float]:
    """Compute simple baselines: majority-class classifier and mean-yield regressor."""
    metrics: Dict[str, float] = {}

    if "yield_class" in panel.columns:
        y = panel["yield_class"].dropna().astype(int)
        majority_class = y.value_counts().idxmax()
        y_pred = np.full_like(y, fill_value=majority_class)
        metrics["baseline_clf_macro_f1"] = f1_score(y, y_pred, average="macro")
        logger.info("Baseline majority-class macro-F1: %.3f",
                    metrics["baseline_clf_macro_f1"])

    if "yield" in panel.columns:
        y_reg = panel["yield"].dropna()
        mean_yield = y_reg.mean()
        y_pred_reg = np.full_like(y_reg, fill_value=mean_yield, dtype=float)
        rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
        r2 = r2_score(y_reg, y_pred_reg)
        metrics["baseline_reg_rmse"] = rmse
        metrics["baseline_reg_r2"] = r2
        logger.info("Baseline mean-yield RMSE: %.3f, R^2: %.3f", rmse, r2)

    return metrics


def train_classifiers(
    panel: pd.DataFrame, aerosol_cols: List[str], met_cols: List[str]
) -> Tuple[Dict[str, Pipeline], List[dict]]:
    """
    Train multinomial logistic regression classifiers:
      - aerosol-only vs yield_class
      - meteorology-only vs yield_class
    """
    tscv = _time_series_split()
    panel_sorted = panel.sort_values("year").reset_index(drop=True)

    metrics: List[dict] = []
    models: Dict[str, Pipeline] = {}

    def eval_model(X: pd.DataFrame, y: np.ndarray, name: str) -> Pipeline:
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500, C=0.5
                    ),
                ),
            ]
        )

        if len(X) < CONFIG.cv.min_samples_cv:
            logger.warning("%s: too few samples for CV, fitting on full data.", name)
            pipe.fit(X, y)
            y_pred_full = pipe.predict(X)
            macro_f1_full = f1_score(y, y_pred_full, average="macro")
            logger.info("%s full-data macro-F1: %.3f", name, macro_f1_full)
            print(f"Full-data classification report ({name}):")
            print(classification_report(y, y_pred_full))
            metrics.append(
                {
                    "model": name,
                    "setting": "full",
                    "macro_f1": macro_f1_full,
                }
            )
            return pipe

        f1_scores = []
        for train_idx, test_idx in tscv.split(X, y):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            f1_scores.append(f1_score(y[test_idx], y_pred, average="macro"))

        mean_f1 = float(np.mean(f1_scores))
        logger.info("%s classifier macro-F1 (time-series CV): %.3f", name, mean_f1)
        metrics.append(
            {
                "model": name,
                "setting": "tscv",
                "macro_f1": mean_f1,
            }
        )

        pipe.fit(X, y)
        print(f"Full-data classification report ({name}):")
        print(classification_report(y, pipe.predict(X)))
        return pipe

    y_full = panel_sorted["yield_class"].astype(int).values

    X_aero_raw = panel_sorted[aerosol_cols].apply(pd.to_numeric, errors="coerce")
    mask_aero = ~X_aero_raw.isna().all(axis=1)
    X_aero = X_aero_raw[mask_aero]
    y_aero = y_full[mask_aero]
    logger.info("Aerosol classifier: using %d rows after cleaning.", len(X_aero))
    aero_model = eval_model(X_aero, y_aero, "Aerosol-only")
    models["aerosol"] = aero_model

    X_met_raw = panel_sorted[met_cols].apply(pd.to_numeric, errors="coerce")
    mask_met = ~X_met_raw.isna().all(axis=1)
    X_met = X_met_raw[mask_met]
    y_met = y_full[mask_met]
    logger.info("Meteorology classifier: using %d rows after cleaning.", len(X_met))
    met_model = eval_model(X_met, y_met, "Meteorology-only")
    models["meteorology"] = met_model

    return models, metrics



def train_combined_classifier(
    panel: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Pipeline, dict]:
    """
    Train RandomForestClassifier on combined features + region/crop.
    """
    tscv = _time_series_split()
    panel_sorted = panel.sort_values("year").reset_index(drop=True)

    X_num_raw = panel_sorted[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask_nonempty = ~X_num_raw.isna().all(axis=1)
    mask_yclass = panel_sorted["yield_class"].notna()
    mask = mask_nonempty & mask_yclass

    X_num = X_num_raw[mask]
    cat_data = panel_sorted.loc[mask, ["region", "crop"]]
    y = panel_sorted.loc[mask, "yield_class"].astype(int).values

    logger.info("Combined classifier: using %d rows after cleaning.", len(X_num))

    numeric_features = feature_cols
    categorical_features = ["region", "crop"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=CONFIG.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

    metrics: dict = {"model": "Combined_RF"}

    if len(X_num) < CONFIG.cv.min_samples_cv:
        logger.warning("Combined classifier: too few samples for CV, fitting on full.")
        X_full = pd.concat([X_num, cat_data], axis=1)
        pipe.fit(X_full, y)
        y_pred = pipe.predict(X_full)
        macro_f1 = f1_score(y, y_pred, average="macro")
        metrics["setting"] = "full"
        metrics["macro_f1"] = macro_f1
        print("Full-data classification report (combined classifier):")
        print(classification_report(y, y_pred))
        return pipe, metrics

    f1_scores = []
    for train_idx, test_idx in tscv.split(X_num, y):
        X_train = pd.concat(
            [X_num.iloc[train_idx], cat_data.iloc[train_idx]], axis=1
        )
        X_test = pd.concat(
            [X_num.iloc[test_idx], cat_data.iloc[test_idx]], axis=1
        )
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    mean_f1 = float(np.mean(f1_scores))
    logger.info("Combined RF classifier macro-F1 (time-series CV): %.3f", mean_f1)

    X_full = pd.concat([X_num, cat_data], axis=1)
    pipe.fit(X_full, y)
    y_pred_full = pipe.predict(X_full)
    print("Full-data classification report (combined classifier):")
    print(classification_report(y, y_pred_full))

    metrics["setting"] = "tscv"
    metrics["macro_f1"] = mean_f1
    return pipe, metrics


def train_yield_regressor(
    panel: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Pipeline, dict]:
    """Ridge regression for continuous yield."""
    tscv = _time_series_split()
    panel_sorted = panel.sort_values("year").reset_index(drop=True)

    X_num_raw = panel_sorted[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask_nonempty = ~X_num_raw.isna().all(axis=1)
    mask_yield = panel_sorted["yield"].notna()
    mask = mask_nonempty & mask_yield

    X_num = X_num_raw[mask]
    cat_data = panel_sorted.loc[mask, ["region", "crop"]]
    y = panel_sorted.loc[mask, "yield"].values

    logger.info("Ridge regressor: using %d rows after cleaning.", len(X_num))

    numeric_features = feature_cols
    categorical_features = ["region", "crop"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Ridge(alpha=1.0)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    metrics: dict = {"model": "Ridge"}

    if len(X_num) < CONFIG.cv.min_samples_cv:
        logger.warning("Too few samples for Ridge CV, fitting on full data.")
        X_full = pd.concat([X_num, cat_data], axis=1)
        pipe.fit(X_full, y)
        metrics["setting"] = "full"
        return pipe, metrics

    rmses = []
    r2s = []

    for train_idx, test_idx in tscv.split(X_num, y):
        X_train = pd.concat(
            [X_num.iloc[train_idx], cat_data.iloc[train_idx]], axis=1
        )
        X_test = pd.concat(
            [X_num.iloc[test_idx], cat_data.iloc[test_idx]], axis=1
        )
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    metrics["setting"] = "tscv"
    metrics["rmse_mean"] = float(np.mean(rmses))
    metrics["r2_mean"] = float(np.mean(r2s))
    logger.info(
        "Ridge regressor RMSE (mean): %.3f, R^2 (mean): %.3f",
        metrics["rmse_mean"],
        metrics["r2_mean"],
    )

    X_full = pd.concat([X_num, cat_data], axis=1)
    pipe.fit(X_full, y)
    return pipe, metrics


def train_yield_regressor_gb(
    panel: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Pipeline, dict]:
    """Gradient Boosting regressor for yield."""
    tscv = _time_series_split()
    panel_sorted = panel.sort_values("year").reset_index(drop=True)

    X_num_raw = panel_sorted[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask_nonempty = ~X_num_raw.isna().all(axis=1)
    mask_yield = panel_sorted["yield"].notna()
    mask = mask_nonempty & mask_yield

    X_num = X_num_raw[mask]
    cat_data = panel_sorted.loc[mask, ["region", "crop"]]
    y = panel_sorted.loc[mask, "yield"].values

    logger.info("Gradient Boosting regressor: using %d rows after cleaning.", len(X_num))

    numeric_features = feature_cols
    categorical_features = ["region", "crop"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=CONFIG.random_state,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", gb)])

    metrics: dict = {"model": "GradientBoosting"}

    if len(X_num) < CONFIG.cv.min_samples_cv:
        logger.warning("Too few samples for GB CV, fitting on full data.")
        X_full = pd.concat([X_num, cat_data], axis=1)
        pipe.fit(X_full, y)
        metrics["setting"] = "full"
        return pipe, metrics

    rmses = []
    r2s = []

    for train_idx, test_idx in tscv.split(X_num, y):
        X_train = pd.concat(
            [X_num.iloc[train_idx], cat_data.iloc[train_idx]], axis=1
        )
        X_test = pd.concat(
            [X_num.iloc[test_idx], cat_data.iloc[test_idx]], axis=1
        )
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    metrics["setting"] = "tscv"
    metrics["rmse_mean"] = float(np.mean(rmses))
    metrics["r2_mean"] = float(np.mean(r2s))
    logger.info(
        "GB regressor RMSE (mean): %.3f, R^2 (mean): %.3f",
        metrics["rmse_mean"],
        metrics["r2_mean"],
    )

    X_full = pd.concat([X_num, cat_data], axis=1)
    pipe.fit(X_full, y)
    return pipe, metrics


def compute_gb_feature_importance(panel: pd.DataFrame, feature_cols, gb_model):
    """
    Compute permutation feature importance for the trained GB regressor *pipeline*.

    The GB model is a sklearn Pipeline with a ColumnTransformer that expects:
      - numeric features: feature_cols
      - categorical features: ['region', 'crop']

    So here we build an X that has exactly those columns, then run
    permutation_importance on the full pipeline. The output CSV will include
    importances for each original column (numeric + region/crop).
    """
    X_num_raw = panel[feature_cols].apply(pd.to_numeric, errors="coerce")


    mask_nonempty = ~X_num_raw.isna().all(axis=1)
    mask_yield = panel["yield"].notna()
    mask_cat = panel["region"].notna() & panel["crop"].notna()
    mask = mask_nonempty & mask_yield & mask_cat

    X_num = X_num_raw.loc[mask]
    cat_data = panel.loc[mask, ["region", "crop"]]
    y = panel.loc[mask, "yield"]

    if X_num.empty:
        logger.warning(
            "No rows available for GB permutation importance; skipping."
        )
        return None

    X = pd.concat([X_num, cat_data], axis=1)

    logger.info(
        "Computing GB permutation importance on %d samples and %d features.",
        len(X),
        X.shape[1],
    )

    result = permutation_importance(
        gb_model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    imp_df = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Top 15 GB permutation importances:\n%s",
        imp_df.head(15).to_string(index=False),
    )


    return imp_df
