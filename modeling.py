# modeling.py  (combined + cleaned)

# - Keeps only: baselines, tuned classifier/regressor (with caching), lag features, target-mean features,
#   OOF (leak-free) metrics, and permutation importance.
# - Removes older “utility” trainers (Ridge/GB/RF/LogReg baselines) because tuning covers them.
# - Includes fixes: NaN-safe preprocessing, no KeyError from te_* in direct classifiers,
#   OOF metrics used as the reported best_score, and quieter warnings.
# - Adds neural networks: MLPClassifier and MLPRegressor candidates (scikit-learn).
#
# ✅ Update (2026-01-02):
# - Made feature-col selection robust to missing columns in the Excel/panel:
#   uses reindex(...) so missing feature columns become NaN instead of raising KeyError.
#   This touches: _build_X, _clean_mask_for_task, _data_signature, compute_permutation_importance.

from __future__ import annotations

import json
import os
import warnings
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, r2_score, make_scorer
)

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.svm import SVR, LinearSVC

# ✅ Neural nets (sklearn)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from threadpoolctl import threadpool_limits

from year_split import YearForwardSplit
from config import CONFIG


# --------------------------
# Version + env knobs
# --------------------------
MODEL_SELECTION_VERSION = "2026-01-02_combined_clean_v1"

# fast: very small search
# metrics: moderate search
# full: widest search (slow)
TUNE_MODE = os.environ.get("CROP_TUNE_MODE", "metrics").strip().lower()

USE_CACHE = os.environ.get("CROP_USE_CACHE", "1").strip() == "1"
FORCE_RETRAIN = os.environ.get("CROP_FORCE_RETRAIN", "0").strip() == "1"

# Optional: manually force which feature set to use (normally keep "auto")
# auto/core/mid/full
FEATURE_SET = os.environ.get("CROP_FEATURE_SET", "auto").strip().lower()

# Reg->tertile thresholds:
# fixed = thresholds computed once from all data (matches yield_class construction style better)
# strict = thresholds recomputed each fit
TERTILES_MODE = os.environ.get("CROP_TERTILES_MODE", "fixed").strip().lower()

# Target-mean baseline features (learned inside CV folds)
USE_TARGET_MEAN = os.environ.get("CROP_USE_TARGET_MEAN", "1").strip() == "1"
TARGET_MEAN_SMOOTHING = float(os.environ.get("CROP_TARGET_MEAN_SMOOTHING", "10.0").strip())

# Optional debug prints
DEBUG_SANITY = os.environ.get("CROP_DEBUG_SANITY", "0").strip() == "1"

TE_COLS = ["te_crop", "te_region", "te_region_crop"]


# --------------------------
# Lag feature engineering
# --------------------------
_LAG_RE = re.compile(r"^yield_lag_?\d+$")
LAG_WINDOWS = [2, 3, 5]

LAG_STATS_COLS = [
    "yield_lag_last",
    "yield_lag_count_nonnull",
    "yield_lag_mean_2", "yield_lag_mean_3", "yield_lag_mean_5",
    "yield_lag_std_3", "yield_lag_std_5",
    "yield_lag_min_3", "yield_lag_max_3",
    "yield_lag_min_5", "yield_lag_max_5",
    "yield_lag_slope_3", "yield_lag_slope_5",
    "yield_lag_momentum_1", "yield_lag_momentum_2", "yield_lag_momentum_4",
    "yield_lag_ratio_last_mean3",
]

LAG_TE_INTER_COLS = [
    "lag_last_minus_te_crop",
    "lag_last_minus_te_region",
    "lag_last_minus_te_region_crop",
    "lag_last_div_te_crop",
    "lag_last_div_te_region_crop",
]


def _find_lag_cols(X: pd.DataFrame) -> List[Tuple[int, str]]:
    cols: List[Tuple[int, str]] = []
    for c in X.columns:
        s = str(c)
        m = re.match(r"^yield_lag_?(\d+)$", s)
        if m:
            cols.append((int(m.group(1)), s))
    cols.sort(key=lambda t: t[0])
    return cols


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)


def _row_nanmean(mat: np.ndarray) -> np.ndarray:
    mask = np.isfinite(mat)
    cnt = mask.sum(axis=1)
    s = np.where(mask, mat, 0.0).sum(axis=1)
    out = np.full(mat.shape[0], np.nan, dtype=float)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out


class LagStatsFeatures(BaseEstimator, TransformerMixin):
    """
    Leak-free numeric features derived from yield_lag* columns only.
    Always creates output columns in LAG_STATS_COLS (NaN if not computable).
    """
    def __init__(self, windows: List[int] = None):
        self.windows = windows if windows is not None else list(LAG_WINDOWS)
        self.lag_cols_: List[Tuple[int, str]] = []

    def fit(self, X: pd.DataFrame, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.lag_cols_ = _find_lag_cols(Xdf)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        for c in LAG_STATS_COLS:
            if c not in Xdf.columns:
                Xdf[c] = np.nan

        lag_cols = self.lag_cols_ if self.lag_cols_ else _find_lag_cols(Xdf)
        if not lag_cols:
            return Xdf

        # define lag1 as smallest-index lag
        lag1_name = lag_cols[0][1]
        lag1 = pd.to_numeric(Xdf[lag1_name], errors="coerce").to_numpy(dtype=float)
        Xdf["yield_lag_last"] = lag1

        # count non-null among first up-to-5 lags
        use_n = min(5, len(lag_cols))
        lag_names_5 = [name for _, name in lag_cols[:use_n]]
        mat5 = np.vstack([pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float) for c in lag_names_5]).T
        mask5 = np.isfinite(mat5)
        Xdf["yield_lag_count_nonnull"] = mask5.sum(axis=1).astype(float)

        def _slope(mat: np.ndarray) -> np.ndarray:
            n, w = mat.shape
            t = np.arange(1, w + 1, dtype=float)
            mask = np.isfinite(mat)
            cnt = mask.sum(axis=1).astype(float)

            sum_t = (mask * t).sum(axis=1)
            sum_t2 = (mask * (t ** 2)).sum(axis=1)
            sum_x = np.nansum(mat, axis=1)
            sum_tx = np.nansum(mat * t, axis=1)

            denom = (cnt * sum_t2 - sum_t ** 2)
            numer = (cnt * sum_tx - sum_t * sum_x)

            out = np.full(n, np.nan, dtype=float)
            ok = (cnt >= 2) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
            out[ok] = numer[ok] / denom[ok]
            return out

        for w in self.windows:
            w = int(w)
            if len(lag_cols) < w:
                continue

            lag_names = [name for _, name in lag_cols[:w]]
            mat = np.vstack([pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float) for c in lag_names]).T

            with np.errstate(all="ignore"):
                Xdf[f"yield_lag_mean_{w}"] = _row_nanmean(mat)

                if w in (3, 5):
                    Xdf[f"yield_lag_std_{w}"] = np.nanstd(mat, axis=1)
                    Xdf[f"yield_lag_min_{w}"] = np.nanmin(mat, axis=1)
                    Xdf[f"yield_lag_max_{w}"] = np.nanmax(mat, axis=1)
                    Xdf[f"yield_lag_slope_{w}"] = _slope(mat)

        def _get_lag(k: int) -> np.ndarray:
            for idx, name in lag_cols:
                if idx == k:
                    return pd.to_numeric(Xdf[name], errors="coerce").to_numpy(dtype=float)
            return np.full(len(Xdf), np.nan, dtype=float)

        lag2 = _get_lag(2)
        lag3 = _get_lag(3)
        lag5 = _get_lag(5)

        Xdf["yield_lag_momentum_1"] = lag1 - lag2
        Xdf["yield_lag_momentum_2"] = lag1 - lag3
        Xdf["yield_lag_momentum_4"] = lag1 - lag5

        mean3 = pd.to_numeric(Xdf.get("yield_lag_mean_3", np.nan), errors="coerce").to_numpy(dtype=float)
        Xdf["yield_lag_ratio_last_mean3"] = _safe_div(lag1, mean3)

        return Xdf


class LagTeInteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Interactions between yield_lag_last and target-mean baselines te_*.
    Always creates columns in LAG_TE_INTER_COLS (NaN if not computable).
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        for c in LAG_TE_INTER_COLS:
            if c not in Xdf.columns:
                Xdf[c] = np.nan

        if "yield_lag_last" not in Xdf.columns:
            return Xdf
        if not all(c in Xdf.columns for c in ["te_crop", "te_region", "te_region_crop"]):
            return Xdf

        lag_last = pd.to_numeric(Xdf["yield_lag_last"], errors="coerce").to_numpy(dtype=float)
        te_crop = pd.to_numeric(Xdf["te_crop"], errors="coerce").to_numpy(dtype=float)
        te_region = pd.to_numeric(Xdf["te_region"], errors="coerce").to_numpy(dtype=float)
        te_rc = pd.to_numeric(Xdf["te_region_crop"], errors="coerce").to_numpy(dtype=float)

        Xdf["lag_last_minus_te_crop"] = lag_last - te_crop
        Xdf["lag_last_minus_te_region"] = lag_last - te_region
        Xdf["lag_last_minus_te_region_crop"] = lag_last - te_rc
        Xdf["lag_last_div_te_crop"] = _safe_div(lag_last, te_crop)
        Xdf["lag_last_div_te_region_crop"] = _safe_div(lag_last, te_rc)

        return Xdf


def _with_lag_stats(numeric_features: List[str]) -> List[str]:
    out = list(numeric_features)
    for c in LAG_STATS_COLS:
        if c not in out:
            out.append(c)
    return out


def _with_lag_te_interactions(numeric_features: List[str]) -> List[str]:
    out = list(numeric_features)
    for c in LAG_TE_INTER_COLS:
        if c not in out:
            out.append(c)
    return out


# Optional libs
XGBClassifier = None
XGBRegressor = None

try:
    from xgboost import XGBClassifier as _XGBClassifier, XGBRegressor as _XGBRegressor
    XGBClassifier = _XGBClassifier
    XGBRegressor = _XGBRegressor
except Exception:
    pass


# --------------------------
# Baselines (kept)
# --------------------------
def compute_baseline_metrics(panel: pd.DataFrame) -> Dict[str, float]:
    """
    Simple baselines:
      - majority-class classifier for yield_class
      - mean-yield regressor for yield
    """
    metrics: Dict[str, float] = {}

    if "yield_class" in panel.columns:
        y = panel["yield_class"].dropna().astype(int)
        if len(y) > 0:
            majority_class = y.value_counts().idxmax()
            y_pred = np.full_like(y, fill_value=majority_class)
            metrics["baseline_clf_macro_f1"] = float(f1_score(y, y_pred, average="macro", zero_division=0))

    if "yield" in panel.columns:
        y_reg = panel["yield"].dropna().astype(float)
        if len(y_reg) > 0:
            mean_yield = float(y_reg.mean())
            y_pred_reg = np.full_like(y_reg, fill_value=mean_yield, dtype=float)
            metrics["baseline_reg_rmse"] = float(np.sqrt(mean_squared_error(y_reg, y_pred_reg)))
            metrics["baseline_reg_r2"] = float(r2_score(y_reg, y_pred_reg))

    return metrics


# --------------------------
# Helpers: performance/robustness
# --------------------------
def _default_outer_jobs() -> int:
    cpu = os.cpu_count() or 4
    if TUNE_MODE == "fast":
        cap = int(os.environ.get("CROP_OUTER_JOBS", "2"))
    elif TUNE_MODE == "metrics":
        cap = int(os.environ.get("CROP_OUTER_JOBS", "3"))
    else:
        cap = int(os.environ.get("CROP_OUTER_JOBS", "4"))
    return int(max(1, min(cap, cpu)))


def _cap_n_iter(param_space: Dict[str, List[Any]], requested: int) -> int:
    if not param_space:
        return 1
    total = 1
    for _, vals in param_space.items():
        if vals is None:
            continue
        total *= max(1, len(list(vals)))
    return int(max(1, min(requested, total)))


def _force_single_thread(estimator: Any) -> Any:
    # Avoid nested parallelism explosions; keep inner estimators single-threaded.
    for attr in ["n_jobs", "nthread", "num_threads", "thread_count"]:
        if hasattr(estimator, attr):
            try:
                estimator.set_params(**{attr: 1})
            except Exception:
                try:
                    setattr(estimator, attr, 1)
                except Exception:
                    pass
    return estimator


def _suppress_noisy_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but .* was fitted with feature names",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message=r"Scoring failed\.", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"Skipping features without any observed values:.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"All-NaN slice encountered", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"Degrees of freedom <= 0 for slice", category=RuntimeWarning)
    # ✅ common when training MLPs
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _onehot_ignore_unknown_dense() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _safe_imputer(*, strategy: str, add_indicator: bool = False, fill_value: Any = None) -> SimpleImputer:
    kwargs = {"strategy": strategy, "add_indicator": add_indicator}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    try:
        return SimpleImputer(**kwargs, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(**kwargs)


def _preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = False
) -> ColumnTransformer:
    # Constant impute prevents CV folds where a feature is all-NaN from blowing up.
    num_steps = [("imputer", _safe_imputer(strategy="constant", fill_value=0.0, add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", _safe_imputer(strategy="most_frequent", add_indicator=False)),
        ("onehot", _onehot_ignore_unknown_dense()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop"
    )


# ✅ NEW: safe column selection so missing Excel columns don't crash.
def _safe_reindex_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # reindex creates any missing columns filled with NaN (instead of KeyError)
    return df.reindex(columns=list(cols))


def _build_X(panel: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X_num = _safe_reindex_cols(panel, feature_cols).apply(pd.to_numeric, errors="coerce")
    X_cat = panel[["region", "crop"]]
    year_num = pd.to_numeric(panel["year"], errors="coerce").to_frame("year")
    return pd.concat([X_num, X_cat, year_num], axis=1)


def _clean_mask_for_task(panel: pd.DataFrame, feature_cols: List[str], task: str) -> pd.Series:
    X_num = _safe_reindex_cols(panel, feature_cols).apply(pd.to_numeric, errors="coerce")

    nonlag_cols = [c for c in feature_cols if not (isinstance(c, str) and _LAG_RE.match(c))]
    if nonlag_cols:
        nonempty = ~X_num[nonlag_cols].isna().all(axis=1)
    else:
        nonempty = pd.Series(True, index=panel.index)

    cat_ok = panel["region"].notna() & panel["crop"].notna() & panel["year"].notna()
    if task == "clf":
        y_ok = panel["yield_class"].notna() & panel["yield"].notna()
    else:
        y_ok = panel["yield"].notna()
    return nonempty & cat_ok & y_ok


def _quick_sanity(X: pd.DataFrame, y: np.ndarray, name: str) -> None:
    if not DEBUG_SANITY:
        return
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_ser = pd.Series(y)
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    nunique = X_df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()

    print("\n--- SANITY:", name, "---")
    print("X shape:", X_df.shape)
    print("y shape:", y_ser.shape)
    print("Any NaN in X:", bool(X_df.isna().any().any()))
    print("All-NaN cols:", len(all_nan_cols))
    print("Constant cols:", len(constant_cols))
    print("y unique:", int(y_ser.nunique(dropna=False)))
    print("y value_counts (top):")
    print(y_ser.value_counts(dropna=False).head(10))


def _drop_all_nan_and_constant_columns(
    X: pd.DataFrame,
    numeric_features: List[str],
    keep_cols_always: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if keep_cols_always is None:
        keep_cols_always = []

    X_df = X.copy()
    dropped: List[str] = []

    for col in list(numeric_features):
        if col in keep_cols_always:
            continue
        if col not in X_df.columns:
            continue
        s = X_df[col]
        if s.isna().all():
            dropped.append(col)
        elif s.nunique(dropna=False) <= 1:
            dropped.append(col)

    if dropped:
        X_df = X_df.drop(columns=dropped, errors="ignore")
        numeric_features = [c for c in numeric_features if c not in dropped]

    return X_df, numeric_features, dropped


def _data_signature(df: pd.DataFrame, cols: List[str]) -> str:
    sub = df.reindex(columns=list(cols)).copy()
    h = pd.util.hash_pandas_object(sub, index=True).values
    return hashlib.md5(h.tobytes()).hexdigest()


def _numeric_features_by_threshold(
    X: pd.DataFrame,
    numeric_features_all: List[str],
    missing_threshold: float,
) -> List[str]:
    kept: List[str] = []
    for c in numeric_features_all:
        if c not in X.columns:
            continue
        s = pd.to_numeric(X[c], errors="coerce")
        miss = float(s.isna().mean())
        if miss <= missing_threshold:
            kept.append(c)
    return kept


def _with_te(numeric_features: List[str]) -> List[str]:
    if not USE_TARGET_MEAN:
        return numeric_features
    out = list(numeric_features)
    for c in TE_COLS:
        if c not in out:
            out.append(c)
    return out


class GroupTargetMeanFeatures(BaseEstimator, TransformerMixin):
    """
    Adds leak-free baseline features learned from y inside each fit():
      te_crop, te_region, te_region_crop

    Smoothing:
      smoothed = (count*mean + m*global_mean) / (count + m)
    """
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = float(smoothing)
        self.global_mean_: float = 0.0
        self.crop_stats_: Dict[str, Tuple[float, int]] = {}
        self.region_stats_: Dict[str, Tuple[float, int]] = {}
        self.region_crop_stats_: Dict[Tuple[str, str], Tuple[float, int]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GroupTargetMeanFeatures":
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        yv = np.asarray(y).astype(float).ravel()
        self.global_mean_ = float(np.nanmean(yv))

        crop = Xdf["crop"].astype(str).fillna("missing")
        region = Xdf["region"].astype(str).fillna("missing")

        crop_df = pd.DataFrame({"crop": crop, "y": yv})
        g = crop_df.groupby("crop")["y"].agg(["mean", "count"])
        self.crop_stats_ = {k: (float(v["mean"]), int(v["count"])) for k, v in g.iterrows()}

        reg_df = pd.DataFrame({"region": region, "y": yv})
        g = reg_df.groupby("region")["y"].agg(["mean", "count"])
        self.region_stats_ = {k: (float(v["mean"]), int(v["count"])) for k, v in g.iterrows()}

        rc_df = pd.DataFrame({"region": region, "crop": crop, "y": yv})
        g = rc_df.groupby(["region", "crop"])["y"].agg(["mean", "count"])
        self.region_crop_stats_ = {
            (str(idx[0]), str(idx[1])): (float(row["mean"]), int(row["count"]))
            for idx, row in g.iterrows()
        }
        return self

    def _smooth(self, mean: float, count: int) -> float:
        m = self.smoothing
        return float((count * mean + m * self.global_mean_) / (count + m))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        crop = Xdf["crop"].astype(str).fillna("missing").values
        region = Xdf["region"].astype(str).fillna("missing").values

        te_crop = np.zeros(len(Xdf), dtype=float)
        te_region = np.zeros(len(Xdf), dtype=float)
        te_region_crop = np.zeros(len(Xdf), dtype=float)

        for i in range(len(Xdf)):
            c = crop[i]
            r = region[i]

            cm, cc = self.crop_stats_.get(c, (self.global_mean_, 0))
            rm, rc = self.region_stats_.get(r, (self.global_mean_, 0))
            rcm, rcc = self.region_crop_stats_.get((r, c), (self.global_mean_, 0))

            te_crop[i] = self._smooth(cm, cc)
            te_region[i] = self._smooth(rm, rc)
            te_region_crop[i] = self._smooth(rcm, rcc)

        Xdf["te_crop"] = te_crop
        Xdf["te_region"] = te_region
        Xdf["te_region_crop"] = te_region_crop
        return Xdf


# --------------------------
# Reg2Tertile wrapper
# --------------------------
def _compute_crop_tertiles(y: np.ndarray, crops: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    per_crop: Dict[str, Tuple[float, float]] = {}
    crops = crops.astype(str)

    t1_all, t2_all = np.quantile(y, [1 / 3, 2 / 3])

    for crop in np.unique(crops):
        mask = (crops == crop)
        if mask.sum() < 6:
            continue
        t1, t2 = np.quantile(y[mask], [1 / 3, 2 / 3])
        per_crop[crop] = (float(t1), float(t2))

    return per_crop, (float(t1_all), float(t2_all))


def _yield_to_class(
    y_hat: np.ndarray,
    crops: np.ndarray,
    per_crop: Dict[str, Tuple[float, float]],
    overall: Tuple[float, float]
) -> np.ndarray:
    crops = crops.astype(str)
    out = np.zeros(len(y_hat), dtype=int)
    for i in range(len(y_hat)):
        crop = crops[i]
        t1, t2 = per_crop.get(crop, overall)
        if y_hat[i] <= t1:
            out[i] = 0
        elif y_hat[i] <= t2:
            out[i] = 1
        else:
            out[i] = 2
    return out


@dataclass
class RegressorToTertileClassifier(BaseEstimator, ClassifierMixin):
    reg_pipeline: Pipeline
    per_crop_tertiles: Dict[str, Tuple[float, float]]
    overall_tertiles: Tuple[float, float]
    freeze_tertiles: bool = False

    def fit(self, X: pd.DataFrame, y_reg: np.ndarray) -> "RegressorToTertileClassifier":
        y_arr = np.asarray(y_reg)
        if y_arr.ndim == 2:
            y_arr = y_arr[:, 0]

        if not self.freeze_tertiles:
            crops = X["crop"].astype(str).values
            self.per_crop_tertiles, self.overall_tertiles = _compute_crop_tertiles(y_arr.astype(float), crops)

        self.reg_pipeline.fit(X, y_arr.astype(float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_hat = self.reg_pipeline.predict(X)
        crops = X["crop"].astype(str).values
        return _yield_to_class(y_hat, crops, self.per_crop_tertiles, self.overall_tertiles)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.predict(X).astype(int)
        proba = np.zeros((len(pred), 3), dtype=float)
        proba[np.arange(len(pred)), pred] = 1.0
        return proba


def _f1_macro_from_combo_y(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true)
    if yt.ndim == 2 and yt.shape[1] >= 2:
        y_class = yt[:, 1].astype(int)
    else:
        y_class = yt.astype(int)
    return float(f1_score(y_class, y_pred.astype(int), average="macro", zero_division=0))


def _make_reg2tert_wrapper(reg_pipe: Pipeline, X: pd.DataFrame, y_reg: np.ndarray) -> RegressorToTertileClassifier:
    if TERTILES_MODE == "fixed":
        per_crop, overall = _compute_crop_tertiles(y_reg.astype(float), X["crop"].astype(str).values)
        return RegressorToTertileClassifier(
            reg_pipeline=reg_pipe,
            per_crop_tertiles=per_crop,
            overall_tertiles=overall,
            freeze_tertiles=True,
        )
    return RegressorToTertileClassifier(
        reg_pipeline=reg_pipe,
        per_crop_tertiles={},
        overall_tertiles=(0.0, 0.0),
        freeze_tertiles=False,
    )


# --------------------------
# OOF evaluation (leak-free)
# --------------------------
def _oof_eval_regressor(best_pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, cv_splits) -> Dict[str, float]:
    y_pred = np.full(len(y), np.nan, dtype=float)

    for tr, te in cv_splits:
        est = clone(best_pipe)
        with threadpool_limits(limits=1):
            est.fit(X.iloc[tr], y[tr])
            y_pred[te] = est.predict(X.iloc[te])

    mask = np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"oof_r2": float("nan"), "oof_rmse": float("nan"), "oof_n": int(mask.sum())}

    rmse = float(np.sqrt(mean_squared_error(y[mask], y_pred[mask])))
    r2 = float(r2_score(y[mask], y_pred[mask]))
    return {"oof_r2": r2, "oof_rmse": rmse, "oof_n": int(mask.sum())}


def _oof_eval_classifier(best_est: Any, best_info: Dict[str, Any],
                         X: pd.DataFrame, y_class: np.ndarray, y_reg: np.ndarray, cv_splits) -> Dict[str, float]:
    y_pred = np.full(len(y_class), -1, dtype=int)
    seen = np.zeros(len(y_class), dtype=bool)

    y_combo = np.column_stack([y_reg.astype(float), y_class.astype(int)])

    for tr, te in cv_splits:
        est = clone(best_est)
        with threadpool_limits(limits=1):
            if best_info.get("approach") == "regression_to_tertiles":
                est.fit(X.iloc[tr], y_reg[tr])
            elif best_info.get("fit_y_combo", False):
                est.fit(X.iloc[tr], y_combo[tr])
            else:
                est.fit(X.iloc[tr], y_class[tr])

            y_pred[te] = est.predict(X.iloc[te]).astype(int)

        seen[te] = True

    if seen.sum() < 2:
        return {"oof_f1_macro": float("nan"), "oof_acc": float("nan"), "oof_n": int(seen.sum())}

    f1m = float(f1_score(y_class[seen], y_pred[seen], average="macro", zero_division=0))
    acc = float(accuracy_score(y_class[seen], y_pred[seen]))
    return {"oof_f1_macro": f1m, "oof_acc": acc, "oof_n": int(seen.sum())}


# --------------------------
# Feature set selection
# --------------------------
def _choose_numeric_feature_set(
    X: pd.DataFrame,
    numeric_features_all: List[str],
    cv_splits,
    task: str,
    y_reg: Optional[np.ndarray] = None,
    y_class: Optional[np.ndarray] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    if FEATURE_SET in {"core", "mid", "full"}:
        thr_map = {"core": 0.35, "mid": 0.60, "full": 0.85}
        chosen_thr = thr_map[FEATURE_SET]
        info = {"feature_set_mode": "forced", "missing_threshold": float(chosen_thr)}
        return _numeric_features_by_threshold(X, numeric_features_all, chosen_thr), info

    thresholds = [0.35, 0.60, 0.85]
    if TUNE_MODE == "full":
        thresholds = [0.25, 0.35, 0.50, 0.60, 0.70, 0.85]

    scores = []
    for thr in thresholds:
        nf = _numeric_features_by_threshold(X, numeric_features_all, thr)

        nf2 = _with_lag_stats(nf)
        if USE_TARGET_MEAN:
            nf2 = _with_te(nf2)
            nf2 = _with_lag_te_interactions(nf2)

        pre = _preprocessor(nf2, ["region", "crop"], scale_numeric=False)

        if task == "reg":
            model = HistGradientBoostingRegressor(random_state=CONFIG.random_state)

            steps = [("lag_stats", LagStatsFeatures())]
            if USE_TARGET_MEAN:
                steps.append(("target_means", GroupTargetMeanFeatures(smoothing=TARGET_MEAN_SMOOTHING)))
                steps.append(("lag_te_inter", LagTeInteractionFeatures()))
            steps += [("preprocess", pre), ("model", model)]
            pipe = Pipeline(steps)

            with threadpool_limits(limits=1):
                sc = cross_val_score(pipe, X, y_reg, cv=cv_splits, scoring="r2", n_jobs=1).mean()
            scores.append((thr, float(sc), len(nf)))

        else:
            reg = HistGradientBoostingRegressor(random_state=CONFIG.random_state)

            steps = [("lag_stats", LagStatsFeatures())]
            if USE_TARGET_MEAN:
                steps.append(("target_means", GroupTargetMeanFeatures(smoothing=TARGET_MEAN_SMOOTHING)))
                steps.append(("lag_te_inter", LagTeInteractionFeatures()))
            steps += [("preprocess", pre), ("model", reg)]
            reg_pipe = Pipeline(steps)

            wrapper = _make_reg2tert_wrapper(reg_pipe, X, y_reg)

            y_combo = np.column_stack([y_reg.astype(float), y_class.astype(int)])
            f1_combo_scorer = make_scorer(_f1_macro_from_combo_y, greater_is_better=True)

            with threadpool_limits(limits=1):
                sc = cross_val_score(wrapper, X, y_combo, cv=cv_splits, scoring=f1_combo_scorer, n_jobs=1).mean()
            scores.append((thr, float(sc), len(nf)))

    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
    best_thr, best_score, best_n = scores_sorted[0]

    info = {
        "feature_set_mode": "auto",
        "missing_threshold": float(best_thr),
        "feature_set_cv_score": float(best_score),
        "n_numeric_selected": int(best_n),
        "all_candidates": [{"thr": float(t), "score": float(s), "n_numeric": int(n)} for (t, s, n) in scores_sorted],
    }
    return _numeric_features_by_threshold(X, numeric_features_all, best_thr), info


# --------------------------
# Candidates
# --------------------------
@dataclass
class _Candidate:
    name: str
    estimator: Any
    scale_numeric: bool
    param_space: Dict[str, List[Any]]
    n_iter: int
    search_n_jobs: int = 0
    use_target_mean: bool = False  # only applies to reg/reg2tert pipelines


def _classifier_candidates() -> List[_Candidate]:
    rs = CONFIG.random_state
    outer_jobs = _default_outer_jobs()
    heavy_jobs = max(1, min(2, outer_jobs))

    cands = [
        _Candidate(
            name="HistGB",
            estimator=HistGradientBoostingClassifier(random_state=rs),
            scale_numeric=False,
            param_space={
                "model__learning_rate": [0.02, 0.03, 0.05],
                "model__max_depth": [2, 3, 4],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [20, 30, 40, 60],
                "model__l2_regularization": [0.01, 0.05, 0.1, 0.3],
            },
            n_iter=12 if TUNE_MODE == "fast" else (22 if TUNE_MODE == "metrics" else 45),
            search_n_jobs=outer_jobs,
            use_target_mean=False,
        ),
        _Candidate(
            name="ExtraTrees",
            estimator=ExtraTreesClassifier(random_state=rs, n_jobs=1, class_weight="balanced"),
            scale_numeric=False,
            param_space={
                "model__n_estimators": [600, 1200],
                "model__max_depth": [6, 8, None],
                "model__min_samples_split": [10, 20, 30],
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_features": ["sqrt", 0.6, 0.8],
            },
            n_iter=10 if TUNE_MODE == "fast" else (18 if TUNE_MODE == "metrics" else 35),
            search_n_jobs=outer_jobs,
            use_target_mean=False,
        ),
    ]

    if TUNE_MODE in {"metrics", "full"}:
        cands.append(
            _Candidate(
                name="LinearSVC",
                estimator=LinearSVC(class_weight="balanced", random_state=rs, dual=False, max_iter=50000),
                scale_numeric=True,
                param_space={"model__C": [0.01, 0.03, 0.05, 0.1, 0.3, 1.0, 3.0]},
                n_iter=10 if TUNE_MODE == "metrics" else 14,
                search_n_jobs=outer_jobs,
                use_target_mean=False,
            )
        )

    # ✅ Neural network classifier (MLP)
    # (Needs scaling; TE features are handled by YieldAwareClassifier wrapper.)
    if TUNE_MODE in {"fast", "metrics", "full"}:
        cands.append(
            _Candidate(
                name="MLP",
                estimator=MLPClassifier(
                    random_state=rs,
                    max_iter=1200,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.15,
                ),
                scale_numeric=True,
                param_space={
                    "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                    "model__activation": ["relu", "tanh"],
                    "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                    "model__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
                    "model__batch_size": [32, 64, 128],
                },
                n_iter=6 if TUNE_MODE == "fast" else (14 if TUNE_MODE == "metrics" else 26),
                search_n_jobs=outer_jobs,
                use_target_mean=False,
            )
        )

    if XGBClassifier is not None and TUNE_MODE in {"metrics", "full"}:
        cands.append(
            _Candidate(
                name="XGBoost",
                estimator=XGBClassifier(
                    random_state=rs,
                    n_estimators=1200,
                    objective="multi:softprob",
                    num_class=3,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    n_jobs=1,
                ),
                scale_numeric=False,
                param_space={
                    "model__max_depth": [2, 3, 4, 5],
                    "model__learning_rate": [0.01, 0.02, 0.03],
                    "model__subsample": [0.65, 0.8, 0.95],
                    "model__colsample_bytree": [0.5, 0.65, 0.8, 0.95],
                    "model__min_child_weight": [3, 5, 10, 15, 25],
                    "model__gamma": [0.0, 0.2, 0.5, 1.0],
                    "model__reg_lambda": [1.0, 5.0, 10.0, 20.0, 50.0],
                    "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
                },
                n_iter=26 if TUNE_MODE == "metrics" else 60,
                search_n_jobs=heavy_jobs,
                use_target_mean=False,
            )
        )

    return cands


def _regressor_candidates() -> List[_Candidate]:
    rs = CONFIG.random_state
    outer_jobs = _default_outer_jobs()
    heavy_jobs = max(1, min(2, outer_jobs))

    cands: List[_Candidate] = [
        _Candidate(
            name="HistGB",
            estimator=HistGradientBoostingRegressor(random_state=rs),
            scale_numeric=False,
            param_space={
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
                "model__max_depth": [2, 3, 4],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 30, 40, 60],
                "model__l2_regularization": [0.0, 0.01, 0.05, 0.1, 0.3],
            },
            n_iter=16 if TUNE_MODE == "fast" else (30 if TUNE_MODE == "metrics" else 55),
            search_n_jobs=outer_jobs,
            use_target_mean=True,
        ),
    ]

    # ✅ Neural network regressor (MLP)
    # (Needs scaling; TE helps, so keep use_target_mean=True.)
    if TUNE_MODE in {"fast", "metrics", "full"}:
        cands.append(
            _Candidate(
                name="MLP",
                estimator=MLPRegressor(
                    random_state=rs,
                    max_iter=2000,
                    early_stopping=True,
                    n_iter_no_change=25,
                    validation_fraction=0.15,
                ),
                scale_numeric=True,
                param_space={
                    "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128)],
                    "model__activation": ["relu", "tanh"],
                    "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    "model__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
                    "model__batch_size": [32, 64, 128],
                },
                n_iter=6 if TUNE_MODE == "fast" else (16 if TUNE_MODE == "metrics" else 30),
                search_n_jobs=outer_jobs,
                use_target_mean=True,
            )
        )

    if XGBRegressor is not None and TUNE_MODE in {"metrics", "full"}:
        cands.append(
            _Candidate(
                name="XGBoost",
                estimator=XGBRegressor(
                    random_state=rs,
                    n_estimators=1600,
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=1,
                ),
                scale_numeric=False,
                param_space={
                    "model__max_depth": [2, 3, 4, 5, 6],
                    "model__learning_rate": [0.005, 0.01, 0.02, 0.03],
                    "model__subsample": [0.6, 0.75, 0.9, 0.98],
                    "model__colsample_bytree": [0.5, 0.65, 0.8, 0.95],
                    "model__min_child_weight": [1, 3, 5, 10, 15, 25],
                    "model__gamma": [0.0, 0.1, 0.2, 0.5, 1.0],
                    "model__reg_lambda": [0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0],
                    "model__reg_alpha": [0.0, 0.1, 0.3, 0.5, 1.0],
                },
                n_iter=40 if TUNE_MODE == "metrics" else 85,
                search_n_jobs=heavy_jobs,
                use_target_mean=True,
            )
        )

    if TUNE_MODE == "full":
        cands.extend([
            _Candidate(
                name="ExtraTrees",
                estimator=ExtraTreesRegressor(random_state=rs, n_jobs=1),
                scale_numeric=False,
                param_space={
                    "model__n_estimators": [600, 1200],
                    "model__max_depth": [6, 8, None],
                    "model__min_samples_split": [10, 20, 30],
                    "model__min_samples_leaf": [5, 10, 20],
                    "model__max_features": ["sqrt", 0.6, 0.8],
                },
                n_iter=35,
                search_n_jobs=outer_jobs,
                use_target_mean=True,
            ),
            _Candidate(
                name="SVR_RBF",
                estimator=SVR(kernel="rbf"),
                scale_numeric=True,
                param_space={
                    "model__C": [0.3, 1.0, 3.0, 10.0],
                    "model__gamma": ["scale", "auto"],
                    "model__epsilon": [0.03, 0.05, 0.1, 0.2],
                },
                n_iter=16,
                search_n_jobs=outer_jobs,
                use_target_mean=True,
            ),
        ])

    return cands


# --------------------------
# Public: tuning API
# --------------------------
def tune_classifier(panel: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    _suppress_noisy_warnings()

    mask = _clean_mask_for_task(panel, feature_cols, task="clf")
    df = panel.loc[mask].sort_values("year").reset_index(drop=True)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "models" / "best_classifier_info.json"
    model_path = out_dir / "models" / "best_classifier.joblib"

    sig_cols = list(feature_cols) + ["region", "crop", "year", "yield", "yield_class"]
    sig = _data_signature(df, sig_cols)

    if USE_CACHE and (not FORCE_RETRAIN) and info_path.exists() and model_path.exists():
        try:
            cached_info = json.loads(info_path.read_text(encoding="utf-8"))
            if (
                cached_info.get("data_signature") == sig
                and cached_info.get("code_version") == MODEL_SELECTION_VERSION
                and cached_info.get("tune_mode") == TUNE_MODE
                and cached_info.get("feature_set") == FEATURE_SET
                and cached_info.get("tertiles_mode") == TERTILES_MODE
                and cached_info.get("use_target_mean") == USE_TARGET_MEAN
                and float(cached_info.get("target_mean_smoothing", TARGET_MEAN_SMOOTHING)) == TARGET_MEAN_SMOOTHING
            ):
                est = joblib.load(model_path)
                return est, cached_info
        except Exception:
            pass

    X = _build_X(df, feature_cols)
    y_class = df["yield_class"].astype(int).values
    y_reg = df["yield"].astype(float).values

    numeric_features_all = list(feature_cols) + ["year"]
    categorical_features = ["region", "crop"]

    _quick_sanity(X, y_class, name="clf_before_drop")

    X, numeric_features_all, dropped_allnan_const = _drop_all_nan_and_constant_columns(
        X, numeric_features=numeric_features_all, keep_cols_always=["year"]
    )

    cv = YearForwardSplit(
        n_splits=CONFIG.cv.n_splits,
        min_train_samples=CONFIG.cv.min_samples_cv,
        min_test_samples=max(2, CONFIG.cv.min_samples_cv // 2),
    )
    cv_splits = list(cv.split(X, y_class))

    numeric_features_base, feature_set_info = _choose_numeric_feature_set(
        X, numeric_features_all, cv_splits=cv_splits, task="clf", y_reg=y_reg, y_class=y_class
    )

    # ============================================================
    # Direct classifier gets TE + lag/TE interactions too
    # We fit on y_combo = [yield, class] so TE uses yield leak-free.
    # ============================================================
    y_combo = np.column_stack([y_reg.astype(float), y_class.astype(int)])
    f1_combo_scorer = make_scorer(_f1_macro_from_combo_y, greater_is_better=True)

    # optional sample weights (balanced)
    USE_SAMPLE_WEIGHT = True
    SAMPLE_WEIGHT_POWER = 1.0

    class YieldAwareClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, model, numeric_features_base, categorical_features,
                    use_target_mean=True, smoothing=10.0, use_sample_weight=False, sample_weight_power=1.0,
                    scale_numeric=False):
            self.model = model
        # IMPORTANT: do NOT copy these lists, sklearn.clone needs same object identity
            self.numeric_features_base = numeric_features_base
            self.categorical_features = categorical_features

            self.use_target_mean = use_target_mean
            self.smoothing = float(smoothing)
            self.use_sample_weight = bool(use_sample_weight)
            self.sample_weight_power = float(sample_weight_power)
            self.scale_numeric = bool(scale_numeric)


        def fit(self, X, y):
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
                y_reg_local = y_arr[:, 0].astype(float)
                y_class_local = y_arr[:, 1].astype(int)
            else:
                y_reg_local = y_arr.astype(float)
                y_class_local = y_arr.astype(int)

            self._lag_stats = LagStatsFeatures()
            Xt = self._lag_stats.fit_transform(X)

            if self.use_target_mean:
                self._te = GroupTargetMeanFeatures(smoothing=self.smoothing)
                Xt = self._te.fit_transform(Xt, y_reg_local)
                self._lag_te = LagTeInteractionFeatures()
                Xt = self._lag_te.fit_transform(Xt)
            else:
                self._te = None
                self._lag_te = None

            numeric_final = list(self.numeric_features_base) + list(LAG_STATS_COLS)
            if self.use_target_mean:
                numeric_final += list(TE_COLS) + list(LAG_TE_INTER_COLS)

            # only keep cols actually present (prevents KeyError)
            numeric_final = [c for c in dict.fromkeys(numeric_final) if c in Xt.columns]

            self._pre = _preprocessor(numeric_final, self.categorical_features, scale_numeric=self.scale_numeric)
            Xp = self._pre.fit_transform(Xt)

            self.model_ = clone(self.model)

            fit_kwargs = {}
            if self.use_sample_weight:
                sw = compute_sample_weight(class_weight="balanced", y=y_class_local)
                if self.sample_weight_power != 1.0:
                    sw = np.power(sw, self.sample_weight_power)
                fit_kwargs["sample_weight"] = sw

            try:
                self.model_.fit(Xp, y_class_local, **fit_kwargs)
            except TypeError:
                self.model_.fit(Xp, y_class_local)

            return self

        def _transform(self, X):
            Xt = self._lag_stats.transform(X)
            if self._te is not None:
                Xt = self._te.transform(Xt)
                Xt = self._lag_te.transform(Xt)
            return self._pre.transform(Xt)

        def predict(self, X):
            return self.model_.predict(self._transform(X))

        def predict_proba(self, X):
            Xt = self._transform(X)
            if hasattr(self.model_, "predict_proba"):
                return self.model_.predict_proba(Xt)
            raise AttributeError("Underlying model does not support predict_proba()")

    failures = []
    best_est = None
    best_info: Dict[str, Any] = {"best_score": -np.inf}

    # A) Yield-aware direct classifiers
    for cand in _classifier_candidates():
        est = _force_single_thread(cand.estimator)

        wrapped = YieldAwareClassifier(
            model=est,
            numeric_features_base=numeric_features_base,
            categorical_features=categorical_features,
            use_target_mean=USE_TARGET_MEAN,
            smoothing=TARGET_MEAN_SMOOTHING,
            use_sample_weight=USE_SAMPLE_WEIGHT,
            sample_weight_power=SAMPLE_WEIGHT_POWER,
            scale_numeric=cand.scale_numeric,
        )

        n_jobs_outer = cand.search_n_jobs if cand.search_n_jobs not in (0, None) else _default_outer_jobs()
        n_iter = _cap_n_iter(cand.param_space, cand.n_iter)

        search = RandomizedSearchCV(
            wrapped,
            param_distributions=cand.param_space,
            n_iter=n_iter,
            scoring=f1_combo_scorer,
            cv=cv_splits,
            random_state=CONFIG.random_state,
            n_jobs=n_jobs_outer,
            verbose=1,
            pre_dispatch=max(1, 2 * n_jobs_outer),
            error_score=np.nan,
            refit=True,
        )

        try:
            with threadpool_limits(limits=1):
                search.fit(X, y_combo)
        except Exception as e:
            failures.append({"model": cand.name, "error": f"{type(e).__name__}: {e}"})
            continue

        score_cv_mean = float(search.best_score_) if search.best_score_ is not None else float("nan")
        if not np.isfinite(score_cv_mean):
            continue

        if score_cv_mean > float(best_info["best_score"]):
            best_est = search.best_estimator_
            best_info = {
                "task": "classification",
                "approach": "direct_yield_te",
                "fit_y_combo": True,
                "best_model": cand.name,
                "best_score_cv_mean_f1_macro": score_cv_mean,
                "best_params": search.best_params_,
                "numeric_features_used": list(numeric_features_base) + list(LAG_STATS_COLS) + (
                    list(TE_COLS) + list(LAG_TE_INTER_COLS) if USE_TARGET_MEAN else []
                ),
                "dropped_numeric_allnan_or_constant": dropped_allnan_const,
                "best_score": score_cv_mean,  # temporary; replaced by OOF
            }

    # B) Reg->Tertile remains available
    reg2tert_best_est = None
    reg2tert_best_info: Dict[str, Any] = {"best_score": -np.inf}

    reg_cands = [c for c in _regressor_candidates() if c.name in {"HistGB", "XGBoost", "MLP"}]

    numeric_features_reg2tert = _with_lag_stats(numeric_features_base)
    if USE_TARGET_MEAN:
        numeric_features_reg2tert = _with_te(numeric_features_reg2tert)
        numeric_features_reg2tert = _with_lag_te_interactions(numeric_features_reg2tert)

    for rc in reg_cands:
        reg_est = _force_single_thread(rc.estimator)
        reg_pre = _preprocessor(numeric_features_reg2tert, categorical_features, scale_numeric=rc.scale_numeric)

        steps = [("lag_stats", LagStatsFeatures())]
        if USE_TARGET_MEAN and rc.use_target_mean:
            steps.append(("target_means", GroupTargetMeanFeatures(smoothing=TARGET_MEAN_SMOOTHING)))
            steps.append(("lag_te_inter", LagTeInteractionFeatures()))
        steps += [("preprocess", reg_pre), ("model", reg_est)]
        reg_pipe = Pipeline(steps)

        wrapper = _make_reg2tert_wrapper(reg_pipe, X, y_reg)
        param_space = {f"reg_pipeline__{k}": v for k, v in rc.param_space.items()}

        n_jobs_outer = rc.search_n_jobs if rc.search_n_jobs not in (0, None) else _default_outer_jobs()
        n_iter = _cap_n_iter(param_space, rc.n_iter)

        search = RandomizedSearchCV(
            wrapper,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=f1_combo_scorer,
            cv=cv_splits,
            random_state=CONFIG.random_state,
            n_jobs=n_jobs_outer,
            verbose=1,
            pre_dispatch=max(1, 2 * n_jobs_outer),
            error_score=np.nan,
            refit=True,
        )

        try:
            with threadpool_limits(limits=1):
                search.fit(X, y_combo)
        except Exception as e:
            failures.append({"model": f"REG2TERT::{rc.name}", "error": f"{type(e).__name__}: {e}"})
            continue

        score_cv_mean = float(search.best_score_) if search.best_score_ is not None else float("nan")
        if not np.isfinite(score_cv_mean):
            continue

        if score_cv_mean > float(reg2tert_best_info["best_score"]):
            reg2tert_best_est = search.best_estimator_
            reg2tert_best_info = {
                "task": "classification",
                "approach": "regression_to_tertiles",
                "best_model": f"Reg2Tertile({rc.name})",
                "best_score_cv_mean_f1_macro": score_cv_mean,
                "best_params": search.best_params_,
                "numeric_features_used": list(numeric_features_reg2tert),
                "best_score": score_cv_mean,  # temporary; replaced by OOF
            }

    if reg2tert_best_est is not None and float(reg2tert_best_info["best_score"]) > float(best_info.get("best_score", -np.inf)):
        best_est = reg2tert_best_est
        best_info = reg2tert_best_info

    if best_est is None:
        raise RuntimeError(f"All classifier candidates failed (or produced NaN scores). Failures={failures}")

    # final fit
    if best_info.get("approach") == "regression_to_tertiles":
        best_est.fit(X, y_reg)
    elif best_info.get("fit_y_combo", False):
        best_est.fit(X, y_combo)
    else:
        best_est.fit(X, y_class)

    y_pred = best_est.predict(X)

    oof = _oof_eval_classifier(best_est, best_info, X, y_class, y_reg, cv_splits=cv_splits)

    best_info.update({
        "train_accuracy": float(accuracy_score(y_class, y_pred)),
        "train_precision_macro": float(precision_score(y_class, y_pred, average="macro", zero_division=0)),
        "train_recall_macro": float(recall_score(y_class, y_pred, average="macro", zero_division=0)),
        "train_f1_macro": float(f1_score(y_class, y_pred, average="macro", zero_division=0)),
        "oof_f1_macro": float(oof.get("oof_f1_macro", float("nan"))),
        "oof_accuracy": float(oof.get("oof_acc", float("nan"))),
        "oof_n": int(oof.get("oof_n", 0)),
        # final reported best_score is the leak-free OOF metric
        "best_score": float(oof.get("oof_f1_macro", float("nan"))),
        "best_score_type": "oof_f1_macro",
        "model_failures": failures,
        "feature_set_info": feature_set_info,
        "tertiles_mode": TERTILES_MODE,
        "use_target_mean": USE_TARGET_MEAN,
        "target_mean_smoothing": TARGET_MEAN_SMOOTHING,
        "data_signature": sig,
        "code_version": MODEL_SELECTION_VERSION,
        "tune_mode": TUNE_MODE,
        "feature_set": FEATURE_SET,
    })

    info_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    joblib.dump(best_est, model_path)

    return best_est, best_info


def tune_regressor(panel: pd.DataFrame, feature_cols: List[str], out_dir: Path) -> Tuple[Pipeline, Dict[str, Any]]:
    _suppress_noisy_warnings()

    mask = _clean_mask_for_task(panel, feature_cols, task="reg")
    df = panel.loc[mask].sort_values("year").reset_index(drop=True)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "models" / "best_regressor_info.json"
    model_path = out_dir / "models" / "best_regressor.joblib"

    sig_cols = list(feature_cols) + ["region", "crop", "year", "yield"]
    sig = _data_signature(df, sig_cols)

    if USE_CACHE and (not FORCE_RETRAIN) and info_path.exists() and model_path.exists():
        try:
            cached_info = json.loads(info_path.read_text(encoding="utf-8"))
            if (
                cached_info.get("data_signature") == sig
                and cached_info.get("code_version") == MODEL_SELECTION_VERSION
                and cached_info.get("tune_mode") == TUNE_MODE
                and cached_info.get("feature_set") == FEATURE_SET
                and cached_info.get("use_target_mean") == USE_TARGET_MEAN
                and float(cached_info.get("target_mean_smoothing", TARGET_MEAN_SMOOTHING)) == TARGET_MEAN_SMOOTHING
            ):
                est = joblib.load(model_path)
                return est, cached_info
        except Exception:
            pass

    X = _build_X(df, feature_cols)
    y = df["yield"].astype(float).values

    numeric_features_all = list(feature_cols) + ["year"]
    categorical_features = ["region", "crop"]

    _quick_sanity(X, y, name="reg_before_drop")

    X, numeric_features_all, dropped_allnan_const = _drop_all_nan_and_constant_columns(
        X, numeric_features=numeric_features_all, keep_cols_always=["year"]
    )

    cv = YearForwardSplit(
        n_splits=CONFIG.cv.n_splits,
        min_train_samples=CONFIG.cv.min_samples_cv,
        min_test_samples=max(2, CONFIG.cv.min_samples_cv // 2),
    )
    cv_splits = list(cv.split(X, y))

    numeric_features_base, feature_set_info = _choose_numeric_feature_set(
        X, numeric_features_all, cv_splits=cv_splits, task="reg", y_reg=y
    )

    # without TE (just lag stats)
    numeric_features_no_te = _with_lag_stats(numeric_features_base)

    # with TE + interactions (if enabled)
    if USE_TARGET_MEAN:
        numeric_features_with_te = _with_lag_stats(numeric_features_base)
        numeric_features_with_te = _with_te(numeric_features_with_te)
        numeric_features_with_te = _with_lag_te_interactions(numeric_features_with_te)
    else:
        numeric_features_with_te = numeric_features_no_te

    _quick_sanity(X, y, name=f"reg_after_feature_set(thr={feature_set_info.get('missing_threshold')})")

    reg_cands = _regressor_candidates()

    best: Optional[Pipeline] = None
    best_info: Dict[str, Any] = {"best_score": -np.inf}
    failures: List[Dict[str, str]] = []

    for cand in reg_cands:
        est = _force_single_thread(cand.estimator)

        use_te = bool(USE_TARGET_MEAN and cand.use_target_mean)
        numeric_features_used = numeric_features_with_te if use_te else numeric_features_no_te

        pre = _preprocessor(numeric_features_used, categorical_features, scale_numeric=cand.scale_numeric)

        steps = [("lag_stats", LagStatsFeatures())]
        if use_te:
            steps.append(("target_means", GroupTargetMeanFeatures(smoothing=TARGET_MEAN_SMOOTHING)))
            steps.append(("lag_te_inter", LagTeInteractionFeatures()))
        steps += [("preprocess", pre), ("model", est)]
        pipe = Pipeline(steps)

        n_jobs_outer = cand.search_n_jobs if cand.search_n_jobs not in (0, None) else _default_outer_jobs()
        n_iter = _cap_n_iter(cand.param_space, cand.n_iter)

        search = RandomizedSearchCV(
            pipe,
            param_distributions=cand.param_space,
            n_iter=n_iter,
            scoring="r2",
            cv=cv_splits,
            random_state=CONFIG.random_state,
            n_jobs=n_jobs_outer,
            verbose=1,
            pre_dispatch=max(1, 2 * n_jobs_outer),
            error_score=np.nan,
            refit=True,
        )

        try:
            with threadpool_limits(limits=1):
                search.fit(X, y)
        except Exception as e:
            failures.append({"model": cand.name, "error": f"{type(e).__name__}: {e}"})
            continue

        score_cv_mean = float(search.best_score_) if search.best_score_ is not None else float("nan")
        if not np.isfinite(score_cv_mean):
            continue

        if score_cv_mean > float(best_info["best_score"]):
            best = search.best_estimator_
            best_info = {
                "task": "regression",
                "best_model": cand.name,
                "best_score_cv_mean_r2": score_cv_mean,
                "best_params": search.best_params_,
                "n_samples": int(len(X)),
                "n_features_numeric_used": int(len(numeric_features_used)),
                "numeric_features_used": list(numeric_features_used),
                "dropped_numeric_allnan_or_constant": dropped_allnan_const,
                "outer_n_jobs": int(n_jobs_outer),
                "cv_n_splits": int(CONFIG.cv.n_splits),
                "model_failures": failures,
                "feature_set_info": feature_set_info,
                "best_score": score_cv_mean,  # temporary; replaced by OOF
            }

    if best is None:
        raise RuntimeError("All regressor candidates failed (or produced NaN scores).")

    best.fit(X, y)
    y_pred = best.predict(X)

    oof = _oof_eval_regressor(best, X, y, cv_splits=cv_splits)

    best_info.update({
        "train_rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "train_r2": float(r2_score(y, y_pred)),
        "oof_r2": float(oof.get("oof_r2", float("nan"))),
        "oof_rmse": float(oof.get("oof_rmse", float("nan"))),
        "oof_n": int(oof.get("oof_n", 0)),
        "best_score": float(oof.get("oof_r2", float("nan"))),  # final reported score
        "best_score_type": "oof_r2",
        "use_target_mean": USE_TARGET_MEAN,
        "target_mean_smoothing": TARGET_MEAN_SMOOTHING,
        "data_signature": sig,
        "code_version": MODEL_SELECTION_VERSION,
        "tune_mode": TUNE_MODE,
        "feature_set": FEATURE_SET,
    })

    info_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    joblib.dump(best, model_path)

    return best, best_info


# --------------------------
# Public: permutation importance (kept, generalized)
# --------------------------
def compute_permutation_importance(
    panel: pd.DataFrame,
    feature_cols: List[str],
    fitted_model: Optional[Pipeline] = None,
    *,
    # backward-compat alias (run_pipeline.py passes model=...)
    model: Optional[Any] = None,
    task: str = "reg",          # "reg" or "clf"
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Permutation importance on the full fitted pipeline.

    Backward compatible:
      - preferred: fitted_model=...
      - also supports: model=... (older caller)

    For tuned models, X must include:
      - numeric: feature_cols + ['year']
      - categorical: ['region', 'crop']
    """
    # accept either keyword
    if fitted_model is None:
        fitted_model = model
    if fitted_model is None:
        raise TypeError("compute_permutation_importance requires a fitted model via fitted_model=... or model=...")

    X_num_raw = panel[feature_cols].apply(pd.to_numeric, errors="coerce")

    mask_nonempty = ~X_num_raw.isna().all(axis=1)
    mask_cat = panel["region"].notna() & panel["crop"].notna()
    mask_year = panel["year"].notna()

    if task == "reg":
        mask_y = panel["yield"].notna()
        y = panel.loc[mask_nonempty & mask_cat & mask_year & mask_y, "yield"].astype(float)
        scoring = None  # uses estimator.score (R2 for regressors)
    else:
        mask_y = panel["yield_class"].notna()
        y = panel.loc[mask_nonempty & mask_cat & mask_year & mask_y, "yield_class"].astype(int)
        scoring = make_scorer(f1_score, average="macro", zero_division=0)

    mask = mask_nonempty & mask_cat & mask_year & mask_y

    X_num = X_num_raw.loc[mask]
    cat_data = panel.loc[mask, ["region", "crop"]]
    year_data = panel.loc[mask, ["year"]].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([X_num, cat_data, year_data], axis=1)

    if X.empty or len(X) < 3:
        return None

    result = permutation_importance(
        fitted_model,
        X,
        y,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        scoring=scoring,
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
    return imp_df



def compute_gb_feature_importance(
    panel,
    feature_cols,
    model,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
):
    # Backwards-compatible name used by run_pipeline.py
    return compute_permutation_importance(
        panel=panel,
        feature_cols=feature_cols,
        fitted_model=model,   # correct keyword
        task="reg",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
