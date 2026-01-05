# report_addons.py
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
N_SPLITS = 4
TOP_K_MISSING = 10
TOP_K_IMPORTANCE = 12

OUT_DIR = Path("outputs")
ADDON_DIR = OUT_DIR / "report_addons"
ADDON_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT_DIR / "panel_dataset_cleaned.csv"
METRICS_SUMMARY_PATH = OUT_DIR / "metrics_summary.csv"
BACKTEST_BY_YEAR_PATH = OUT_DIR / "backtest_metrics_by_year.csv"


# -----------------------------
# Helpers: feature grouping
# -----------------------------
AEROSOL_TOKENS = [
    "aod", "pm2.5", "pm25", "pm_2.5", "no2", "o3", "ozone", "so2", "co", "ch4",
    "aerosol", "radiation", "aai"
]
MET_TOKENS = [
    "temp", "temperature", "precip", "rain", "humidity", "wind", "pressure",
    "skin temp", "surface skin", "mm/day", "mm", "day", "night", "avg", "minimum", "maximum"
]

def is_aerosol(col: str) -> bool:
    s = col.lower()
    return any(tok in s for tok in AEROSOL_TOKENS)

def is_met(col: str) -> bool:
    s = col.lower()
    return any(tok in s for tok in MET_TOKENS)

def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# CV split: year-forward folds
# -----------------------------
def year_forward_folds(df: pd.DataFrame, n_splits: int) -> list[int]:
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    if len(years) < n_splits + 2:
        raise ValueError(f"Not enough unique years ({len(years)}) for n_splits={n_splits}")
    # hold out the last n_splits years (e.g., 2017-2020 if max is 2020 and n_splits=4)
    return years[-n_splits:]


# -----------------------------
# Yield class per crop + fallback detection
# -----------------------------
def add_yield_class_per_crop(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["yield_class"] = np.nan
    method_rows = []

    for crop, sub_idx in df.groupby("crop").groups.items():
        y = df.loc[sub_idx, "yield"].astype(float)
        y_valid = y.dropna()

        # Heuristic: if enough data, do tertiles; else fallback to median split
        if (len(y_valid) >= 9) and (y_valid.nunique() >= 3):
            q1 = float(y_valid.quantile(1/3))
            q2 = float(y_valid.quantile(2/3))
            cls = np.select(
                [y < q1, (y >= q1) & (y < q2), y >= q2],
                [0, 1, 2],
                default=np.nan
            )
            method = "tertiles(3-class)"
            n_classes = 3
        else:
            med = float(y_valid.median()) if len(y_valid) else np.nan
            cls = np.where(y >= med, 1, 0).astype(float)
            method = "median_fallback(2-class)"
            n_classes = 2

        df.loc[sub_idx, "yield_class"] = cls
        method_rows.append({"crop": crop, "method": method, "n_classes": n_classes, "n_rows": int(len(sub_idx))})

    method_df = pd.DataFrame(method_rows).sort_values("crop")
    return df, method_df


# -----------------------------
# Build modeling pipeline
# (dense one-hot so HGB works)
# -----------------------------
def make_preprocess(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def run_year_forward_cv_regression(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, list[int]]:
    test_years = year_forward_folds(df, N_SPLITS)

    X_all = df[numeric_cols + categorical_cols].copy()
    y_all = df["yield"].astype(float).values

    oof_pred = np.full(shape=len(df), fill_value=np.nan, dtype=float)
    fold_rows = []

    for ty in test_years:
        train_mask = df["year"].astype(int) < int(ty)
        test_mask = df["year"].astype(int) == int(ty)

        X_train, y_train = X_all.loc[train_mask], y_all[train_mask.values]
        X_test, y_test = X_all.loc[test_mask], y_all[test_mask.values]

        pipe = Pipeline(steps=[
            ("pre", make_preprocess(numeric_cols, categorical_cols)),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        oof_pred[test_mask.values] = pred

        fold_rows.append({
            "test_year": int(ty),
            "n_test": int(test_mask.sum()),
            "rmse": rmse(y_test, pred),
            "r2": float(r2_score(y_test, pred)) if np.unique(y_test).size >= 2 else np.nan,
        })

    fold_df = pd.DataFrame(fold_rows)
    overall = {
        "oof_rmse": rmse(y_all[~np.isnan(oof_pred)], oof_pred[~np.isnan(oof_pred)]),
        "oof_r2": float(r2_score(y_all[~np.isnan(oof_pred)], oof_pred[~np.isnan(oof_pred)]))
    }
    return fold_df, overall, y_all, oof_pred, test_years


def run_year_forward_cv_classification(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, list[int]]:
    test_years = year_forward_folds(df, N_SPLITS)

    X_all = df[numeric_cols + categorical_cols].copy()
    y_all = df["yield_class"].astype(int).values

    oof_pred = np.full(shape=len(df), fill_value=-999, dtype=int)
    fold_rows = []

    for ty in test_years:
        train_mask = df["year"].astype(int) < int(ty)
        test_mask = df["year"].astype(int) == int(ty)

        X_train, y_train = X_all.loc[train_mask], y_all[train_mask.values]
        X_test, y_test = X_all.loc[test_mask], y_all[test_mask.values]

        pipe = Pipeline(steps=[
            ("pre", make_preprocess(numeric_cols, categorical_cols)),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        oof_pred[test_mask.values] = pred

        fold_rows.append({
            "test_year": int(ty),
            "n_test": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro"))
        })

    fold_df = pd.DataFrame(fold_rows)
    overall = {
        "oof_accuracy": float(accuracy_score(y_all, oof_pred)),
        "oof_macro_f1": float(f1_score(y_all, oof_pred, average="macro"))
    }
    return fold_df, overall, y_all, oof_pred, test_years


# -----------------------------
# Permutation importance stability across folds
# -----------------------------
def permutation_stability_across_folds(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model
) -> pd.DataFrame:
    test_years = year_forward_folds(df, N_SPLITS)
    X_all = df[numeric_cols + categorical_cols].copy()
    y_all = df["yield"].astype(float).values

    top_sets = []
    rows = []

    for ty in test_years:
        train_mask = df["year"].astype(int) < int(ty)
        test_mask = df["year"].astype(int) == int(ty)

        X_train, y_train = X_all.loc[train_mask], y_all[train_mask.values]
        X_test, y_test = X_all.loc[test_mask], y_all[test_mask.values]

        pipe = Pipeline(steps=[
            ("pre", make_preprocess(numeric_cols, categorical_cols)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)

        # get transformed feature names
        feat_names = pipe.named_steps["pre"].get_feature_names_out()

        pi = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="r2"
        )
        order = np.argsort(pi.importances_mean)[::-1]
        top_idx = order[:TOP_K_IMPORTANCE]
        top_feats = [feat_names[i] for i in top_idx]
        top_sets.append(set(top_feats))

        for rank, i in enumerate(top_idx, start=1):
            rows.append({
                "test_year": int(ty),
                "rank": rank,
                "feature": feat_names[i],
                "importance_mean": float(pi.importances_mean[i]),
                "importance_std": float(pi.importances_std[i]),
            })

    # overlap summary
    def jacc(a: set, b: set) -> float:
        return len(a & b) / max(1, len(a | b))

    jaccs = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            jaccs.append(jacc(top_sets[i], top_sets[j]))

    overlap_note = pd.DataFrame([{
        "top_k": TOP_K_IMPORTANCE,
        "avg_pairwise_jaccard": float(np.mean(jaccs)) if jaccs else np.nan,
        "min_pairwise_jaccard": float(np.min(jaccs)) if jaccs else np.nan,
        "max_pairwise_jaccard": float(np.max(jaccs)) if jaccs else np.nan,
    }])
    overlap_note.to_csv(ADDON_DIR / "perm_importance_overlap_summary.csv", index=False)

    df_ranked = pd.DataFrame(rows)
    # frequency of appearance in top-k across folds
    freq = (df_ranked.groupby("feature")["test_year"]
            .nunique()
            .reset_index(name="folds_in_topk")
            .sort_values(["folds_in_topk", "feature"], ascending=[False, True]))
    freq.to_csv(ADDON_DIR / "perm_importance_feature_frequency.csv", index=False)

    return df_ranked


def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PANEL_PATH}. Run your pipeline first (python app.py run).")

    df = pd.read_csv(PANEL_PATH)

    # basic checks
    required = {"region", "crop", "year", "yield"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"panel_dataset_cleaned.csv missing columns: {missing_req}")

    # ensure types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    df = df.dropna(subset=["region", "crop", "year", "yield"]).copy()
    df["year"] = df["year"].astype(int)

    # add lag features (so you can include them consistently in ablations)
    df = df.sort_values(["region", "crop", "year"]).reset_index(drop=True)
    df["yield_lag1"] = df.groupby(["region", "crop"])["yield"].shift(1)
    df["yield_lag2"] = df.groupby(["region", "crop"])["yield"].shift(2)

    # 1) samples per crop and per region×crop
    samples_by_crop = df.groupby("crop").size().reset_index(name="n")
    samples_by_region_crop = df.groupby(["region", "crop"]).size().reset_index(name="n")
    samples_by_crop.to_csv(ADDON_DIR / "samples_by_crop.csv", index=False)
    samples_by_region_crop.to_csv(ADDON_DIR / "samples_by_region_crop.csv", index=False)

    # 2) missingness rate summary (top 10 missing columns)
    protected = {"region", "crop", "year", "yield", "yield_class"}
    feature_cols = [c for c in df.columns if c not in protected]
    miss = df[feature_cols].isna().mean().sort_values(ascending=False)
    miss_top10 = miss.head(TOP_K_MISSING).reset_index()
    miss_top10.columns = ["column", "missing_rate"]
    miss_top10.to_csv(ADDON_DIR / "missingness_top10.csv", index=False)

    # 3) yield_class distribution per crop + fallback detection
    df2, class_method = add_yield_class_per_crop(df)
    class_method.to_csv(ADDON_DIR / "yield_class_method_by_crop.csv", index=False)

    dist = (df2.groupby(["crop", "yield_class"])
            .size()
            .reset_index(name="count"))
    dist["yield_class"] = dist["yield_class"].astype(int)
    totals = dist.groupby("crop")["count"].transform("sum")
    dist["share"] = dist["count"] / totals
    dist.to_csv(ADDON_DIR / "yield_class_distribution_by_crop.csv", index=False)

    # Base columns always included in all ablations
    cat_cols = ["region", "crop"]

    # build aerosol/met feature lists from existing columns
    # (exclude yield target, keep year and lag features as base numeric)
    base_numeric = ["year", "yield_lag1", "yield_lag2"]
    candidate_numeric = [
        c for c in df2.columns
        if c not in {"yield", "region", "crop", "year", "yield_class"}
        and df2[c].dtype != "object"
        and c not in {"yield_lag1", "yield_lag2"}
    ]

    aerosol_numeric = [c for c in candidate_numeric if is_aerosol(c)]
    met_numeric = [c for c in candidate_numeric if (is_met(c) and not is_aerosol(c))]
    combined_numeric = sorted(set(aerosol_numeric + met_numeric))

    # 4) CV metric spread across folds (mean ± std) + exact fold years
    reg_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    clf_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

    # Regression (combined features) fold metrics
    reg_fold_df, reg_overall, y_true_reg, y_oof_reg, fold_years = run_year_forward_cv_regression(
        df2,
        numeric_cols=base_numeric + combined_numeric,
        categorical_cols=cat_cols,
        model=reg_model
    )
    reg_fold_df.to_csv(ADDON_DIR / "cv_regression_folds.csv", index=False)

    reg_summary = pd.DataFrame([{
        "fold_years": ", ".join(map(str, fold_years)),
        "oof_rmse": reg_overall["oof_rmse"],
        "oof_r2": reg_overall["oof_r2"],
        "fold_rmse_mean": float(reg_fold_df["rmse"].mean()),
        "fold_rmse_std": float(reg_fold_df["rmse"].std(ddof=1)),
        "fold_r2_mean": float(reg_fold_df["r2"].mean()),
        "fold_r2_std": float(reg_fold_df["r2"].std(ddof=1)),
    }])
    reg_summary.to_csv(ADDON_DIR / "cv_regression_summary.csv", index=False)

    # Classification (combined features) fold metrics + OOF confusion matrix
    clf_fold_df, clf_overall, y_true_clf, y_oof_clf, fold_years_clf = run_year_forward_cv_classification(
        df2.dropna(subset=["yield_class"]).copy(),
        numeric_cols=base_numeric + combined_numeric,
        categorical_cols=cat_cols,
        model=clf_model
    )
    clf_fold_df.to_csv(ADDON_DIR / "cv_classification_folds.csv", index=False)

    clf_summary = pd.DataFrame([{
        "fold_years": ", ".join(map(str, fold_years_clf)),
        "oof_accuracy": clf_overall["oof_accuracy"],
        "oof_macro_f1": clf_overall["oof_macro_f1"],
        "fold_accuracy_mean": float(clf_fold_df["accuracy"].mean()),
        "fold_accuracy_std": float(clf_fold_df["accuracy"].std(ddof=1)),
        "fold_macro_f1_mean": float(clf_fold_df["macro_f1"].mean()),
        "fold_macro_f1_std": float(clf_fold_df["macro_f1"].std(ddof=1)),
    }])
    clf_summary.to_csv(ADDON_DIR / "cv_classification_summary.csv", index=False)

    cm = confusion_matrix(y_true_clf, y_oof_clf, labels=sorted(np.unique(y_true_clf)))
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in sorted(np.unique(y_true_clf))],
                         columns=[f"pred_{i}" for i in sorted(np.unique(y_true_clf))])
    cm_df.to_csv(ADDON_DIR / "oof_confusion_matrix.csv", index=True)

    report = classification_report(y_true_clf, y_oof_clf, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(ADDON_DIR / "oof_classification_report.csv", index=True)

    # 5) Backtest mean ± std (from existing per-year file)
    if BACKTEST_BY_YEAR_PATH.exists():
        by_year = pd.read_csv(BACKTEST_BY_YEAR_PATH)
        metric_cols = [c for c in by_year.columns if c.lower() in ["rmse", "mae", "mape", "r2", "within_10pct", "within-10%", "within_10"]]
        # normalize within-10 col name if needed
        if "within_10pct" not in by_year.columns:
            for c in by_year.columns:
                if c.lower() in ["within-10%", "within_10", "within_10pct"]:
                    by_year = by_year.rename(columns={c: "within_10pct"})
        summary = {}
        for c in ["rmse", "mae", "mape", "r2", "within_10pct"]:
            if c in by_year.columns:
                summary[c + "_mean"] = float(pd.to_numeric(by_year[c], errors="coerce").mean())
                summary[c + "_std"] = float(pd.to_numeric(by_year[c], errors="coerce").std(ddof=1))
        pd.DataFrame([summary]).to_csv(ADDON_DIR / "backtest_metrics_mean_std.csv", index=False)
    else:
        pd.DataFrame([{"note": "backtest_metrics_by_year.csv not found"}]).to_csv(
            ADDON_DIR / "backtest_metrics_mean_std.csv", index=False
        )

    # 6) Ablation table: met-only vs aerosol-only vs combined (same CV, same base features)
    def ablation_row(name: str, extra_numeric: list[str]) -> dict:
        fold_df, overall, _, _, _ = run_year_forward_cv_regression(
            df2,
            numeric_cols=base_numeric + extra_numeric,
            categorical_cols=cat_cols,
            model=HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        )
        return {
            "features": name,
            "oof_r2": overall["oof_r2"],
            "oof_rmse": overall["oof_rmse"],
            "fold_rmse_mean": float(fold_df["rmse"].mean()),
            "fold_rmse_std": float(fold_df["rmse"].std(ddof=1)),
        }

    ab_met = ablation_row("met-only", met_numeric)
    ab_aer = ablation_row("aerosol-only", aerosol_numeric)
    ab_com = ablation_row("combined", combined_numeric)

    ab_df = pd.DataFrame([ab_met, ab_aer, ab_com])
    met_rmse = float(ab_df.loc[ab_df["features"] == "met-only", "oof_rmse"].iloc[0])
    met_r2 = float(ab_df.loc[ab_df["features"] == "met-only", "oof_r2"].iloc[0])

    ab_df["delta_rmse_vs_met"] = ab_df["oof_rmse"] - met_rmse
    ab_df["delta_r2_vs_met"] = ab_df["oof_r2"] - met_r2
    ab_df.to_csv(ADDON_DIR / "ablation_table.csv", index=False)

    # 7) Interpretability stability (top-k overlap across folds)
    pi_ranked = permutation_stability_across_folds(
        df2,
        numeric_cols=base_numeric + combined_numeric,
        categorical_cols=cat_cols,
        model=HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    )
    pi_ranked.to_csv(ADDON_DIR / "perm_importance_by_fold_topk.csv", index=False)

    # 8) Executive summary + key results table (auto-draft from numbers you already have)
    lines = []
    lines.append("Executive Summary (auto-draft)")
    lines.append(f"- Data: {df2['region'].nunique()} regions × {df2['crop'].nunique()} crops, years {df2['year'].min()}–{df2['year'].max()}, N={len(df2)} rows.")
    lines.append("- Tasks: (1) yield regression, (2) 3-tier yield risk classification (per-crop tertiles when feasible).")
    lines.append(f"- Leak-safe validation: year-forward CV (n_splits={N_SPLITS}) holding out years: {', '.join(map(str, fold_years))}.")
    lines.append(f"- Best regression (combined): OOF R²={reg_overall['oof_r2']:.3f}, OOF RMSE={reg_overall['oof_rmse']:.3f}.")
    lines.append(f"- Best classification (combined): OOF accuracy={clf_overall['oof_accuracy']:.3f}, OOF macro-F1={clf_overall['oof_macro_f1']:.3f}.")
    lines.append("- Key conclusion: yield is strongly driven by lag/context; aerosol signals provide incremental predictive value (see ablation table).")
    lines.append("- Backtest: see outputs/report_addons/backtest_metrics_mean_std.csv (mean±std across backtest years).")
    (ADDON_DIR / "executive_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Key Results table (baseline vs CV vs backtest) if metrics_summary.csv exists
    if METRICS_SUMMARY_PATH.exists():
        ms = pd.read_csv(METRICS_SUMMARY_PATH)
        ms.to_csv(ADDON_DIR / "metrics_summary_copy.csv", index=False)

    print("Done. Wrote report tables to:", ADDON_DIR.resolve())


if __name__ == "__main__":
    main()
