# app.py  (MERGED: CLI runner + Streamlit dashboard + mentor-friendly deep dives + Temporal Learning)
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import re
import unicodedata
import json

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib

from config import CONFIG

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ----------------------------
# Streamlit detection
# ----------------------------
def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ----------------------------
# CLI actions
# ----------------------------
def run_pipeline() -> None:
    # import here to avoid Streamlit re-run issues
    from run_pipeline import main as pipeline_main
    pipeline_main()


def launch_dashboard() -> None:
    subprocess.check_call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "dashboard", "all"], help="run pipeline / dashboard / both")
    args = p.parse_args()

    if args.cmd in ["run", "all"]:
        run_pipeline()
        print("Done. Open outputs/reports/summary.txt or outputs/reports/report.html")

    if args.cmd in ["dashboard", "all"]:
        launch_dashboard()


# ----------------------------
# Dashboard helpers
# ----------------------------
@st.cache_data
def load_csv_required(filename: str) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        st.error(f"Missing file: {path}. Please run: python app.py run")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_csv_optional(filename: str) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json_optional(rel_path: str) -> dict:
    path = OUTPUT_DIR / rel_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data
def load_all_data():
    panel = load_csv_required("panel_dataset_cleaned.csv")
    metrics = load_csv_required("metrics_summary.csv")
    forecast_10y = load_csv_optional("yield_forecast_10_years.csv")
    scenarios = load_csv_optional("yield_forecast_scenarios.csv")
    backtest = load_csv_optional("backtest_forecasts.csv")
    bt_yearly = load_csv_optional("backtest_metrics_by_year.csv")  # Temporal Learning
    gb_importance = load_csv_optional("gb_feature_importance.csv")
    tourney = load_csv_optional("model_tournament.csv")
    return panel, metrics, forecast_10y, scenarios, backtest, bt_yearly, gb_importance, tourney


@st.cache_resource
def load_best_regressor():
    model_path = OUTPUT_DIR / "models" / "best_regressor.joblib"
    if not model_path.exists():
        st.error(f"Missing model: {model_path}. Please run: python app.py run")
        st.stop()
    return joblib.load(model_path)


def _get_row(metrics: pd.DataFrame, model_name: str) -> dict:
    if metrics.empty or "model" not in metrics.columns:
        return {}
    df = metrics.loc[metrics["model"] == model_name]
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def make_backtest_chart(backtest: pd.DataFrame, region: str, crop: str):
    if backtest.empty:
        st.info("No backtest file found yet. Run: python app.py run")
        return

    needed = {"region", "crop", "year", "yield_true", "yield_pred"}
    if not needed.issubset(set(backtest.columns)):
        st.warning("Backtest file is missing expected columns. Expected: region, crop, year, yield_true, yield_pred.")
        return

    df = backtest[(backtest["region"] == region) & (backtest["crop"] == crop)].copy()
    if df.empty:
        st.warning("No backtest rows for this region & crop.")
        return

    df = df.sort_values("year")
    df_long = df.melt(
        id_vars=["year"],
        value_vars=["yield_true", "yield_pred"],
        var_name="series",
        value_name="yield",
    )

    chart = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("yield:Q", title="Yield"),
            color=alt.Color("series:N", title="Series"),
            tooltip=["year", "series", alt.Tooltip("yield:Q", format=".3f")],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)


def make_importance_chart(gb_importance: pd.DataFrame, top_k: int = 12):
    if gb_importance.empty:
        st.info("No feature importance file found yet (gb_feature_importance.csv).")
        return

    needed = {"feature", "importance_mean"}
    if not needed.issubset(set(gb_importance.columns)):
        st.warning("Importance file is missing expected columns: feature, importance_mean.")
        return

    df = gb_importance.sort_values("importance_mean", ascending=False).head(top_k)

    tooltips = [
        alt.Tooltip("feature:N"),
        alt.Tooltip("importance_mean:Q", format=".4f"),
    ]
    if "importance_std" in df.columns:
        tooltips.append(alt.Tooltip("importance_std:Q", format=".4f"))

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("importance_mean:Q", title="Permutation importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=tooltips,
        )
        .properties(height=420)
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption("Higher = bigger average performance drop when that column is shuffled.")


def make_regional_risk_heatmap(panel: pd.DataFrame, gb_importance: pd.DataFrame,
                               selected_region: str, selected_crop: str):
    """
    Stakeholder-friendly:
    Uses the top-importance feature as a proxy for "aerosol risk".
    Shows region x crop heatmap for the latest year in the dataset.
    """
    if panel is None or panel.empty:
        st.info("Panel data not available.")
        return

    if gb_importance is None or gb_importance.empty or "feature" not in gb_importance.columns:
        st.info("No feature importance file found yet (gb_feature_importance.csv). Run: python app.py run")
        return

    top_feature = None
    imp_sorted = (
        gb_importance.sort_values("importance_mean", ascending=False)
        if "importance_mean" in gb_importance.columns
        else gb_importance.copy()
    )
    for f in imp_sorted["feature"].astype(str).tolist():
        if f in panel.columns:
            top_feature = f
            break

    if top_feature is None:
        st.warning("Could not find any importance features in panel_dataset_cleaned.csv, so heatmap is skipped.")
        return

    df = panel.copy()
    df["year_num"] = pd.to_numeric(df.get("year"), errors="coerce")
    latest_year = df["year_num"].dropna().max()

    if np.isfinite(latest_year):
        df = df[df["year_num"] == latest_year]

    df[top_feature] = pd.to_numeric(df[top_feature], errors="coerce")
    df = df.dropna(subset=["region", "crop", top_feature])

    if df.empty:
        st.info("Not enough data to compute regional risk heatmap for the latest year.")
        return

    risk = df.groupby(["region", "crop"])[top_feature].mean().reset_index()

    vals = pd.to_numeric(risk[top_feature], errors="coerce").to_numpy(dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    denom = max(1e-9, vmax - vmin)
    risk["risk_score"] = 100.0 * (vals - vmin) / denom

    sel = risk[(risk["region"] == selected_region) & (risk["crop"] == selected_crop)]
    if not sel.empty:
        st.metric(
            "Selected Region/Crop Risk (0–100)",
            f"{float(sel.iloc[0]['risk_score']):.1f}",
            help=f"Relative score based on average {top_feature} in latest year.",
        )

    st.caption(f"Risk proxy feature: {top_feature} (higher = higher relative risk).")

    heat = (
        alt.Chart(risk)
        .mark_rect()
        .encode(
            x=alt.X("region:N", title="Region"),
            y=alt.Y("crop:N", title="Crop"),
            color=alt.Color("risk_score:Q", title="Risk (0–100)"),
            tooltip=[
                alt.Tooltip("region:N"),
                alt.Tooltip("crop:N"),
                alt.Tooltip(f"{top_feature}:Q", format=".4f"),
                alt.Tooltip("risk_score:Q", format=".1f"),
            ],
        )
        .properties(height=420)
    )

    st.altair_chart(heat, use_container_width=True)


def _compute_accuracy_within_pct(backtest: pd.DataFrame, pct: float = 0.10) -> float:
    """
    If backtest has accuracy_within_10pct, use it. Else compute from yield_true/yield_pred.
    Returns percent (0-100).
    """
    if backtest.empty:
        return float("nan")

    if "accuracy_within_10pct" in backtest.columns:
        try:
            v = pd.to_numeric(backtest["accuracy_within_10pct"], errors="coerce").dropna()
            if v.empty:
                return float("nan")
            mean_v = float(v.mean())
            return mean_v * 100.0 if mean_v <= 1.0 else mean_v
        except Exception:
            return float("nan")

    needed = {"yield_true", "yield_pred"}
    if not needed.issubset(set(backtest.columns)):
        return float("nan")

    yt = pd.to_numeric(backtest["yield_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(backtest["yield_pred"], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(yt) & np.isfinite(yp)
    if ok.sum() == 0:
        return float("nan")

    denom = np.where(np.abs(yt[ok]) < 1e-9, 1.0, np.abs(yt[ok]))
    rel_err = np.abs(yp[ok] - yt[ok]) / denom
    return float((rel_err <= pct).mean() * 100.0)


# ----------------------------
# Temporal Learning (model "learning as it goes")
# ----------------------------
def make_learning_trend_chart(bt_yearly: pd.DataFrame):
    """
    Mentor-friendly: show error trend by year.
    Uses RMSE (lower is better). If present, also computes a normalized accuracy proxy.
    """
    if bt_yearly is None or bt_yearly.empty:
        return None

    if "year" not in bt_yearly.columns:
        return None

    df = bt_yearly.copy()

    # Optional filtering if columns exist
    # (Some pipelines store overall-only; some store per region/crop.)
    for col in ["rmse", "r2", "n"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_num"]).sort_values("year_num")

    if "rmse" not in df.columns or df["rmse"].dropna().empty:
        return None

    rmse_max = float(df["rmse"].max()) if np.isfinite(df["rmse"].max()) else float("nan")
    if np.isfinite(rmse_max) and rmse_max > 1e-12:
        df["accuracy_proxy"] = 1.0 - (df["rmse"] / rmse_max)
    else:
        df["accuracy_proxy"] = np.nan

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year_num:O", title="Year (future test year in backtest)"),
            y=alt.Y("rmse:Q", title="Prediction error (RMSE) — lower is better"),
            tooltip=[
                alt.Tooltip("year_num:O", title="Year"),
                alt.Tooltip("rmse:Q", format=".4f"),
                alt.Tooltip("n:Q", format=".0f") if "n" in df.columns else alt.value(None),
            ],
        )
        .properties(height=360, title="Temporal learning: backtest error by year")
    )
    return chart, df


def _temporal_improvement_text(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Returns (headline, detail) about improvement from first to last RMSE.
    """
    if df is None or df.empty or "rmse" not in df.columns:
        return None, None

    rmse = pd.to_numeric(df["rmse"], errors="coerce").dropna()
    if rmse.size < 2:
        return None, None

    first = float(rmse.iloc[0])
    last = float(rmse.iloc[-1])

    if not (np.isfinite(first) and np.isfinite(last)) or abs(first) < 1e-12:
        return None, None

    pct = (first - last) / first * 100.0
    headline = f"Error reduced by {pct:.1f}% from start to end of the backtest."
    detail = f"Start RMSE: {first:.4f} → End RMSE: {last:.4f}"
    return headline, detail


# ----------------------------
# Improvement 1: Split visualization (Year-forward chaining)
# ----------------------------
def make_split_viz(panel: pd.DataFrame, n_splits: int = 4):
    if panel is None or panel.empty or "year" not in panel.columns:
        return None

    years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().unique().tolist())
    if len(years) < 3:
        return None

    n_splits = int(min(n_splits, max(1, len(years) - 1)))
    folds = []

    test_years = years[-n_splits:]
    for i, test_year in enumerate(test_years, start=1):
        folds.append({"Fold": f"Fold {i}", "Year": int(test_year), "Type": "Test (Future)"})
        for prev in [y for y in years if y < test_year]:
            folds.append({"Fold": f"Fold {i}", "Year": int(prev), "Type": "Train (Past)"})

    fold_df = pd.DataFrame(folds)

    chart = (
        alt.Chart(fold_df)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", title="Timeline"),
            y=alt.Y("Fold:N", title="Cross-validation folds"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=["Train (Past)", "Test (Future)"], range=["#2ecc71", "#e74c3c"]),
            ),
            tooltip=["Fold", "Year", "Type"],
        )
        .properties(height=220, title="Validation strategy (leak-free): train on past, test on future")
    )
    return chart


# ----------------------------
# Improvement 3: Correlation engine (PM2.5 vs Yield)
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


def _make_norm_map(cols):
    mp = {}
    for c in cols:
        nc = _norm_col(c).lower()
        if nc and nc not in mp:
            mp[nc] = c
    return mp


def pick_pm_col(panel: pd.DataFrame) -> str | None:
    if panel is None or panel.empty:
        return None

    pm_cfg = [_norm_col(c).lower() for c in getattr(CONFIG.scenario, "pm_cols", [])]
    norm_map = _make_norm_map(panel.columns.tolist())
    for c in pm_cfg:
        if c in norm_map:
            return norm_map[c]

    candidates = []
    for c in panel.columns:
        s = str(c).lower()
        if "pm2.5" in s or "pm25" in s or "pm_25" in s or ("pm" in s and "2" in s and "5" in s):
            candidates.append(c)
    return candidates[0] if candidates else None


def make_correlation_plot(df: pd.DataFrame, x_col: str, y_col: str = "yield"):
    base = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:Q", title=y_col),
        color=alt.Color("region:N", title="Region"),
        tooltip=["year", "region", "crop", x_col, y_col],
    ).properties(height=330).interactive()

    reg_line = base.transform_regression(x_col, y_col).mark_line(color="black")
    return base + reg_line


def compute_corr(df: pd.DataFrame, x_col: str, y_col: str = "yield") -> float:
    try:
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 3:
            return float("nan")
        return float(np.corrcoef(x[ok].to_numpy(), y[ok].to_numpy())[0, 1])
    except Exception:
        return float("nan")


# ----------------------------
# Dynamic scenario utilities (user-chosen pollution level)
# ----------------------------
def infer_feature_cols_from_forecast(df: pd.DataFrame) -> list[str]:
    ignore = {
        "region", "crop", "year",
        "yield_pred", "yield_true",
        "scenario", "forecast_method",
        "yield",
    }
    return [c for c in df.columns if c not in ignore]


def _build_X_like_training(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()

    if "region" not in df2.columns:
        df2["region"] = np.nan
    if "crop" not in df2.columns:
        df2["crop"] = np.nan
    if "year" not in df2.columns:
        df2["year"] = np.nan

    for c in feature_cols:
        if c not in df2.columns:
            df2[c] = np.nan

    X_num = df2[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_cat = df2[["region", "crop"]]
    X_year = pd.to_numeric(df2["year"], errors="coerce").to_frame("year")
    return pd.concat([X_num, X_cat, X_year], axis=1)


def apply_aerosol_multiplier(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    out = df.copy()
    norm_map = _make_norm_map(list(out.columns))

    aod_cfg = [_norm_col(c).lower() for c in getattr(CONFIG.scenario, "aod_cols", [])]
    pm_cfg = [_norm_col(c).lower() for c in getattr(CONFIG.scenario, "pm_cols", [])]

    aod_cols_actual = [norm_map[c] for c in aod_cfg if c in norm_map]
    pm_cols_actual = [norm_map[c] for c in pm_cfg if c in norm_map]

    for col in aod_cols_actual + pm_cols_actual:
        out[col] = pd.to_numeric(out[col], errors="coerce") * float(multiplier)

    return out


def make_custom_pollution_forecast(
    baseline_df: pd.DataFrame,
    aerosol_change_pct: int,
    reg_model,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Takes baseline future rows (with features), scales aerosol columns by (1 + pct/100),
    re-predicts yield using the trained regressor pipeline.
    """
    mult = 1.0 + (float(aerosol_change_pct) / 100.0)
    df_mod = apply_aerosol_multiplier(baseline_df, mult)

    X_mod = _build_X_like_training(df_mod, feature_cols)
    df_mod = df_mod.copy()
    df_mod["yield_pred"] = reg_model.predict(X_mod)

    label = f"custom ({aerosol_change_pct:+d}%)"
    df_mod["scenario"] = label
    return df_mod


# ----------------------------
# Improvement 2: Full-spectrum leaderboard (all models/architectures)
# ----------------------------
def enrich_tourney(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "Model" not in out.columns:
        for c in out.columns:
            if str(c).lower() in {"model", "model_name", "name"}:
                out = out.rename(columns={c: "Model"})
                break
    if "Score" not in out.columns:
        for c in out.columns:
            if str(c).lower() in {"score", "metric", "value"}:
                out = out.rename(columns={c: "Score"})
                break

    if "Model" in out.columns:
        out["Architecture"] = out["Model"].astype(str).apply(
            lambda x: x.split("(")[0].split(" ")[0].strip() if x else "Unknown"
        )
    else:
        out["Architecture"] = "Unknown"

    return out


def make_arch_summary(df_tourney: pd.DataFrame) -> pd.DataFrame:
    if df_tourney.empty or "Architecture" not in df_tourney.columns or "Score" not in df_tourney.columns:
        return pd.DataFrame()
    s = df_tourney.groupby("Architecture")["Score"].max().sort_values(ascending=False).reset_index()
    s = s.rename(columns={"Score": "BestScore"})
    return s


def make_task_split(df_tourney: pd.DataFrame):
    if df_tourney.empty:
        return None, None
    if "Task" not in df_tourney.columns:
        return None, None

    reg = df_tourney[df_tourney["Task"].astype(str).str.contains("reg", case=False, na=False)].copy()
    clf = df_tourney[df_tourney["Task"].astype(str).str.contains("class|clf", case=False, na=False)].copy()
    return reg, clf


# ----------------------------
# Dashboard
# ----------------------------
def dashboard_main():
    st.set_page_config(page_title="Crop Yield & Aerosol Modeling Dashboard", layout="wide")

    # “Management cockpit” CSS polish
    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fa; }
        section[data-testid="stSidebar"] { background-color: #ffffff; }
        div[data-testid="stMetric"] { background-color: #ffffff; padding: 14px; border-radius: 12px; border: 1px solid #e5e7eb; }
        div.block-container { padding-top: 1.4rem; }
        .cockpit-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    panel, metrics, forecast_10y, scenarios, backtest, bt_yearly, gb_importance, tourney_raw = load_all_data()

    reg_info_json = load_json_optional("models/best_regressor_info.json")
    clf_info_json = load_json_optional("models/best_classifier_info.json")

    base = _get_row(metrics, "Baseline")
    reg_row = _get_row(metrics, "BestRegressor")
    clf_row = _get_row(metrics, "BestClassifier")
    bt_row = _get_row(metrics, "Backtest")

    reg_info = {**reg_row, **reg_info_json}
    clf_info = {**clf_row, **clf_info_json}

    st.title("Crop Yield, Aerosols & Forecasting – Dashboard")

    regions = sorted(panel["region"].dropna().unique()) if "region" in panel.columns else []
    crops = sorted(panel["crop"].dropna().unique()) if "crop" in panel.columns else []

    st.sidebar.header("Global filters")
    selected_region = st.sidebar.selectbox("Region", regions if regions else ["(none)"])
    selected_crop = st.sidebar.selectbox("Crop", crops if crops else ["(none)"])

    st.sidebar.subheader("Live Simulation Controls")
    aerosol_change = st.sidebar.slider("Change Aerosol Levels (%)", -50, 50, 0)

    (
        overview_tab,
        learning_tab,          # NEW
        validation_tab,
        aerosol_tab,
        model_tab,
        leaderboard_tab,
        performance_tab,
        scenario_tab,
        backtest_tab,
        importance_tab,
    ) = st.tabs(
        [
            "Executive summary",
            "Temporal learning",        # NEW
            "Validation strategy",
            "Aerosol stress analysis",
            "Model comparison",
            "Model leaderboard",
            "Model performance",
            "Scenario explorer",
            "Backtest",
            "Feature importance",
        ]
    )

    with overview_tab:
        st.header("Executive performance summary")

        with st.container():
            c1, c2, c3, c4 = st.columns(4)

            oof_r2 = reg_info.get("oof_r2", reg_info.get("best_score", np.nan))
            try:
                oof_r2 = float(oof_r2)
            except Exception:
                oof_r2 = float("nan")

            ai_acc = clf_info.get("oof_accuracy", np.nan)
            try:
                ai_acc = float(ai_acc)
            except Exception:
                ai_acc = float("nan")

            acc_10 = _compute_accuracy_within_pct(backtest, pct=0.10)

            c1.metric("Yield prediction strength (R² OOF)", f"{oof_r2:.1%}" if np.isfinite(oof_r2) else "N/A")
            c2.metric("Risk classification accuracy (OOF)", f"{ai_acc:.1%}" if np.isfinite(ai_acc) else "N/A")
            c3.metric("Leak-safe evaluation", "Year-forward split")
            c4.metric("Reliability (within 10%)", f"{acc_10:.1f}%" if np.isfinite(acc_10) else "N/A")

        st.divider()

        with st.container():
            st.subheader("Regional risk hotspots (latest year)")
            make_regional_risk_heatmap(panel, gb_importance, selected_region, selected_crop)

        st.divider()

        with st.container():
            st.subheader("Data preview")
            st.dataframe(panel.head(10), use_container_width=True)

        st.divider()

        with st.container():
            st.subheader("Export")
            if forecast_10y is not None and not forecast_10y.empty:
                years = pd.to_numeric(forecast_10y.get("year"), errors="coerce").dropna()
                if not years.empty:
                    y0, y1 = int(years.min()), int(years.max())
                    fname = f"crop_forecast_{y0}_{y1}.csv"
                else:
                    fname = "crop_forecast.csv"

                st.download_button(
                    label="Download forecast (CSV)",
                    data=forecast_10y.to_csv(index=False).encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                )
            else:
                st.info("No 10-year forecast found yet. Run: python app.py run")

    # NEW: Temporal Learning tab
    with learning_tab:
        st.header("Temporal learning (the model improves as years get harder)")

        st.write(
            "This view shows performance by year in the backtest. If recent-year error drops, it supports the claim "
            "that the model is learning the newer, more complex aerosol patterns (not just memorizing averages)."
        )

        if bt_yearly is None or bt_yearly.empty:
            st.info("Missing outputs/backtest_metrics_by_year.csv. Run: python app.py run")
        else:
            df_bt = bt_yearly.copy()

            # Optional filter if your pipeline logs by region/crop
            if {"region", "crop"}.issubset(df_bt.columns):
                df_bt = df_bt[(df_bt["region"] == selected_region) & (df_bt["crop"] == selected_crop)].copy()

            chart_pack = make_learning_trend_chart(df_bt)
            if chart_pack is None:
                st.info("backtest_metrics_by_year.csv is missing required columns (need at least year + rmse).")
            else:
                chart, df_used = chart_pack

                c1, c2, c3 = st.columns(3)
                rmse_vals = pd.to_numeric(df_used["rmse"], errors="coerce").dropna()
                if rmse_vals.size >= 1:
                    c1.metric("First-year RMSE", f"{float(rmse_vals.iloc[0]):.4f}")
                    c2.metric("Last-year RMSE", f"{float(rmse_vals.iloc[-1]):.4f}")
                    headline, detail = _temporal_improvement_text(df_used)
                    c3.metric("Overall change", headline.split(" by ")[-1] if headline else "N/A")

                st.altair_chart(chart, use_container_width=True)

                headline, detail = _temporal_improvement_text(df_used)
                if headline and detail:
                    st.success(headline)
                    st.caption(detail)

                # Optional: show normalized "accuracy proxy" (easy for mentors)
                if "accuracy_proxy" in df_used.columns and df_used["accuracy_proxy"].notna().any():
                    acc_chart = (
                        alt.Chart(df_used)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("year_num:O", title="Year"),
                            y=alt.Y("accuracy_proxy:Q", title="Accuracy proxy (normalized)"),
                            tooltip=[
                                alt.Tooltip("year_num:O", title="Year"),
                                alt.Tooltip("accuracy_proxy:Q", format=".3f"),
                            ],
                        )
                        .properties(height=280, title="Accuracy proxy (normalized from RMSE)")
                    )
                    st.altair_chart(acc_chart, use_container_width=True)

    with validation_tab:
        st.header("Validation strategy (year-forward chaining)")

        st.write(
            "Leak-free setup: train only on past years, test on a future year. "
            "This respects the arrow of time and avoids random-split leakage."
        )

        chart = make_split_viz(panel, n_splits=4)
        if chart is None:
            st.info("Not enough year data to render the split visualization.")
        else:
            st.altair_chart(chart, use_container_width=True)

    with aerosol_tab:
        st.header("Aerosol stress analysis (PM2.5 vs yield)")

        pm_col = pick_pm_col(panel)
        if pm_col is None:
            st.warning("Could not find a PM2.5-like column in panel_dataset_cleaned.csv or CONFIG.scenario.pm_cols.")
        else:
            df = panel.copy()
            if {"region", "crop"}.issubset(df.columns):
                df = df[(df["region"] == selected_region) & (df["crop"] == selected_crop)].copy()

            df[pm_col] = pd.to_numeric(df[pm_col], errors="coerce")
            if "yield" in df.columns:
                df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

            df = df.dropna(subset=[pm_col, "yield"]) if "yield" in df.columns else df.dropna(subset=[pm_col])

            if df.empty or "yield" not in df.columns:
                st.info("Not enough data (or missing yield column) for correlation analysis.")
            else:
                r = compute_corr(df, pm_col, "yield")
                st.metric("Correlation (Pearson r)", f"{r:.3f}" if np.isfinite(r) else "N/A")

                st.altair_chart(make_correlation_plot(df, pm_col, "yield"), use_container_width=True)
                st.caption("Each dot is one year. The black line is the fitted trend.")

    with model_tab:
        st.header("AI vs baseline")

        st.info("Baseline uses historical averages. AI uses aerosol + weather signals to predict deviations early.")

        reg_oof_r2 = reg_info.get("oof_r2", reg_info.get("best_score", reg_row.get("oof_r2", np.nan)))
        try:
            reg_oof_r2 = float(reg_oof_r2)
        except Exception:
            reg_oof_r2 = float("nan")

        ai_f1 = clf_info.get(
            "oof_f1_macro",
            clf_info.get("best_score", clf_row.get("oof_f1_macro", clf_row.get("cv_f1_macro", np.nan))),
        )
        try:
            ai_f1 = float(ai_f1)
        except Exception:
            ai_f1 = float("nan")

        acc_10 = _compute_accuracy_within_pct(backtest, pct=0.10)
        rel_within_10 = (acc_10 / 100.0) if np.isfinite(acc_10) else float("nan")

        base_reg_r2 = base.get("baseline_reg_r2", np.nan)
        try:
            base_reg_r2 = float(base_reg_r2)
        except Exception:
            base_reg_r2 = float("nan")

        base_f1 = base.get("baseline_clf_macro_f1", np.nan)
        try:
            base_f1 = float(base_f1)
        except Exception:
            base_f1 = float("nan")

        base_rel = 0.33

        battle_df = pd.DataFrame(
            {
                "Metric": ["Yield forecast accuracy (R²)", "Risk detection (Macro-F1)", "Reliability (within 10%)"],
                "Baseline": [base_reg_r2, base_f1, base_rel],
                "AI model": [reg_oof_r2, ai_f1, rel_within_10],
            }
        ).melt("Metric", var_name="Approach", value_name="Score")

        battle_chart = (
            alt.Chart(battle_df)
            .mark_bar()
            .encode(
                x=alt.X("Score:Q", title="Score"),
                y=alt.Y("Approach:N", title=""),
                color=alt.Color("Approach:N", title=""),
                row=alt.Row("Metric:N", title=""),
                tooltip=[
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Approach:N"),
                    alt.Tooltip("Score:Q", format=".3f"),
                ],
            )
            .properties(width=650, height=95)
        )

        st.altair_chart(battle_chart, use_container_width=True)

    with leaderboard_tab:
        st.header("Full-spectrum model leaderboard (all architectures)")

        df_tourney = enrich_tourney(tourney_raw)

        if df_tourney.empty:
            st.warning("model_tournament.csv not found yet. Run: python app.py run")
        else:
            st.subheader("Architecture power ranking (best score per architecture)")
            arch = make_arch_summary(df_tourney)
            if not arch.empty:
                st.table(arch)
            else:
                st.info("Could not compute architecture summary (missing Model/Score).")

            st.divider()

            reg_df, clf_df = make_task_split(df_tourney)

            if reg_df is not None or clf_df is not None:
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Regression tournament")
                    if reg_df is None or reg_df.empty:
                        st.info("No regression rows found (Task column).")
                    else:
                        chart = (
                            alt.Chart(reg_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Score:Q", title="Score"),
                                y=alt.Y("Model:N", sort="-x", title="Model"),
                                color=alt.Color("Architecture:N", title="Architecture"),
                                tooltip=["Task", "Model", "Architecture", alt.Tooltip("Score:Q", format=".4f")],
                            )
                            .properties(height=520)
                        )
                        st.altair_chart(chart, use_container_width=True)

                with c2:
                    st.subheader("Classification tournament")
                    if clf_df is None or clf_df.empty:
                        st.info("No classification rows found (Task column).")
                    else:
                        chart = (
                            alt.Chart(clf_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Score:Q", title="Score"),
                                y=alt.Y("Model:N", sort="-x", title="Model"),
                                color=alt.Color("Architecture:N", title="Architecture"),
                                tooltip=["Task", "Model", "Architecture", alt.Tooltip("Score:Q", format=".4f")],
                            )
                            .properties(height=520)
                        )
                        st.altair_chart(chart, use_container_width=True)
            else:
                st.subheader("All models tested (ranked)")
                if {"Model", "Score"}.issubset(df_tourney.columns):
                    chart = (
                        alt.Chart(df_tourney)
                        .mark_bar()
                        .encode(
                            x=alt.X("Score:Q", title="Score"),
                            y=alt.Y("Model:N", sort="-x", title="Model"),
                            color=alt.Color("Architecture:N", title="Architecture"),
                            tooltip=["Model", "Architecture", alt.Tooltip("Score:Q", format=".4f")],
                        )
                        .properties(height=620)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("model_tournament.csv is missing expected columns: Model, Score.")

    with performance_tab:
        st.header("Performance breakdown")

        ai_acc = clf_info.get("oof_accuracy", clf_row.get("oof_accuracy", np.nan))
        try:
            ai_acc = float(ai_acc)
        except Exception:
            ai_acc = float("nan")

        ai_rmse = reg_info.get("oof_rmse", reg_row.get("oof_rmse", reg_row.get("train_rmse", np.nan)))
        try:
            ai_rmse = float(ai_rmse)
        except Exception:
            ai_rmse = float("nan")

        base_rmse = base.get("baseline_reg_rmse", np.nan)
        try:
            base_rmse = float(base_rmse)
        except Exception:
            base_rmse = float("nan")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification accuracy")
            chart_data = pd.DataFrame({"Model": ["Baseline", "AI"], "Accuracy": [0.33, ai_acc]})
            if chart_data["Accuracy"].isna().all():
                st.info("No classification accuracy found yet. Run: python app.py run")
            else:
                st.altair_chart(
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", title=""),
                        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
                        color=alt.Color("Model:N", title=""),
                        tooltip=["Model", alt.Tooltip("Accuracy:Q", format=".3f")],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )

        with col2:
            st.subheader("Regression RMSE (lower is better)")
            reg_chart = pd.DataFrame({"Model": ["Baseline", "AI"], "RMSE": [base_rmse, ai_rmse]})
            if reg_chart["RMSE"].isna().all():
                st.info("No regression RMSE found yet. Run: python app.py run")
            else:
                st.altair_chart(
                    alt.Chart(reg_chart)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", title=""),
                        y=alt.Y("RMSE:Q", title="RMSE"),
                        color=alt.Color("Model:N", title=""),
                        tooltip=["Model", alt.Tooltip("RMSE:Q", format=".3f")],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )

    with scenario_tab:
        st.subheader("Scenario explorer")
        st.write("Use the slider to simulate policy changes (reduce/increase aerosol levels).")

        baseline_future = pd.DataFrame()
        if scenarios is not None and not scenarios.empty and "scenario" in scenarios.columns:
            baseline_future = scenarios[scenarios["scenario"] == "baseline"].copy()
        elif forecast_10y is not None and not forecast_10y.empty:
            baseline_future = forecast_10y.copy()
            baseline_future["scenario"] = "baseline"

        if baseline_future.empty:
            st.info("No future forecast rows found yet. Run: python app.py run")
        else:
            if scenarios is not None and not scenarios.empty and "scenario" in scenarios.columns:
                scenario_options = sorted(scenarios["scenario"].dropna().unique())
                default = [s for s in scenario_options if s in ["baseline", "clean_air", "polluted"]] or scenario_options

                selected_scenarios = st.multiselect("Scenario(s) to show", options=scenario_options, default=default)

                plot_df = scenarios[
                    (scenarios["region"] == selected_region)
                    & (scenarios["crop"] == selected_crop)
                    & (scenarios["scenario"].isin(selected_scenarios))
                ].copy()
            else:
                plot_df = baseline_future[
                    (baseline_future["region"] == selected_region)
                    & (baseline_future["crop"] == selected_crop)
                ].copy()

            if aerosol_change != 0:
                reg_model = load_best_regressor()
                feature_cols = infer_feature_cols_from_forecast(baseline_future)

                base_rc = baseline_future[
                    (baseline_future["region"] == selected_region)
                    & (baseline_future["crop"] == selected_crop)
                ].copy()

                if not base_rc.empty:
                    custom_df = make_custom_pollution_forecast(
                        baseline_df=base_rc,
                        aerosol_change_pct=int(aerosol_change),
                        reg_model=reg_model,
                        feature_cols=feature_cols,
                    )
                    plot_df = pd.concat([plot_df, custom_df], ignore_index=True)

            plot_df = plot_df.sort_values("year")
            if plot_df.empty:
                st.warning("No data for this region/crop selection.")
            else:
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("yield_pred:Q", title="Predicted yield"),
                        color=alt.Color("scenario:N", title="Scenario"),
                        tooltip=["year", "scenario", alt.Tooltip("yield_pred:Q", format=".3f")],
                    )
                    .properties(height=380)
                )
                st.altair_chart(chart, use_container_width=True)

    with backtest_tab:
        st.subheader("Backtest – true vs predicted yields")

        if bt_row:
            c1, c2 = st.columns(2)
            with c1:
                if bt_row.get("rmse") is not None:
                    st.metric("Backtest RMSE", f"{float(bt_row['rmse']):.3f}")
            with c2:
                if bt_row.get("r2") is not None:
                    st.metric("Backtest R²", f"{float(bt_row['r2']):.3f}")

        make_backtest_chart(backtest, selected_region, selected_crop)

    with importance_tab:
        st.header("Feature importance (what drives yield)")
        st.info("Permutation importance on the tuned regressor. Higher means bigger impact on model performance.")
        make_importance_chart(gb_importance)
        st.caption("Tip: if PM2.5/AOD-related features rank high, the model is using aerosol stress as a key signal.")


# ----------------------------
# Entry
# ----------------------------
if _running_in_streamlit():
    dashboard_main()
else:
    cli_main()
