# app.py  (MERGED: CLI runner + Streamlit dashboard)
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

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
def load_all_data():
    panel = load_csv_required("panel_dataset_cleaned.csv")
    metrics = load_csv_required("metrics_summary.csv")
    forecast_10y = load_csv_optional("yield_forecast_10_years.csv")
    scenarios = load_csv_optional("yield_forecast_scenarios.csv")
    backtest = load_csv_optional("backtest_forecasts.csv")
    gb_importance = load_csv_optional("gb_feature_importance.csv")
    return panel, metrics, forecast_10y, scenarios, backtest, gb_importance


def _get_row(metrics: pd.DataFrame, model_name: str) -> dict:
    df = metrics.loc[metrics["model"] == model_name]
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def build_f1_df(metrics: pd.DataFrame) -> pd.DataFrame:
    base = _get_row(metrics, "Baseline")
    clf = _get_row(metrics, "BestClassifier")

    rows = []
    if base.get("baseline_clf_macro_f1") is not None:
        rows.append({"model": "Baseline", "macro_f1": float(base["baseline_clf_macro_f1"])})

    # Prefer OOF (leak-free). Fall back to CV if needed.
    f1_val = clf.get("oof_f1_macro", clf.get("cv_f1_macro"))
    if f1_val is not None:
        rows.append({"model": "BestClassifier", "macro_f1": float(f1_val)})

    return pd.DataFrame(rows)


def build_rmse_df(metrics: pd.DataFrame) -> pd.DataFrame:
    base = _get_row(metrics, "Baseline")
    reg = _get_row(metrics, "BestRegressor")
    bt = _get_row(metrics, "Backtest")

    rows = []
    if base.get("baseline_reg_rmse") is not None:
        rows.append({"model": "Baseline", "rmse": float(base["baseline_reg_rmse"])})

    rmse_val = reg.get("oof_rmse", reg.get("train_rmse"))
    if rmse_val is not None:
        rows.append({"model": "BestRegressor (OOF)", "rmse": float(rmse_val)})

    if bt.get("rmse") is not None:
        rows.append({"model": "Backtest", "rmse": float(bt["rmse"])})

    return pd.DataFrame(rows)


def make_bar_chart(df: pd.DataFrame, x: str, y: str, y_title: str):
    if df.empty:
        st.info("No data available for this chart.")
        return None

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title="Model"),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=[x, alt.Tooltip(f"{y}:Q", format=".3f")],
        )
        .properties(height=320)
    )

    labels = chart.mark_text(dy=-8).encode(text=alt.Text(f"{y}:Q", format=".3f"))
    return chart + labels


def make_scenario_chart(scenarios: pd.DataFrame, region: str, crop: str, scenario_list):
    if scenarios.empty:
        st.info("No scenario forecast file found yet. Run: python app.py run")
        return

    df = (
        scenarios[
            (scenarios["region"] == region)
            & (scenarios["crop"] == crop)
            & (scenarios["scenario"].isin(scenario_list))
        ]
        .copy()
        .sort_values("year")
    )

    if df.empty:
        st.warning("No data for this region / crop / scenario selection.")
        return

    chart = (
        alt.Chart(df)
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


def make_backtest_chart(backtest: pd.DataFrame, region: str, crop: str):
    if backtest.empty:
        st.info("No backtest file found yet. Run: python app.py run")
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

    df = gb_importance.sort_values("importance_mean", ascending=False).head(top_k)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("importance_mean:Q", title="Permutation importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("importance_mean:Q", format=".4f"),
                alt.Tooltip("importance_std:Q", format=".4f"),
            ],
        )
        .properties(height=420)
    )

    st.subheader("Global feature importance (tuned regressor)")
    st.altair_chart(chart, use_container_width=True)
    st.caption("Higher = bigger average performance drop when that column is shuffled.")


def dashboard_main():
    st.set_page_config(page_title="Crop Yield & Aerosol Modeling Dashboard", layout="wide")

    panel, metrics, forecast_10y, scenarios, backtest, gb_importance = load_all_data()

    st.title("Crop Yield, Aerosols & Forecasting – Interactive Dashboard")

    regions = sorted(panel["region"].dropna().unique())
    crops = sorted(panel["crop"].dropna().unique())

    st.sidebar.header("Global filters")
    selected_region = st.sidebar.selectbox("Region", regions if regions else ["(none)"])
    selected_crop = st.sidebar.selectbox("Crop", crops if crops else ["(none)"])

    overview_tab, model_tab, scenario_tab, backtest_tab, importance_tab = st.tabs(
        ["Overview", "Model comparison", "Scenario explorer", "Backtest", "Feature importance"]
    )

    with overview_tab:
        summary_path = OUTPUT_DIR / "reports" / "summary.txt"
        if summary_path.exists():
            st.subheader("Quick results summary")
            st.code(summary_path.read_text(encoding="utf-8"))
        else:
            st.info("No summary found yet. Run: python app.py run")

        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{len(panel):,}")
        with c2:
            st.metric("Regions", f"{panel['region'].nunique()}")
        with c3:
            st.metric("Crops", f"{panel['crop'].nunique()}")

        st.markdown("Sample rows (filtered by Region & Crop):")

        filtered = panel[(panel["region"] == selected_region) & (panel["crop"] == selected_crop)]
        st.dataframe((filtered if not filtered.empty else panel).head(25), use_container_width=True)

    with model_tab:
        st.subheader("Model comparison")

        f1_df = build_f1_df(metrics)
        rmse_df = build_rmse_df(metrics)

        st.markdown("Macro-F1 (yield class prediction):")
        ch1 = make_bar_chart(f1_df, "model", "macro_f1", "Macro-F1 (higher is better)")
        if ch1 is not None:
            st.altair_chart(ch1, use_container_width=True)

        st.markdown("RMSE (yield regression):")
        ch2 = make_bar_chart(rmse_df, "model", "rmse", "RMSE (lower is better)")
        if ch2 is not None:
            st.altair_chart(ch2, use_container_width=True)

        clf = _get_row(metrics, "BestClassifier")
        reg = _get_row(metrics, "BestRegressor")
        if clf:
            st.caption(f"BestClassifier model: {clf.get('best_model_name', '(unknown)')}")
        if reg:
            st.caption(f"BestRegressor model: {reg.get('best_model_name', '(unknown)')}")

    with scenario_tab:
        st.subheader("Scenario explorer")

        if scenarios.empty:
            st.info("No scenario forecast file found yet. Run: python app.py run")
        else:
            scenario_options = sorted(scenarios["scenario"].dropna().unique())
            default = [s for s in scenario_options if s in ["baseline", "clean_air", "polluted"]] or scenario_options

            selected_scenarios = st.multiselect(
                "Scenario(s) to show",
                options=scenario_options,
                default=default,
            )

            make_scenario_chart(scenarios, selected_region, selected_crop, selected_scenarios)

    with backtest_tab:
        st.subheader("Backtest – true vs predicted yields")

        bt = _get_row(metrics, "Backtest")
        if bt:
            c1, c2 = st.columns(2)
            with c1:
                if bt.get("rmse") is not None:
                    st.metric("Backtest RMSE", f"{float(bt['rmse']):.3f}")
            with c2:
                if bt.get("r2") is not None:
                    st.metric("Backtest R²", f"{float(bt['r2']):.3f}")
        make_backtest_chart(backtest, selected_region, selected_crop)

    with importance_tab:
        make_importance_chart(gb_importance)


# ----------------------------
# Entry
# ----------------------------
if _running_in_streamlit():
    dashboard_main()
else:
    cli_main()
