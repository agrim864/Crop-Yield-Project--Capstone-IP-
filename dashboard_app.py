import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Crop Yield & Aerosol Modeling Dashboard",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        st.error(f"Missing file: {path}. Please run run_pipeline.py first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_all_data():
    panel = load_csv("panel_dataset_cleaned.csv")
    metrics = load_csv("metrics_summary.csv")
    forecast_10y = load_csv("yield_forecast_10_years.csv")
    scenarios = load_csv("yield_forecast_scenarios.csv")
    backtest = load_csv("backtest_forecasts.csv")
    gb_importance = load_csv("gb_feature_importance.csv")
    return panel, metrics, forecast_10y, scenarios, backtest, gb_importance


def build_f1_df(metrics: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 for baseline, aerosols, met, combined."""
    base = metrics.loc[metrics["model"] == "Baseline"].iloc[0]
    aero = metrics.loc[metrics["model"] == "Aerosol-only"].iloc[0]
    met = metrics.loc[metrics["model"] == "Meteorology-only"].iloc[0]
    comb = metrics.loc[metrics["model"] == "Combined_RF"].iloc[0]

    df = pd.DataFrame(
        {
            "model": ["Baseline", "Aerosols", "Meteorology", "Combined RF"],
            "macro_f1": [
                base["baseline_clf_macro_f1"],
                aero["macro_f1"],
                met["macro_f1"],
                comb["macro_f1"],
            ],
        }
    )
    return df


def build_rmse_df(metrics: pd.DataFrame) -> pd.DataFrame:
    """RMSE for baseline, Ridge, GB."""
    base = metrics.loc[metrics["model"] == "Baseline"].iloc[0]
    ridge = metrics.loc[metrics["model"] == "Ridge"].iloc[0]
    gb = metrics.loc[metrics["model"] == "GradientBoosting"].iloc[0]

    df = pd.DataFrame(
        {
            "model": ["Baseline", "Ridge", "GB"],
            "rmse": [
                base["baseline_reg_rmse"],
                ridge["rmse_mean"],
                gb["rmse_mean"],
            ],
        }
    )
    return df


def make_f1_chart(f1_df: pd.DataFrame):
    chart = (
        alt.Chart(f1_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("macro_f1:Q", title="Macro F1 (higher is better)"),
            tooltip=["model", alt.Tooltip("macro_f1:Q", format=".3f")],
        )
        .properties(width=400, height=300)
    )

    labels = chart.mark_text(
        dy=-8,
        fontSize=12,
    ).encode(text=alt.Text("macro_f1:Q", format=".3f"))

    return chart + labels


def make_rmse_chart(rmse_df: pd.DataFrame):
    chart = (
        alt.Chart(rmse_df)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("rmse:Q", title="RMSE (lower is better)"),
            tooltip=["model", alt.Tooltip("rmse:Q", format=".3f")],
        )
        .properties(width=400, height=300)
    )

    labels = chart.mark_text(
        dy=-8,
        fontSize=12,
    ).encode(text=alt.Text("rmse:Q", format=".3f"))

    return chart + labels


def make_scenario_chart(scenarios: pd.DataFrame, region: str, crop: str, scenario_list):
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
        st.warning("No data for this combination of region / crop / scenario.")
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
        .properties(width=700, height=350)
    )
    st.altair_chart(chart, width="stretch")


def make_backtest_chart(backtest: pd.DataFrame, region: str, crop: str):
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
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(
                    domain=["yield_true", "yield_pred"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
            tooltip=["year", "series", alt.Tooltip("yield:Q", format=".3f")],
        )
        .properties(width=700, height=350)
    )
    st.altair_chart(chart, width="stretch")


def make_importance_chart(gb_importance: pd.DataFrame, top_k: int = 10):
    df = gb_importance.sort_values("importance_mean", ascending=False).head(top_k)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("importance_mean:Q", title="Permutation importance (ΔR²)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("importance_mean:Q", format=".3f"),
                alt.Tooltip("importance_std:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )

    st.subheader("Global feature importance (GB regressor)")
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Permutation importance of the Gradient Boosting yield model. "
        "Higher values = larger average drop in R² when that feature is shuffled."
    )


def main():
    panel, metrics, forecast_10y, scenarios, backtest, gb_importance = load_all_data()

    st.title("Crop Yield, Aerosols & Forecasting – Interactive Dashboard")

    regions = sorted(panel["region"].unique())
    crops = sorted(panel["crop"].unique())

    st.sidebar.header("Global filters")
    selected_region = st.sidebar.selectbox("Region", regions)
    selected_crop = st.sidebar.selectbox("Crop", crops)

    overview_tab, model_tab, scenario_tab, backtest_tab, importance_tab = st.tabs(
        [
            "Overview",
            "Model comparison",
            "Scenario explorer",
            "Backtest diagnostics",
            "Feature importance",
        ]
    )

    # -------------------------
    # Overview
    # -------------------------
    with overview_tab:
        st.subheader("Dataset overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows in panel dataset", f"{len(panel):,}")
        with col2:
            st.metric("Regions", f"{panel['region'].nunique()}")
        with col3:
            st.metric("Crops", f"{panel['crop'].nunique()}")

        st.markdown(
            """
This dashboard reads the outputs from your Python pipeline:

- panel_dataset_cleaned.csv  
- metrics_summary.csv  
- yield_forecast_10_years.csv  
- yield_forecast_scenarios.csv  
- backtest_forecasts.csv  
- gb_feature_importance.csv  
            """
        )

        st.markdown("Sample of cleaned panel dataset:")
        st.dataframe(panel.head(20))

    # -------------------------
    # Model comparison
    # -------------------------
    with model_tab:
        st.subheader("Model comparison (classification & regression)")

        f1_df = build_f1_df(metrics)
        rmse_df = build_rmse_df(metrics)

        st.markdown("Macro-F1 scores (yield class prediction):")
        st.altair_chart(make_f1_chart(f1_df), width="stretch")

        st.markdown("RMSE for continuous yield prediction:")
        st.altair_chart(make_rmse_chart(rmse_df), width="stretch")

        st.markdown(
            """
Interpretation:

- Classification: combined aerosol + meteorology model clearly beats the majority-class baseline,
  and also improves over aerosols-only and met-only models.
- Regression: Gradient Boosting significantly reduces RMSE compared to the mean-yield baseline
  and the linear Ridge model.
            """
        )

    # -------------------------
    # Scenario explorer
    # -------------------------
    with scenario_tab:
        st.subheader("Scenario explorer – clean air vs polluted futures")

        scenario_options = sorted(scenarios["scenario"].unique())
        default_scenarios = [
            s for s in scenario_options if s in ["baseline", "clean_air", "polluted"]
        ]
        if not default_scenarios:
            default_scenarios = scenario_options

        selected_scenarios = st.multiselect(
            "Scenario(s) to show",
            options=scenario_options,
            default=default_scenarios,
        )

        make_scenario_chart(
            scenarios=scenarios,
            region=selected_region,
            crop=selected_crop,
            scenario_list=selected_scenarios,
        )

        st.markdown(
            """
Each line shows the 10-year forecasted yield under a different aerosol scenario
for the chosen region and crop.

- clean_air: AOD and PM2.5 reduced by 20%  
- polluted: AOD and PM2.5 increased by 20%  
- baseline: no perturbation to aerosol levels
            """
        )

    # -------------------------
    # Backtest diagnostics
    # -------------------------
    with backtest_tab:
        st.subheader("Backtest – true vs predicted yields")

        gb_back = metrics.loc[metrics["model"] == "GB_backtest"].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Backtest RMSE (GB)", f"{gb_back['rmse']:.3f}")
        with col2:
            st.metric("Backtest R² (GB)", f"{gb_back['r2']:.3f}")

        st.markdown(
            "Per-region & crop backtest curve (train on early years, test on later years):"
        )
        make_backtest_chart(backtest, selected_region, selected_crop)

        st.markdown(
            """
The backtest R² close to 0.8 shows that the Gradient Boosting model generalizes
reasonably well to unseen years, not just fitting noise in the training period.
            """
        )

    # -------------------------
    # Feature importance tab
    # -------------------------
    with importance_tab:
        make_importance_chart(gb_importance)


if __name__ == "__main__":
    main()
