Crop Yield Project (Capstone-IP)

End-to-end data science project to model how aerosol pollution and meteorological factors affect crop yields (wheat, rice, maize) across multiple regions in India. The pipeline builds a cleaned panel dataset from Excel, trains classification and regression models, generates 10-year scenario forecasts, and serves an interactive Streamlit dashboard and a Power BI report.

Project structure

* compiled data FINAL (3).xlsx – Raw multi-sheet Excel data (regions × crops).
* data_loading.py – Reads all sheets and builds a unified panel dataset.
* features.py – Feature engineering (yield classes, lags, aerosol/met variables).
* modeling.py – Baselines, logistic classifiers, Random Forest classifier, Ridge and Gradient Boosting regressors, permutation feature importance.
* forecasting.py – 10-year yield forecasts and clean_air / polluted scenario generation, plus backtesting.
* run_pipeline.py – Orchestrates the full pipeline and writes CSV outputs to outputs/.
* dashboard_app.py – Streamlit dashboard for model comparison, scenario explorer, backtest diagnostics, and feature importance.
* outputs/ – Generated CSVs (cleaned panel, metrics summary, forecasts, backtest, feature importances).
* CropYield_Aerosols_Report.pbix – Power BI report using the forecast and backtest CSVs.
* requirements.txt – Python dependencies.

Quickstart

1. Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate   (Windows)

2. Install dependencies:

pip install -r requirements.txt

3. Run the full pipeline:

python run_pipeline.py

This will create/update CSVs in the outputs/ folder:

* panel_dataset_cleaned.csv
* metrics_summary.csv
* yield_forecast_10_years.csv
* yield_forecast_scenarios.csv
* backtest_forecasts.csv
* gb_feature_importance.csv

4. Launch the Streamlit dashboard:

python -m streamlit run dashboard_app.py

The app provides:

* Model comparison (macro-F1 and RMSE vs baselines)
* Scenario explorer (baseline vs clean_air vs polluted yields)
* Backtest curves (true vs predicted yield over time)
* Global feature importance (permutation importance of the GB regressor)

Power BI report

The file CropYield_Aerosols_Report.pbix connects directly to:

* outputs/yield_forecast_scenarios.csv
* outputs/backtest_forecasts.csv

It includes:

* A bar chart of average yield gain from clean_air vs polluted scenarios by region.
* A line chart of true vs predicted yields from the backtest, with the GB backtest R² highlighted.

Together, the Python pipeline, Streamlit app, and Power BI report demonstrate the full workflow from data engineering and modeling to forecasting, scenario analysis, and interactive visualization.
