import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "outputs"

# Load data
metrics = pd.read_csv(OUT_DIR / "metrics_summary.csv")
gb_imp = pd.read_csv(OUT_DIR / "gb_feature_importance.csv")
backtest = pd.read_csv(OUT_DIR / "backtest_forecasts.csv")
scenarios = pd.read_csv(OUT_DIR / "yield_forecast_scenarios.csv")

# 1) Macro-F1 bar chart
base = metrics[metrics["model"] == "Baseline"].iloc[0]
aero = metrics[metrics["model"] == "Aerosol-only"].iloc[0]
met = metrics[metrics["model"] == "Meteorology-only"].iloc[0]
comb = metrics[metrics["model"] == "Combined_RF"].iloc[0]

f1_models = ["Baseline", "Aerosols", "Meteorology", "Combined RF"]
f1_values = [
    base["baseline_clf_macro_f1"],
    aero["macro_f1"],
    met["macro_f1"],
    comb["macro_f1"],
]

plt.figure()
plt.bar(f1_models, f1_values)
plt.ylabel("Macro-F1")
plt.title("Macro-F1 comparison (yield class)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_f1_bar.png")
plt.close()

# 2) RMSE bar chart
ridge = metrics[metrics["model"] == "Ridge"].iloc[0]
gb = metrics[metrics["model"] == "GradientBoosting"].iloc[0]

rmse_models = ["Baseline", "Ridge", "GB"]
rmse_values = [
    base["baseline_reg_rmse"],
    ridge["rmse_mean"],
    gb["rmse_mean"],
]

plt.figure()
plt.bar(rmse_models, rmse_values)
plt.ylabel("RMSE")
plt.title("RMSE comparison (continuous yield)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_rmse_bar.png")
plt.close()

# 3) GB permutation importance (top 10)
top10 = gb_imp.sort_values("importance_mean", ascending=False).head(10)

plt.figure()
plt.barh(top10["feature"], top10["importance_mean"])
plt.xlabel("Permutation importance (ΔR²)")
plt.title("Top 10 features – GB permutation importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_gb_importance_top10.png")
plt.close()

# 4) Backtest: true vs predicted (mean over regions/crops)
bt_year = (
    backtest.groupby("year")[["yield_true", "yield_pred"]]
    .mean()
    .reset_index()
)

plt.figure()
plt.plot(bt_year["year"], bt_year["yield_true"], marker="o", label="True")
plt.plot(bt_year["year"], bt_year["yield_pred"], marker="o", label="Predicted")
plt.xlabel("Year")
plt.ylabel("Yield")
plt.title("Backtest: true vs predicted yield (mean over regions/crops)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_backtest_true_vs_pred.png")
plt.close()

# 5) Scenario example: first region/crop combo
first_region = scenarios["region"].iloc[0]
first_crop = scenarios["crop"].iloc[0]

sc_ex = scenarios[
    (scenarios["region"] == first_region) &
    (scenarios["crop"] == first_crop)
].copy()

plt.figure()
for scen, df_s in sc_ex.groupby("scenario"):
    plt.plot(df_s["year"], df_s["yield_pred"], marker="o", label=scen)

plt.xlabel("Year")
plt.ylabel("Predicted yield")
plt.title(f"Scenario forecast – {first_region}, {first_crop}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_scenario_example.png")
plt.close()
