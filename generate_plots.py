# save as: generate_plots.py
# run: python generate_plots.py
# expects CSVs in: outputs/
# saves PNGs to: outputs/plots/

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def load_csv(filename: str) -> pd.DataFrame:
    p = OUT / filename
    if not p.exists():
        print(f"missing: {p}")
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"failed reading {p}: {e}")
        return pd.DataFrame()

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def plot_validation_split(panel: pd.DataFrame):
    if panel.empty or "year" not in panel.columns:
        return
    years = sorted(to_num(panel["year"]).dropna().unique())
    if len(years) < 3:
        return

    n_splits = min(4, len(years) - 1)
    test_years = years[-n_splits:]
    mat = np.array([[1 if y == ty else 0 for y in years] for ty in test_years])

    plt.figure(figsize=(10, 2.8))
    plt.imshow(mat, aspect="auto")
    plt.yticks(range(n_splits), [f"Fold {i+1} (test={int(ty)})" for i, ty in enumerate(test_years)])
    plt.xticks(range(len(years)), [int(y) for y in years], rotation=45)
    plt.title("Leak-safe validation (train on past, test on future)")
    plt.xlabel("Year")
    plt.ylabel("Fold")
    savefig(PLOTS / "01_validation_split.png")

def plot_backtest_true_vs_pred(backtest: pd.DataFrame):
    if backtest.empty or not {"year", "yield_true", "yield_pred"}.issubset(backtest.columns):
        return

    df = backtest.copy()
    df["year"] = to_num(df["year"])
    df["yield_true"] = to_num(df["yield_true"])
    df["yield_pred"] = to_num(df["yield_pred"])
    df = df.dropna(subset=["year", "yield_true", "yield_pred"]).sort_values("year")
    if df.empty:
        return

    # if region/crop exists, plot the largest group for a cleaner time-series
    if {"region", "crop"}.issubset(df.columns):
        df["region"] = df["region"].astype(str)
        df["crop"] = df["crop"].astype(str)
        grp = df.groupby(["region", "crop"]).size().sort_values(ascending=False)
        if len(grp) > 0:
            r, c = grp.index[0]
            df = df[(df["region"] == r) & (df["crop"] == c)]
            title_suffix = f" (region={r}, crop={c})"
        else:
            title_suffix = ""
    else:
        title_suffix = ""

    plt.figure(figsize=(10, 4))
    plt.plot(df["year"], df["yield_true"], marker="o", label="True")
    plt.plot(df["year"], df["yield_pred"], marker="o", label="Predicted")
    plt.title("Backtest: true vs predicted" + title_suffix)
    plt.xlabel("Year")
    plt.ylabel("Yield")
    plt.legend()
    savefig(PLOTS / "02_backtest_true_vs_pred.png")

def plot_true_vs_pred_scatter(backtest: pd.DataFrame):
    if backtest.empty or not {"yield_true", "yield_pred"}.issubset(backtest.columns):
        return

    yt = to_num(backtest["yield_true"])
    yp = to_num(backtest["yield_pred"])
    ok = yt.notna() & yp.notna()
    yt, yp = yt[ok], yp[ok]
    if len(yt) < 3:
        return

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp)
    plt.plot([lo, hi], [lo, hi])
    plt.title("Backtest: true vs predicted (scatter)")
    plt.xlabel("True yield")
    plt.ylabel("Predicted yield")
    savefig(PLOTS / "03_true_vs_pred_scatter.png")

def plot_residual_hist(backtest: pd.DataFrame):
    if backtest.empty or not {"yield_true", "yield_pred"}.issubset(backtest.columns):
        return
    resid = (to_num(backtest["yield_pred"]) - to_num(backtest["yield_true"])).dropna()
    if resid.empty:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=25)
    plt.title("Residuals (pred - true)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    savefig(PLOTS / "04_residual_hist.png")

def plot_temporal_learning(bt_yearly: pd.DataFrame):
    if bt_yearly.empty or "year" not in bt_yearly.columns:
        return

    if "rmse" in bt_yearly.columns:
        df = bt_yearly.copy()
        df["year"] = to_num(df["year"])
        df["rmse"] = to_num(df["rmse"])
        df = df.dropna(subset=["year", "rmse"]).sort_values("year")
        if not df.empty:
            plt.figure(figsize=(9, 4))
            plt.plot(df["year"], df["rmse"], marker="o")
            plt.title("Temporal learning: RMSE by test year")
            plt.xlabel("Year")
            plt.ylabel("RMSE")
            savefig(PLOTS / "05_rmse_by_year.png")

    if "accuracy_within_10pct" in bt_yearly.columns:
        df = bt_yearly.copy()
        df["year"] = to_num(df["year"])
        df["accuracy_within_10pct"] = to_num(df["accuracy_within_10pct"])
        df = df.dropna(subset=["year", "accuracy_within_10pct"]).sort_values("year")
        if not df.empty:
            plt.figure(figsize=(9, 4))
            plt.plot(df["year"], df["accuracy_within_10pct"], marker="o")
            plt.title("Temporal learning: accuracy within 10% by test year")
            plt.xlabel("Year")
            plt.ylabel("Accuracy within 10%")
            savefig(PLOTS / "06_accuracy_within_10pct_by_year.png")

def plot_error_heatmap(backtest: pd.DataFrame):
    if backtest.empty or not {"region", "crop", "yield_true", "yield_pred"}.issubset(backtest.columns):
        return

    df = backtest.copy()
    yt = to_num(df["yield_true"])
    yp = to_num(df["yield_pred"])
    denom = yt.abs().where(yt.abs() > 1e-9, 1.0)
    df["ape"] = (yp - yt).abs() / denom
    df = df.dropna(subset=["ape", "region", "crop"])
    if df.empty:
        return

    pivot = df.pivot_table(index="crop", columns="region", values="ape", aggfunc="mean")
    if pivot.empty:
        return

    plt.figure(figsize=(10, max(3, 0.5 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.title("Mean absolute percentage error by crop x region")
    plt.colorbar(label="APE")
    savefig(PLOTS / "07_error_heatmap_crop_region.png")

def plot_feature_importance(imp: pd.DataFrame, top_k: int = 15):
    if imp.empty or not {"feature", "importance_mean"}.issubset(imp.columns):
        return

    df = imp.copy()
    df["importance_mean"] = to_num(df["importance_mean"])
    df = df.dropna(subset=["importance_mean"]).sort_values("importance_mean", ascending=False).head(top_k)
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance_mean"][::-1])
    plt.title(f"Permutation feature importance (top {top_k})")
    plt.xlabel("Importance (mean drop)")
    plt.ylabel("Feature")
    savefig(PLOTS / "08_feature_importance.png")

def plot_scenarios(scen: pd.DataFrame):
    if scen.empty or not {"year", "yield_pred", "scenario"}.issubset(scen.columns):
        return

    df = scen.copy()
    df["year"] = to_num(df["year"])
    df["yield_pred"] = to_num(df["yield_pred"])
    df = df.dropna(subset=["year", "yield_pred", "scenario"])
    if df.empty:
        return

    # if region/crop exist, pick largest group to avoid clutter
    title_suffix = ""
    group_cols = [c for c in ["region", "crop"] if c in df.columns]
    if group_cols:
        grp = df.groupby(group_cols).size().sort_values(ascending=False)
        if len(grp) > 0:
            key = grp.index[0]
            if isinstance(key, tuple):
                for col, val in zip(group_cols, key):
                    df = df[df[col].astype(str) == str(val)]
                title_suffix = " (" + ", ".join([f"{c}={v}" for c, v in zip(group_cols, key)]) + ")"
            else:
                col = group_cols[0]
                df = df[df[col].astype(str) == str(key)]
                title_suffix = f" ({col}={key})"

    plt.figure(figsize=(10, 4))
    for s, g in df.groupby("scenario"):
        g = g.sort_values("year")
        plt.plot(g["year"], g["yield_pred"], marker="o", label=str(s))
    plt.title("Scenario forecast" + title_suffix)
    plt.xlabel("Year")
    plt.ylabel("Predicted yield")
    plt.legend()
    savefig(PLOTS / "09_scenario_forecasts.png")

def main():
    panel = load_csv("panel_dataset_cleaned.csv")
    backtest = load_csv("backtest_forecasts.csv")
    bt_yearly = load_csv("backtest_metrics_by_year.csv")
    imp = load_csv("gb_feature_importance.csv")
    scen = load_csv("yield_forecast_scenarios.csv")

    plot_validation_split(panel)
    plot_backtest_true_vs_pred(backtest)
    plot_true_vs_pred_scatter(backtest)
    plot_residual_hist(backtest)
    plot_temporal_learning(bt_yearly)
    plot_error_heatmap(backtest)
    plot_feature_importance(imp)
    plot_scenarios(scen)

    print(f"done. plots saved to: {PLOTS}")

if __name__ == "__main__":
    main()
