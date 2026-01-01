# report_generation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _fmt(x: Any, digits: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _df_tail_text(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or df.empty:
        return "(not available)"
    show = df.tail(n).copy()
    return show.to_string(index=False)


def write_summary(outputs_dir: Path) -> None:
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel_path = outputs_dir / "panel_dataset_cleaned.csv"
    metrics_path = outputs_dir / "metrics_summary.csv"

    bt_by_year_path = outputs_dir / "backtest_metrics_by_year.csv"
    timing_path = outputs_dir / "timing_summary.csv"

    panel = pd.read_csv(panel_path) if panel_path.exists() else pd.DataFrame()
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    bt_by_year = pd.read_csv(bt_by_year_path) if bt_by_year_path.exists() else pd.DataFrame()
    timing = pd.read_csv(timing_path) if timing_path.exists() else pd.DataFrame()

    clf_info = _load_json(outputs_dir / "models" / "best_classifier_info.json")
    reg_info = _load_json(outputs_dir / "models" / "best_regressor_info.json")

    def _row(model_name: str) -> Dict[str, Any]:
        if metrics.empty or "model" not in metrics.columns:
            return {}
        r = metrics.loc[metrics["model"] == model_name]
        return {} if r.empty else r.iloc[0].to_dict()

    base = _row("Baseline")
    clf = _row("BestClassifier")
    reg = _row("BestRegressor")
    bt = _row("Backtest")

    lines: list[str] = []
    lines.append("Crop Yield, Aerosols & Forecasting – Summary")
    lines.append("")

    if not panel.empty:
        lines.append("Dataset:")
        lines.append(f"- Rows: {len(panel):,}")
        lines.append(f"- Regions: {panel.get('region', pd.Series(dtype=object)).nunique()}")
        lines.append(f"- Crops: {panel.get('crop', pd.Series(dtype=object)).nunique()}")
        yr_min = pd.to_numeric(panel.get("year"), errors="coerce").min()
        yr_max = pd.to_numeric(panel.get("year"), errors="coerce").max()
        if pd.notna(yr_min) and pd.notna(yr_max):
            lines.append(f"- Years: {int(yr_min)} → {int(yr_max)}")
        lines.append("")
    else:
        lines.append("Dataset: (panel_dataset_cleaned.csv not found)")
        lines.append("")

    lines.append("Baselines:")
    lines.append(f"- Classifier macro-F1: {_fmt(base.get('baseline_clf_macro_f1'))}")
    lines.append(f"- Regressor RMSE: {_fmt(base.get('baseline_reg_rmse'))}")
    lines.append(f"- Regressor R²: {_fmt(base.get('baseline_reg_r2'))}")
    lines.append("")

    lines.append("BestClassifier:")
    lines.append(f"- Model: {clf.get('best_model_name', clf_info.get('best_model', 'N/A'))}")
    lines.append(f"- OOF macro-F1: {_fmt(clf.get('oof_f1_macro'))}")
    lines.append(f"- OOF accuracy: {_fmt(clf.get('oof_accuracy'))}")
    lines.append(f"- Train macro-F1: {_fmt(clf.get('train_f1_macro'))}")
    lines.append("")

    lines.append("BestRegressor:")
    lines.append(f"- Model: {reg.get('best_model_name', reg_info.get('best_model', 'N/A'))}")
    lines.append(f"- OOF RMSE: {_fmt(reg.get('oof_rmse'))}")
    lines.append(f"- OOF R²: {_fmt(reg.get('oof_r2'))}")
    lines.append("")

    lines.append("Backtest (walk-forward, leak-free):")
    lines.append(f"- RMSE: {_fmt(bt.get('rmse'))}")
    lines.append(f"- MAE: {_fmt(bt.get('mae'))}")
    lines.append(f"- MAPE (%): {_fmt(bt.get('mape'))}")
    lines.append(f"- R²: {_fmt(bt.get('r2'))}")
    lines.append(f"- Rows: {bt.get('n', 'N/A')}")
    lines.append("")

    lines.append("Time-based backtest metrics by year (last few years):")
    lines.append(_df_tail_text(bt_by_year, n=5))
    lines.append("")

    if not timing.empty and {"stage", "seconds"}.issubset(set(timing.columns)):
        total_row = timing.loc[timing["stage"] == "total"]
        if not total_row.empty:
            lines.append("Pipeline timing:")
            lines.append(f"- Total seconds: {_fmt(total_row.iloc[0]['seconds'], digits=2)}")
            lines.append("")
    else:
        lines.append("Pipeline timing: (timing_summary.csv not found)")
        lines.append("")

    lines.append("Outputs written:")
    lines.append(f"- {outputs_dir / 'panel_dataset_cleaned.csv'}")
    lines.append(f"- {outputs_dir / 'metrics_summary.csv'}")
    lines.append(f"- {outputs_dir / 'backtest_forecasts.csv'}")
    lines.append(f"- {outputs_dir / 'backtest_metrics_by_year.csv'}")
    lines.append(f"- {outputs_dir / 'yield_forecast_10_years.csv'}")
    lines.append(f"- {outputs_dir / 'yield_forecast_scenarios.csv'}")
    lines.append(f"- {outputs_dir / 'yield_forecast_past_only.csv'}")
    lines.append(f"- {outputs_dir / 'yield_forecast_scenarios_past_only.csv'}")
    lines.append(f"- {outputs_dir / 'permutation_importance.csv'}")
    lines.append(f"- {outputs_dir / 'gb_feature_importance.csv'}")
    lines.append(f"- {outputs_dir / 'timing_summary.csv'}")
    lines.append("")

    summary_txt = "\n".join(lines)
    (reports_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")

    html = "<html><head><meta charset='utf-8'><title>Crop Yield Report</title></head><body>"
    html += "<h1>Crop Yield, Aerosols & Forecasting – Report</h1>"
    html += "<pre style='white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;'>"
    html += summary_txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += "</pre></body></html>"
    (reports_dir / "report.html").write_text(html, encoding="utf-8")
