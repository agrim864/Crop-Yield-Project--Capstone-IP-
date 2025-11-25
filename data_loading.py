from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def find_header_row(df_raw: pd.DataFrame) -> int:
    """Find row index that contains 'time' or 'year' → treat as header row."""
    header_keywords = ["time", "year"]
    for idx, row in df_raw.iterrows():
        vals = [str(x).strip().lower() for x in row.values]
        if any(v.startswith(k) for v in vals for k in header_keywords):
            return idx
    return 0


def load_and_clean_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet, fix headers, pick year & yield, drop junk columns."""
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = find_header_row(df_raw)
    header = df_raw.iloc[header_row].tolist()

    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = header

    mask_notna = pd.notna(df.columns)
    df = df.loc[:, mask_notna]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.dropna(axis=1, how="all")

    year_col = None
    for col in df.columns:
        if isinstance(col, str) and ("time" in col.lower() or "year" in col.lower()):
            year_col = col
            break
    if year_col is None:
        year_col = df.columns[0]

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df["year"] = df[year_col].astype(int)

    yield_col = None
    for col in df.columns:
        if isinstance(col, str) and "yield" in col.lower():
            yield_col = col
            break
    if yield_col is None:
        raise ValueError(f"No yield column found in sheet {sheet_name}")

    df["yield"] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=["yield"])

    cols_to_drop = []
    for c in df.columns:
        if c in ["year", "yield"]:
            continue
        if isinstance(c, str):
            cl = c.strip().lower()
            if (
                cl in ["time", "year", "years"]
                or cl.startswith("unnamed")
                or "for wheat" in cl
            ):
                cols_to_drop.append(c)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    df["sheet"] = sheet_name
    return df


def parse_region_crop(sheet_name: str) -> Tuple[str, str]:
    """Split sheet name like 'VNS_WHEAT' → region='VNS', crop='wheat'."""
    parts = sheet_name.split("_")
    region = parts[0]
    crop = parts[1].lower() if len(parts) > 1 else "unknown"
    return region, crop


def build_panel_dataset(path: Path) -> pd.DataFrame:
    """Load all sheets and stack into a single panel dataframe."""
    logger.info(f"Loading Excel file: {path}")
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        logger.info(f"Processing sheet: {sheet}")
        df = load_and_clean_sheet(path, sheet)
        region, crop = parse_region_crop(sheet)
        df["region"] = region
        df["crop"] = crop
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["region", "crop", "year"]).reset_index(drop=True)

    logger.info(
        "Built panel dataset with %d rows, %d columns, %d regions, %d crops",
        len(panel),
        panel.shape[1],
        panel["region"].nunique(),
        panel["crop"].nunique(),
    )
    return panel
