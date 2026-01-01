# data_loading.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict
import logging
import re
import unicodedata

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _norm_text(x) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKC", s)     # fixes weird unicode variants
    s = s.replace("\n", " ").replace("\t", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)              # collapse multiple spaces
    return s


def _canonical_col(name) -> str:
    """
    Canonicalize column header strings so near-duplicates merge.
    Keeps meaning, removes spacing noise, and normalizes unicode.
    """
    s = _norm_text(name)
    if s == "":
        return s

    # Normalize common micro symbol variants to "u"
    s = s.replace("μ", "u").replace("µ", "u")

    # Lowercase everything for stable matching
    s_low = s.lower()

    # Some optional targeted fixes for common columns in your dataset
    # (Add more here if you see duplicates in outputs)
    replacements = {
        "aod ": "aod",
        " aod": "aod",
        "pm2.5  ugm-3": "pm2.5 ugm-3",
        "pm2.5 (ugm-3)": "pm2.5 ugm-3",
        "pm2.5 ugm-3 ": "pm2.5 ugm-3",
        "ch4 ": "ch4",
        "co ": "co",
        "radiation": "radiation",
    }

    s_low = replacements.get(s_low, s_low)
    s_low = s_low.strip()
    s_low = re.sub(r"\s+", " ", s_low)

    return s_low


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If normalization causes duplicate column names, merge them by taking
    first non-null value row-wise.
    """
    if not df.columns.duplicated().any():
        return df

    cols = list(df.columns)
    groups: Dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(c, []).append(c)

    # If duplicates exist, pandas will show the same key multiple times.
    # We rebuild a new df with merged columns.
    new_cols = []
    out = pd.DataFrame(index=df.index)

    # We must refer to columns by position since names collide.
    # So we iterate with enumerate.
    for col_name in pd.unique(df.columns):
        idxs = [i for i, c in enumerate(df.columns) if c == col_name]
        if len(idxs) == 1:
            out[col_name] = df.iloc[:, idxs[0]]
        else:
            block = df.iloc[:, idxs]
            # take first non-null across duplicates
            merged = block.bfill(axis=1).iloc[:, 0]
            out[col_name] = merged
        new_cols.append(col_name)

    return out


def find_header_row(df_raw: pd.DataFrame, max_scan: int = 30) -> int:
    """
    Robustly find the real header row in sheets loaded with header=None.

    Heuristics:
      - Prefer rows with many non-empty string labels (not mostly numeric).
      - Must contain a time/year token AND a yield token.
      - Scans only first `max_scan` rows for speed.
    """

    def _is_blank(x) -> bool:
        if x is None:
            return True
        try:
            if pd.isna(x):
                return True
        except Exception:
            pass
        s = str(x).strip()
        return (s == "") or (s.lower() in {"nan", "none", "null"})

    def _norm_cell(x) -> str:
        if _is_blank(x):
            return ""
        s = str(x)
        s = re.sub(r"[\u00a0\t\r\n]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    max_scan = min(max_scan, len(df_raw))
    best_row = 0
    best_score = -1e9

    for r in range(max_scan):
        row = df_raw.iloc[r].tolist()
        tokens = [_norm_cell(v) for v in row]
        non_empty = [t for t in tokens if t]
        if not non_empty:
            continue

        n_str, n_num = 0, 0
        for v in row:
            if _is_blank(v):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                n_num += 1
            else:
                s = _norm_cell(v)
                if re.fullmatch(r"-?\d+(\.\d+)?", s):
                    n_num += 1
                else:
                    n_str += 1

        has_time = any(t.startswith("time") or t.startswith("year") or t == "years" for t in non_empty)
        has_yield = any("yield" in t for t in non_empty)

        frac_num = n_num / max(1, (n_num + n_str))

        score = 0.0
        score += 4.0 * has_time + 5.0 * has_yield
        score += 0.20 * len(non_empty) + 0.30 * n_str
        score -= 3.5 * frac_num

        common = ["aod", "pm", "precip", "temp", "humidity", "wind", "ch4", "no2", "o3", "so2", "co"]
        score += 0.4 * sum(any(c in t for t in non_empty) for c in common)

        if has_time and has_yield and score > best_score:
            best_score = score
            best_row = r

    return int(best_row)



def load_and_clean_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Load one sheet and return a clean dataframe with:
      - canonical 'year' column
      - canonical 'yield' column
      - numeric features coerced to float where possible
      - multi-row header merged (parent + subheader + units)
      - precipitation unit extracted from helper 'years' column
      - year-like junk columns dropped
    """
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    header_row = find_header_row(df_raw)
    unit_row = header_row - 1 if header_row - 1 >= 0 else None
    parent_row = header_row - 2 if header_row - 2 >= 0 else None

    def _is_blank(x) -> bool:
        if x is None:
            return True
        try:
            if pd.isna(x):
                return True
        except Exception:
            pass
        s = str(x).strip()
        return (s == "") or (s.lower() in {"nan", "none", "null"})

    def _norm_cell(x) -> str:
        if _is_blank(x):
            return ""
        s = str(x)
        s = re.sub(r"[\u00a0\t\r\n]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    # Forward-fill parent labels across blocks (fixes met blocks like Surface Skin Temp + DAY/NIGHT/AVG)
    parent_vals = df_raw.iloc[parent_row].copy() if parent_row is not None else None
    if parent_vals is not None:
        parent_vals = parent_vals.ffill()

    unit_vals = df_raw.iloc[unit_row].copy() if unit_row is not None else None

    headers = []
    for c in range(df_raw.shape[1]):
        main = df_raw.iat[header_row, c]
        unit = unit_vals.iat[c] if unit_vals is not None else None
        parent = parent_vals.iat[c] if parent_vals is not None else None

        main_s = _norm_cell(main)
        unit_s = _norm_cell(unit)
        parent_s = _norm_cell(parent)

        if main_s in {"day", "night", "avg", "minimum", "maximum"} and parent_s:
            combined = f"{parent_s} {main_s}"
        elif main_s in {"years", "year"} and parent_s:
            combined = f"{parent_s} years"
        else:
            combined = main_s or parent_s

        # Attach units if meaningful
        if unit_s and unit_s not in {"for wheat", "for rice", "for maize", "for corn"}:
            if re.search(r"[a-zA-Z]", unit_s) and ("/" in unit_s or unit_s in {"ppbv", "ugm-3", "μgm-3", "mm/day", "mm"}):
                if unit_s not in combined:
                    combined = (combined + f" ({unit_s})").strip()

        col = _canonical_col(combined)
        if (not col) or (col in {"nan", "none"}):
            col = f"col_{c}"
        headers.append(col)

    df = df_raw.iloc[header_row + 1:].copy()
    df.columns = headers

    df = df.dropna(axis=1, how="all")
    df = _dedupe_columns(df)

    # --- precipitation unit extraction from "* years (mm/day)" helper column ---
    cols = list(df.columns)
    rename = {}
    drop_cols = []
    for i, c in enumerate(cols):
        m = re.match(r"^(.*)\s+years\s*\(([^)]+)\)$", c)
        if not m:
            continue
        base = m.group(1).strip()
        unit = m.group(2).strip()
        if i + 1 < len(cols):
            nxt = cols[i + 1]
            if nxt.startswith(base) and f"({unit})" not in nxt:
                rename[nxt] = f"{nxt} ({unit})"
        drop_cols.append(c)

    if rename:
        df = df.rename(columns=rename)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- identify primary year column ---
    time_candidates = [c for c in df.columns if ("time" in c) or (c == "year") or (c == "years") or ("year" in c)]
    time_col = time_candidates[0] if time_candidates else None

    if time_col is None:
        best = None
        best_frac = -1
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            frac = s.between(1900, 2100).mean()
            if frac > best_frac and frac > 0.6:
                best_frac = frac
                best = c
        time_col = best

    if time_col is None:
        raise ValueError(f"Could not find a usable year/time column in sheet '{sheet_name}' (header_row={header_row})")

    df = df.rename(columns={time_col: "year"})

    # --- yield column ---
    yield_candidates = [c for c in df.columns if "yield" in c]
    if not yield_candidates:
        raise ValueError(f"Could not find a yield column in sheet '{sheet_name}'")
    df = df.rename(columns={yield_candidates[0]: "yield"})

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    # --- drop other year-like junk columns ---
    junk = []
    for c in df.columns:
        if c in {"year", "yield"}:
            continue
        name = _norm_cell(c)
        if name in {"year", "years", "time", "date"}:
            junk.append(c)
            continue
        if re.fullmatch(r"(19|20)\d{2}", name):
            junk.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.7 and s.between(1900, 2100).mean() > 0.9:
            junk.append(c)

    if junk:
        df = df.drop(columns=junk, errors="ignore")

    # drop rows missing target essentials
    df = df.dropna(subset=["year", "yield"], how="any")

    # coerce everything else numeric where possible
    for c in df.columns:
        if c == "year":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)



def parse_region_crop(sheet_name: str) -> Tuple[str, str]:
    """Split sheet name like 'VNS_WHEAT' → region='VNS', crop='wheat'."""
    parts = _norm_text(sheet_name).split("_")
    region = parts[0]
    crop = parts[1].lower() if len(parts) > 1 else "unknown"
    return region, crop


def build_panel_dataset(path: Path) -> pd.DataFrame:
    """Load all sheets and stack into a single panel dataframe."""
    logger.info("Loading Excel file: %s", path)
    xls = pd.ExcelFile(path)

    frames = []
    for sheet in xls.sheet_names:
        logger.info("Processing sheet: %s", sheet)
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
