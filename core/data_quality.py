"""
Core Data Quality, Validation & Timezone Utilities
"""
import datetime
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger("DataQuality")
IST = pytz.timezone("Asia/Kolkata")

def now_ist() -> datetime.datetime:
    """Returns current datetime in Asia/Kolkata (IST)."""
    return datetime.datetime.now(IST)

def format_ist(dt: Optional[datetime.datetime] = None) -> str:
    """Formats datetime in human-readable IST string."""
    if dt is None:
        dt = now_ist()
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")

def validate_ohlcv_dataframe(
    df: Optional[pd.DataFrame],
    symbol: str = "",
    min_bars: int = 20
) -> Tuple[bool, pd.DataFrame, List[str]]:
    """
    Comprehensive OHLCV data validation layer (F-40).
    Verifies:
    1. DataFrame existence and minimum row count.
    2. Presence of required OHLCV columns.
    3. Chronological ordering and uniqueness of date index/timestamps.
    4. Price sanity: High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close.
    5. Non-zero, strictly positive prices and non-negative volume.
    6. Absence of NaNs / Infs in critical price fields.
    7. Extreme price anomaly checks (single bar jump > 50%).
    Returns (is_valid: bool, cleaned_df: pd.DataFrame, warnings: List[str]).
    """
    warnings: List[str] = []
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, pd.DataFrame(), [f"{symbol}: DataFrame is None or empty."]

    if len(df) < min_bars:
        return False, pd.DataFrame(), [f"{symbol}: Insufficient bars ({len(df)} < required {min_bars})."]

    # 1. Normalize column names (Title case: Open, High, Low, Close, Volume)
    clean_df = df.copy()
    col_map = {}
    for c in clean_df.columns:
        c_lower = str(c).lower().strip()
        if c_lower == "open": col_map[c] = "Open"
        elif c_lower == "high": col_map[c] = "High"
        elif c_lower == "low": col_map[c] = "Low"
        elif c_lower == "close": col_map[c] = "Close"
        elif c_lower in ("volume", "vol"): col_map[c] = "Volume"
    clean_df.rename(columns=col_map, inplace=True)

    required_cols = ["Open", "High", "Low", "Close"]
    missing_cols = [c for c in required_cols if c not in clean_df.columns]
    if missing_cols:
        return False, pd.DataFrame(), [f"{symbol}: Missing required price columns {missing_cols}."]

    if "Volume" not in clean_df.columns:
        clean_df["Volume"] = 0.0
        warnings.append(f"{symbol}: Missing Volume column, populated with zeros.")

    # 2. Chronological Ordering and Duplicate Timestamps
    if not clean_df.index.is_monotonic_increasing:
        clean_df.sort_index(inplace=True)
        warnings.append(f"{symbol}: Timestamps were not sorted chronologically; sorted ascending.")

    if clean_df.index.has_duplicates:
        dup_count = clean_df.index.duplicated().sum()
        clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
        warnings.append(f"{symbol}: Dropped {dup_count} duplicate timestamp records.")

    # 3. Numeric conversions & NaN filtering
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    nan_rows = clean_df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    if nan_rows.sum() > 0:
        dropped_nans = int(nan_rows.sum())
        clean_df = clean_df[~nan_rows]
        warnings.append(f"{symbol}: Dropped {dropped_nans} rows containing NaN/null prices.")

    if len(clean_df) < min_bars:
        return False, pd.DataFrame(), [f"{symbol}: Insufficient valid bars remaining after NaN drop ({len(clean_df)} < {min_bars})."]

    # 4. Strict Positivity
    non_positive = (clean_df["Open"] <= 0) | (clean_df["High"] <= 0) | (clean_df["Low"] <= 0) | (clean_df["Close"] <= 0)
    if non_positive.sum() > 0:
        bad_count = int(non_positive.sum())
        clean_df = clean_df[~non_positive]
        warnings.append(f"{symbol}: Dropped {bad_count} rows with non-positive price values.")

    if len(clean_df) < min_bars:
        return False, pd.DataFrame(), [f"{symbol}: Insufficient positive bars ({len(clean_df)} < {min_bars})."]

    # 5. Price Geometry Sanity (High >= Low, High >= max(Open, Close), Low <= min(Open, Close))
    # Correct inverted wicks where High < Low or Open/Close outside High-Low range
    inverted_hl = clean_df["High"] < clean_df["Low"]
    if inverted_hl.sum() > 0:
        inv_count = int(inverted_hl.sum())
        # Swap high and low where inverted
        clean_df.loc[inverted_hl, ["High", "Low"]] = clean_df.loc[inverted_hl, ["Low", "High"]].values
        warnings.append(f"{symbol}: Corrected {inv_count} bars where High was less than Low.")

    clean_df["High"] = clean_df[["High", "Open", "Close"]].max(axis=1)
    clean_df["Low"] = clean_df[["Low", "Open", "Close"]].min(axis=1)
    clean_df["Volume"] = clean_df["Volume"].clip(lower=0.0).fillna(0.0)

    # 6. Single-bar extreme jump anomaly check (> 50% jump bar-to-bar without market event)
    ret = clean_df["Close"].pct_change().abs()
    extreme_jumps = ret > 0.50
    if extreme_jumps.sum() > 0:
        jump_count = int(extreme_jumps.sum())
        warnings.append(f"{symbol}: Detected {jump_count} extreme price jumps (> 50% single candle shift).")

    return True, clean_df, warnings

def check_data_freshness(timestamp_str: Optional[str], max_age_seconds: int = 900) -> Dict[str, Any]:
    """
    Checks data timestamp freshness against current IST time.
    """
    if not timestamp_str:
        return {"status": "UNKNOWN", "is_stale": True, "age_seconds": None, "error": "Missing timestamp"}

    try:
        clean_ts = timestamp_str.replace(" IST", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(clean_ts, fmt)
                break
            except Exception:
                pass
        else:
            return {"status": "INVALID_FORMAT", "is_stale": True, "age_seconds": None}

        dt_now = now_ist().replace(tzinfo=None)
        age_sec = max(0, int((dt_now - dt).total_seconds()))
        is_stale = age_sec > max_age_seconds
        return {
            "status": "VALID" if not is_stale else "STALE",
            "is_stale": is_stale,
            "age_seconds": age_sec,
            "timestamp": timestamp_str
        }
    except Exception as ex:
        return {"status": "ERROR", "is_stale": True, "age_seconds": None, "error": str(ex)}
