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
        return False, pd.DataFrame(), [f"{symbol}: Missing required Volume column; volume must not be fabricated as zero."]

    # 2. Chronological Ordering and Duplicate Timestamps
    if not clean_df.index.is_monotonic_increasing:
        clean_df.sort_index(inplace=True)
        warnings.append(f"{symbol}: Timestamps were not sorted chronologically; sorted ascending.")

    if clean_df.index.has_duplicates:
        dup_count = clean_df.index.duplicated().sum()
        clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
        warnings.append(f"{symbol}: Dropped {dup_count} duplicate timestamp records.")

    # 3. Numeric conversions & NaN/non-finite filtering
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    finite_rows = np.isfinite(clean_df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_rows.all():
        bad_count = int((~finite_rows).sum())
        clean_df = clean_df.loc[finite_rows].copy()
        warnings.append(f"{symbol}: Quarantined {bad_count} rows with non-finite OHLCV values.")

    nan_rows = clean_df[["Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1)
    if nan_rows.sum() > 0:
        dropped_nans = int(nan_rows.sum())
        clean_df = clean_df[~nan_rows]
        warnings.append(f"{symbol}: Dropped {dropped_nans} rows containing NaN/null prices or volume.")

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
    # Quarantine/drop corrupt bars where geometry is violated (H-01: do NOT silently coerce into false candles)
    inverted_hl = clean_df["High"] < clean_df["Low"]
    geom_violation = inverted_hl | \
                     (clean_df["High"] < clean_df["Open"]) | \
                     (clean_df["High"] < clean_df["Close"]) | \
                     (clean_df["Low"] > clean_df["Open"]) | \
                     (clean_df["Low"] > clean_df["Close"])

    if geom_violation.sum() > 0:
        inv_count = int(inverted_hl.sum())
        geom_count = int(geom_violation.sum())
        inv_ratio = geom_count / max(1, len(clean_df))
        if inv_ratio > 0.05:
            reason = f"High < Low ({inv_count} bars)" if inv_count > 0 else f"geometry violation ({geom_count} bars)"
            return False, pd.DataFrame(), [f"{symbol}: Unacceptable data corruption: {geom_count}/{len(clean_df)} bars ({inv_ratio*100:.1f}%) have {reason}."]
        clean_df = clean_df[~geom_violation]
        if inv_count > 0:
            warnings.append(f"{symbol}: Quarantined {geom_count} corrupted bars where High was less than Low or candle geometry was violated.")
        else:
            warnings.append(f"{symbol}: Quarantined {geom_count} corrupted bars with invalid candle geometry.")

    if len(clean_df) < min_bars:
        return False, pd.DataFrame(), [f"{symbol}: Insufficient valid bars remaining after corruption quarantine ({len(clean_df)} < {min_bars})."]

    # 5b. Volume non-negativity and finite validation (do NOT silently fill or clip corrupt volume)
    neg_vol = clean_df["Volume"] < 0
    if neg_vol.sum() > 0:
        bad_v_count = int(neg_vol.sum())
        clean_df = clean_df[~neg_vol]
        warnings.append(f"{symbol}: Quarantined {bad_v_count} rows with negative volume.")

    if len(clean_df) < min_bars:
        return False, pd.DataFrame(), [f"{symbol}: Insufficient valid bars remaining after volume quarantine ({len(clean_df)} < {min_bars})."]

    # 6. Single-bar extreme jump anomaly check (> 50% jump bar-to-bar without market event)
    ret = clean_df["Close"].pct_change().abs()
    extreme_jumps = ret > 0.50
    if extreme_jumps.sum() > 0:
        jump_count = int(extreme_jumps.sum())
        return False, pd.DataFrame(), [
            f"{symbol}: Rejected {jump_count} unexplained price jump(s) greater than 50%; verify corporate actions/adjusted prices before ingestion."
        ]

    return True, clean_df, warnings

def check_data_freshness(timestamp_str: Optional[str], max_age_seconds: int = 900) -> Dict[str, Any]:
    """
    Checks data timestamp freshness against current IST time (Finding 4.4).
    Flags future timestamps as errors and missing timestamps as DATA_UNAVAILABLE.
    """
    if not timestamp_str or str(timestamp_str).strip() in ("", "--", "None", "null"):
        return {"status": "DATA_UNAVAILABLE", "is_stale": True, "age_seconds": None, "error": "Missing timestamp"}

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
        diff_sec = (dt_now - dt).total_seconds()

        # Reject timestamps in the future (> 60s clock skew)
        if diff_sec < -60:
            return {
                "status": "FUTURE_TIMESTAMP_ERROR",
                "is_stale": True,
                "age_seconds": None,
                "timestamp": timestamp_str,
                "error": f"Timestamp is in the future ({int(abs(diff_sec))}s ahead of server IST)"
            }

        age_sec = max(0, int(diff_sec))
        is_stale = age_sec > max_age_seconds
        return {
            "status": "VALID" if not is_stale else "STALE",
            "is_stale": is_stale,
            "age_seconds": age_sec,
            "timestamp": timestamp_str
        }
    except Exception as ex:
        return {"status": "ERROR", "is_stale": True, "age_seconds": None, "error": str(ex)}
