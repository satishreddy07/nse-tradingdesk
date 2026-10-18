"""
Core Data Quality, Validation & Timezone Utilities.

The validators in this module are intentionally fail-closed: they must not repair
market prices or manufacture volume values because doing so can turn corrupt input
into apparently tradable data.
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
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")


def validate_ohlcv_dataframe(
    df: Optional[pd.DataFrame],
    symbol: str = "",
    min_bars: int = 20,
) -> Tuple[bool, pd.DataFrame, List[str]]:
    """
    Validate an OHLCV frame without inventing or silently repairing market data.

    Invalid critical data rejects the entire frame (returns False and an empty
    DataFrame). Sorting an otherwise valid frame is allowed and explicitly warned.
    Extreme returns are warned about, not automatically rejected, because corporate
    actions and genuine market events can cause large moves.
    """
    warnings: List[str] = []
    prefix = f"{symbol}: " if symbol else ""

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, pd.DataFrame(), [f"{prefix}DataFrame is None or empty."]
    if not isinstance(min_bars, int) or min_bars < 1:
        return False, pd.DataFrame(), [f"{prefix}min_bars must be a positive integer."]
    if len(df) < min_bars:
        return False, pd.DataFrame(), [
            f"{prefix}Insufficient bars ({len(df)} < required {min_bars})."
        ]

    clean_df = df.copy()
    col_map = {}
    for col in clean_df.columns:
        normalized = str(col).lower().strip()
        if normalized == "open":
            col_map[col] = "Open"
        elif normalized == "high":
            col_map[col] = "High"
        elif normalized == "low":
            col_map[col] = "Low"
        elif normalized == "close":
            col_map[col] = "Close"
        elif normalized in ("volume", "vol"):
            col_map[col] = "Volume"
    clean_df.rename(columns=col_map, inplace=True)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        return False, pd.DataFrame(), [
            f"{prefix}Missing required OHLCV columns {missing_cols}; no synthetic values were inserted."
        ]

    # A timestamp index is required so duplicate, unordered, and future bars cannot
    # pass as valid history merely because the rows happen to be present.
    try:
        parsed_index = pd.to_datetime(clean_df.index, errors="coerce")
        if pd.isna(parsed_index).any():
            return False, pd.DataFrame(), [
                f"{prefix}Index contains missing or unparseable timestamps."
            ]
        clean_df.index = pd.DatetimeIndex(parsed_index)
    except (TypeError, ValueError, OverflowError) as exc:
        return False, pd.DataFrame(), [
            f"{prefix}Index timestamps are invalid: {exc}"
        ]

    if clean_df.index.has_duplicates:
        dup_count = int(clean_df.index.duplicated(keep=False).sum())
        return False, pd.DataFrame(), [
            f"{prefix}Duplicate timestamps detected ({dup_count} rows involved); ambiguous bars were rejected."
        ]

    if not clean_df.index.is_monotonic_increasing:
        clean_df = clean_df.sort_index()
        warnings.append(f"{prefix}Timestamps were out of order; rows were sorted chronologically.")

    # Reject materially future-dated bars. Naive market timestamps are interpreted
    # as IST; timezone-aware timestamps are compared as absolute instants.
    now = pd.Timestamp.now(tz=IST)
    latest = clean_df.index.max()
    if latest.tzinfo is None:
        latest = latest.tz_localize(IST)
    else:
        latest = latest.tz_convert(IST)
    if latest > now + pd.Timedelta(seconds=60):
        return False, pd.DataFrame(), [
            f"{prefix}Future-dated bar detected ({latest.isoformat()}); frame rejected."
        ]

    # Do not silently drop malformed rows: a partial history can change indicators,
    # training labels, and risk calculations without downstream code noticing.
    for col in required_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    values = clean_df[required_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_rows = int((~np.isfinite(values)).any(axis=1).sum())
        return False, pd.DataFrame(), [
            f"{prefix}{bad_rows} rows contain NaN, infinite, or non-numeric OHLCV values; frame rejected."
        ]

    price_cols = ["Open", "High", "Low", "Close"]
    prices = clean_df[price_cols]
    non_positive = (prices <= 0).any(axis=1)
    if non_positive.any():
        return False, pd.DataFrame(), [
            f"{prefix}{int(non_positive.sum())} rows contain non-positive prices; frame rejected."
        ]

    negative_volume = clean_df["Volume"] < 0
    if negative_volume.any():
        return False, pd.DataFrame(), [
            f"{prefix}{int(negative_volume.sum())} rows contain negative volume; frame rejected."
        ]

    # Validate candle geometry. Never expand High/Low to conceal bad source data.
    invalid_geometry = (
        (clean_df["High"] < clean_df["Low"])
        | (clean_df["High"] < clean_df["Open"])
        | (clean_df["High"] < clean_df["Close"])
        | (clean_df["Low"] > clean_df["Open"])
        | (clean_df["Low"] > clean_df["Close"])
    )
    if invalid_geometry.any():
        return False, pd.DataFrame(), [
            f"{prefix}{int(invalid_geometry.sum())} rows violate OHLC candle geometry; frame rejected."
        ]

    returns = clean_df["Close"].pct_change().abs()
    extreme_jumps = returns > 0.50
    if extreme_jumps.any():
        warnings.append(
            f"{prefix}Detected {int(extreme_jumps.sum())} extreme close-to-close price jumps (>50%); verify corporate actions/source data."
        )

    return True, clean_df, warnings


def check_data_freshness(
    timestamp_str: Optional[str],
    max_age_seconds: int = 900,
) -> Dict[str, Any]:
    """
    Check timestamp freshness against current IST time.

    Missing, malformed, and future timestamps are never treated as fresh.
    """
    if not timestamp_str or str(timestamp_str).strip() in ("", "--", "None", "null"):
        return {
            "status": "DATA_UNAVAILABLE",
            "is_stale": True,
            "age_seconds": None,
            "error": "Missing timestamp",
        }

    try:
        clean_ts = str(timestamp_str).replace(" IST", "").strip()
        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(clean_ts, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return {
                "status": "INVALID_FORMAT",
                "is_stale": True,
                "age_seconds": None,
                "error": "Unsupported timestamp format",
            }

        dt_now = now_ist().replace(tzinfo=None)
        diff_sec = (dt_now - dt).total_seconds()
        if diff_sec < 0:
            return {
                "status": "FUTURE_TIMESTAMP_ERROR",
                "is_stale": True,
                "age_seconds": None,
                "timestamp": timestamp_str,
                "error": f"Timestamp is in the future ({int(abs(diff_sec))}s ahead of server IST)",
            }

        age_sec = int(diff_sec)
        is_stale = age_sec > max_age_seconds
        return {
            "status": "VALID" if not is_stale else "STALE",
            "is_stale": is_stale,
            "age_seconds": age_sec,
            "timestamp": timestamp_str,
        }
    except Exception as exc:
        logger.warning("Unable to validate data timestamp %r: %s", timestamp_str, exc)
        return {
            "status": "ERROR",
            "is_stale": True,
            "age_seconds": None,
            "error": str(exc),
        }
