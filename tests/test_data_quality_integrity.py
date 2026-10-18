"""Regression tests for fail-closed market-data validation."""

import numpy as np
import pandas as pd

from core.data_quality import check_data_freshness, validate_ohlcv_dataframe


def valid_frame(rows: int = 25) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(rows, 1000.0),
        },
        index=idx,
    )


def test_valid_ohlcv_is_accepted_without_repair():
    frame = valid_frame()
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST", min_bars=20)
    assert ok
    pd.testing.assert_frame_equal(cleaned, frame)
    assert warnings == []


def test_missing_volume_is_rejected_not_fabricated():
    frame = valid_frame().drop(columns=["Volume"])
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("Missing required OHLCV" in warning for warning in warnings)


def test_non_finite_values_are_rejected():
    frame = valid_frame()
    frame.iloc[3, frame.columns.get_loc("Close")] = np.inf
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("infinite" in warning for warning in warnings)


def test_negative_volume_is_rejected_not_clipped():
    frame = valid_frame()
    frame.iloc[3, frame.columns.get_loc("Volume")] = -1
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("negative volume" in warning for warning in warnings)


def test_invalid_candle_geometry_is_rejected_not_expanded():
    frame = valid_frame()
    frame.iloc[3, frame.columns.get_loc("High")] = frame.iloc[3]["Close"] - 2
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("candle geometry" in warning for warning in warnings)


def test_duplicate_timestamps_are_rejected():
    frame = valid_frame()
    frame.index = list(frame.index[:-1]) + [frame.index[-2]]
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("Duplicate timestamps" in warning for warning in warnings)


def test_future_dated_bars_are_rejected():
    frame = valid_frame()
    frame.index = pd.date_range(
        pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(days=2),
        periods=len(frame),
        freq="D",
    )
    ok, cleaned, warnings = validate_ohlcv_dataframe(frame, "TEST")
    assert not ok
    assert cleaned.empty
    assert any("Future-dated" in warning for warning in warnings)


def test_missing_freshness_timestamp_is_unavailable_not_fresh():
    result = check_data_freshness(None)
    assert result["status"] == "DATA_UNAVAILABLE"
    assert result["is_stale"] is True
    assert result["age_seconds"] is None


def test_future_freshness_timestamp_is_not_valid():
    future = (pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(hours=1))
    timestamp = future.strftime("%Y-%m-%d %H:%M:%S IST")
    result = check_data_freshness(timestamp)
    assert result["status"] == "FUTURE_TIMESTAMP_ERROR"
    assert result["is_stale"] is True
    assert result["age_seconds"] is None
