import datetime
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from engines.nifty_analytics_engine import nifty_analytics_engine, now_ist, format_ist
from execution.state_engine import state_engine
from web.api import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_slot_determination_strictly_waits_for_slot_completion():
    """Verifies that 09:30 AM slot is NOT prematurely triggered at 09:16 or 09:28 AM."""
    # 09:16 AM
    dt_0916 = datetime.datetime(2026, 10, 14, 9, 16, 0)
    assert nifty_analytics_engine._determine_current_slot(dt_0916, "30m") is None

    # 09:28 AM - previous bug triggered 09:30 AM early; must now be None
    dt_0928 = datetime.datetime(2026, 10, 14, 9, 28, 59)
    assert nifty_analytics_engine._determine_current_slot(dt_0928, "30m") is None

    # 09:30 AM - exactly at interval completion
    dt_0930 = datetime.datetime(2026, 10, 14, 9, 30, 0)
    assert nifty_analytics_engine._determine_current_slot(dt_0930, "30m") == "09:30"

    # 09:45 AM - within 09:30 slot window for 30m mode
    dt_0945 = datetime.datetime(2026, 10, 14, 9, 45, 0)
    assert nifty_analytics_engine._determine_current_slot(dt_0945, "30m") == "09:30"

    # 10:00 AM - next 30m slot
    dt_1000 = datetime.datetime(2026, 10, 14, 10, 0, 0)
    assert nifty_analytics_engine._determine_current_slot(dt_1000, "30m") == "10:00"


def test_short_straddle_and_short_strangle_builder():
    """Verifies that SHORT_STRADDLE and SHORT_STRANGLE strategies are constructed with valid legs, credit, target, and SL."""
    strikes_data = [
        {"strike": 22400, "ce_ltp": 250.0, "pe_ltp": 40.0, "ce_iv": 13.0, "pe_iv": 14.0},
        {"strike": 22450, "ce_ltp": 210.0, "pe_ltp": 60.0, "ce_iv": 13.0, "pe_iv": 13.5},
        {"strike": 22500, "ce_ltp": 145.0, "pe_ltp": 140.0, "ce_iv": 13.2, "pe_iv": 13.1},
        {"strike": 22550, "ce_ltp": 95.0, "pe_ltp": 185.0, "ce_iv": 13.0, "pe_iv": 13.4},
        {"strike": 22600, "ce_ltp": 55.0, "pe_ltp": 240.0, "ce_iv": 13.5, "pe_iv": 14.0},
    ]

    # Test Short Straddle
    ss = nifty_analytics_engine._build_strategy_legs_and_metrics(
        strategy_type="SHORT_STRADDLE",
        strikes_data=strikes_data,
        atm_strike=22500,
        spot_price=22500.0,
        lot_size=75
    )
    assert ss["strategy_type"] == "SHORT_STRADDLE"
    assert len(ss["legs"]) == 2
    assert ss["legs"][0]["action"] == "SELL" and ss["legs"][0]["type"] == "CE" and ss["legs"][0]["strike"] == 22500
    assert ss["legs"][1]["action"] == "SELL" and ss["legs"][1]["type"] == "PE" and ss["legs"][1]["strike"] == 22500
    assert ss["net_premium"] > 0
    assert ss["target_pnl"] > 0
    assert ss["stoploss_pnl"] < 0
    assert len(ss["breakevens"]) == 2
    assert ss["breakevens"][0] < 22500 < ss["breakevens"][1]

    # Test Short Strangle
    sg = nifty_analytics_engine._build_strategy_legs_and_metrics(
        strategy_type="SHORT_STRANGLE",
        strikes_data=strikes_data,
        atm_strike=22500,
        spot_price=22500.0,
        lot_size=75
    )
    assert sg["strategy_type"] == "SHORT_STRANGLE"
    assert len(sg["legs"]) == 2
    assert sg["legs"][0]["action"] == "SELL" and sg["legs"][0]["type"] == "PE" and sg["legs"][0]["strike"] == 22400
    assert sg["legs"][1]["action"] == "SELL" and sg["legs"][1]["type"] == "CE" and sg["legs"][1]["strike"] == 22600
    assert sg["net_premium"] > 0


def test_premarket_prediction_locking_and_non_repainting():
    """Verifies that on trading days, locked premarket prediction is preserved and does not repaint during live hours."""
    today_str = now_ist().strftime("%Y-%m-%d")
    locked_key = f"nifty_prediction_locked_{today_str}"

    orig_persisted = state_engine.get_kv(locked_key)
    with nifty_analytics_engine._lock:
        orig_mem = nifty_analytics_engine._predictions_mem_cache

    try:
        mock_locked = {
            "status": "OK",
            "prediction_date": today_str,
            "is_premarket_locked": True,
            "premarket_locked_at_ist": "2026-10-14 08:50:00 IST",
            "premarket_spot_price": 22650.0,
            "composite_score": -14.5,
            "composite_score_t1": -14.5,
            "composite_score_t5": -18.2,
            "t1_forecast": {
                "verdict": "MODERATELY BEARISH",
                "prob_up": 0.20,
                "prob_flat": 0.30,
                "prob_down": 0.50,
                "expected_pts": -85.0,
                "expected_pct": -0.38,
                "expected_close": 22565.0,
                "range_1sigma": [22450.0, 22680.0],
                "basis_breakdown": {"verdict": "MODERATELY BEARISH"}
            },
            "t5_forecast": {
                "verdict": "MODERATELY BEARISH",
                "prob_up": 0.22,
                "prob_flat": 0.28,
                "prob_down": 0.50,
                "expected_pts": -180.0,
                "expected_pct": -0.80,
                "expected_close": 22470.0,
                "range_1sigma": [22300.0, 22700.0],
                "basis_breakdown": {"verdict": "MODERATELY BEARISH"}
            }
        }

        # Store locked forecast
        state_engine.set_kv(locked_key, mock_locked)
        with nifty_analytics_engine._lock:
            nifty_analytics_engine._predictions_mem_cache = mock_locked

        # Calling compute_nifty_predictions with force_refresh=True MUST NOT repaint or overwrite the locked prediction
        res = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=False)
        assert res["is_premarket_locked"] is True
        assert res["premarket_locked_at_ist"] == "2026-10-14 08:50:00 IST"
        assert res["premarket_spot_price"] == 22650.0
        assert res["t1_forecast"]["verdict"] == "MODERATELY BEARISH"
    finally:
        state_engine.set_kv(locked_key, orig_persisted)
        with nifty_analytics_engine._lock:
            nifty_analytics_engine._predictions_mem_cache = orig_mem


def test_premarket_telegram_bulletin_dispatch():
    """Verifies that dispatch_premarket_prediction_telegram_alert formats and delivers pre-market bulletin."""
    today_str = now_ist().strftime("%Y-%m-%d")
    mock_payload = {
        "status": "OK",
        "prediction_date": today_str,
        "is_premarket_locked": True,
        "premarket_locked_at_ist": "2026-10-14 08:50:00 IST",
        "premarket_spot_price": 22700.0,
        "composite_score_t1": -12.0,
        "composite_score_t5": -15.0,
        "t1_forecast": {
            "verdict": "MODERATELY BEARISH",
            "expected_pts": -65.0,
            "expected_pct": -0.29,
            "expected_close": 22635.0,
            "range_1sigma": [22550.0, 22750.0]
        },
        "t5_forecast": {
            "verdict": "MODERATELY BEARISH",
            "range_1sigma": [22400.0, 22800.0]
        },
        "crude_oil_status": {
            "ltp": 78.50,
            "ret_1d": 0.85,
            "impact": "HEADWIND"
        },
        "pillars": {
            "macro_cross_asset": [
                {"ticker": "^INDIAVIX", "ltp": 14.2}
            ]
        }
    }

    with patch("core.telegram_vault.telegram_vault.send_alert_and_sync") as mock_tg:
        sent = nifty_analytics_engine.dispatch_premarket_prediction_telegram_alert(mock_payload)
        assert sent is True
        assert mock_tg.called
        call_args = mock_tg.call_args[1]
        msg = call_args.get("message", "")
        assert "OFFICIAL NIFTY 50 PRE-MARKET FORECAST BULLETIN" in msg
        assert "MODERATELY BEARISH" in msg
        assert "Brent Crude: $78.50" in msg
        assert "LOCKED" in msg


def test_api_endpoints_premarket_lock_and_eod_eval(test_client):
    """Verifies that POST /api/nifty/premarket-forecast/lock and POST /api/nifty/eod-evaluation/run work cleanly."""
    with patch("core.telegram_vault.telegram_vault.send_alert_and_sync"):
        with patch.object(state_engine, "get_kv", return_value=None):
            with patch.object(nifty_analytics_engine, "compute_nifty_predictions", return_value={
                "status": "OK",
                "prediction_date": "2026-10-14",
                "premarket_locked_at_ist": "2026-10-14 08:45:00 IST",
                "t1_forecast": {"verdict": "MODERATELY BEARISH"},
                "t5_forecast": {"verdict": "MODERATELY BEARISH"}
            }):
                res_lock = test_client.post("/api/nifty/premarket-forecast/lock")
                assert res_lock.status_code == 200
                data_lock = res_lock.json()
                assert data_lock["status"] in ("OK", "LOCKED")
                assert data_lock["t1_verdict"] == "MODERATELY BEARISH"

        with patch.object(nifty_analytics_engine, "evaluate_and_grade_eod_predictions", return_value={
            "status": "OK",
            "outcome": "CORRECT",
            "cone_hit": True
        }):
            res_eval = test_client.post("/api/nifty/eod-evaluation/run")
            assert res_eval.status_code == 200
            data_eval = res_eval.json()
            assert data_eval["status"] == "OK"
            assert data_eval["outcome"] == "CORRECT"


def test_forward_testing_ledger_dates_and_eod_lifecycle():
    """
    Verifies that the forward-testing ledger keys every prediction by the Target Session Date (target_date),
    strictly preserves the Data Cutoff Date (data_date), and marks active forecasts as PENDING until EOD.
    """
    preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=True)
    assert preds["status"] == "OK"
    target_dt = preds.get("target_date")
    data_dt = preds.get("data_date")
    assert target_dt is not None
    assert data_dt is not None
    assert preds.get("data_as_of_date") == data_dt

    ledger = preds.get("prediction_ledger", [])
    assert len(ledger) > 0

    top_entry = ledger[0]
    # Top entry must be for the active target session
    assert top_entry["prediction_date"] == target_dt
    assert top_entry["target_date"] == target_dt
    assert top_entry["data_date"] == data_dt
    assert top_entry["t1_validated"] == "PENDING"
    assert top_entry["actual_t1_close"] is None

    # Verify prior entries are closed and graded
    if len(ledger) > 1:
        prev_entry = ledger[1]
        assert prev_entry["actual_t1_close"] is not None
        assert prev_entry["t1_validated"] in ("VALIDATED_HIT", "DIRECTION_MISS")
        assert prev_entry["target_date"] is not None
        assert prev_entry["data_date"] is not None


def test_strict_market_hours_lock():
    """Verifies that between 09:00 and 15:30 IST on a trading day, recomputation is strictly blocked even with force_recalculate=True."""
    from core.market_calendar import NSEMarketCalendar
    today_str = now_ist().strftime("%Y-%m-%d")
    mock_pred = {
        "status": "OK",
        "prediction_date": today_str,
        "is_premarket_locked": True,
        "spot_price": 24500.0,
        "t1_forecast": {"verdict": "STRONG BULLISH"},
        "t5_forecast": {"verdict": "MODERATELY BULLISH"}
    }
    old_cache = nifty_analytics_engine._predictions_mem_cache
    try:
        with patch.object(NSEMarketCalendar, "is_trading_day", return_value=True):
            fake_market_dt = datetime.datetime.strptime(f"{today_str} 11:30:00", "%Y-%m-%d %H:%M:%S")
            with patch("engines.nifty_analytics_engine.now_ist", return_value=fake_market_dt):
                with patch.object(state_engine, "get_kv", return_value=mock_pred):
                    nifty_analytics_engine._predictions_mem_cache = mock_pred
                    res = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=True)
                    assert res["status"] == "OK"
                    assert res.get("recompute_blocked") is True
                    assert res.get("market_hours_locked") is True
                    assert res.get("lock_status") == "MARKET_HOURS_STRICT_LOCK"
    finally:
        nifty_analytics_engine._predictions_mem_cache = old_cache


def test_macro_events_filter_completed_and_past():
    """Verifies that completed events and past events are auto-removed from the news and macro catalysts section."""
    today_dt = now_ist().date()
    news = nifty_analytics_engine._fetch_live_news_and_events()
    events = news.get("central_bank_events", [])
    for ev in events:
        assert ev.get("status") != "COMPLETED", f"Completed event found in upcoming list: {ev}"
        assert ev.get("days_from_today", 0) >= 0, f"Past event found in upcoming list: {ev}"


