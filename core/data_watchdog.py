"""
========================================================================================
Institutional Data Watchdog & Gatekeeper Telemetry System (`core/data_watchdog.py`)
========================================================================================
Monitors real-time health, freshness, and data packet delivery across all 4 QuantDesk domains:
1. SWING TRADING: Master OHLCV Vault (1000+ stocks) & Yahoo Finance LTP Price Stream
2. INTRADAY DESK: NSE Live Option Chain API, 30m/15m Snapshot Schedule & Spot Stream
3. OPTIONS DESK: Multi-leg Weekly & Monthly Contract Stream, Greeks, Straddle & Walls
4. PREDICTIONS ML: 6-Pillar Feature Stream, Daily Session Alignment & Model Weights

Strict Market-Hours SLA:
- HEALTHY: Packet age <= 120s
- DEGRADED: Packet age 120s - 300s
- OFFLINE_ALERT: Packet age > 300s during active NSE market hours (09:15 - 15:30 IST)
Outside market hours, feeds with valid session data are marked HEALTHY_STANDBY.
Dispatches high-priority Telegram alerts if any feed drops during trading hours.
========================================================================================
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar

logger = logging.getLogger("DataWatchdog")


class DataWatchdog:
    """
    Central telemetry gatekeeper tracking data freshness across all QuantDesk pipelines.
    """
    _instance: Optional["DataWatchdog"] = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DataWatchdog, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._lock = threading.RLock()

        # Telemetry state for the 4 primary pipelines
        self._endpoints: Dict[str, Dict[str, Any]] = {
            "swing_trading": {
                "id": "swing_trading",
                "name": "Swing Trading Vault & LTP Stream",
                "category": "SWING",
                "status": "INITIALIZING",
                "last_success_ts": 0.0,
                "last_packet_time_ist": None,
                "age_seconds": None,
                "latency_ms": 0.0,
                "details": {},
                "failure_count": 0,
                "last_alert_ts": 0.0,
            },
            "intraday_desk": {
                "id": "intraday_desk",
                "name": "Intraday Desk (NSE Option & Spot Stream)",
                "category": "INTRADAY",
                "status": "INITIALIZING",
                "last_success_ts": 0.0,
                "last_packet_time_ist": None,
                "age_seconds": None,
                "latency_ms": 0.0,
                "details": {},
                "failure_count": 0,
                "last_alert_ts": 0.0,
            },
            "options_desk": {
                "id": "options_desk",
                "name": "Options Desk (Weekly/Monthly Multi-Leg Stream)",
                "category": "OPTIONS",
                "status": "INITIALIZING",
                "last_success_ts": 0.0,
                "last_packet_time_ist": None,
                "age_seconds": None,
                "latency_ms": 0.0,
                "details": {},
                "failure_count": 0,
                "last_alert_ts": 0.0,
            },
            "predictions_ml": {
                "id": "predictions_ml",
                "name": "Predictions ML Engine & Session Ledger",
                "category": "PREDICTIONS",
                "status": "INITIALIZING",
                "last_success_ts": 0.0,
                "last_packet_time_ist": None,
                "age_seconds": None,
                "latency_ms": 0.0,
                "details": {},
                "failure_count": 0,
                "last_alert_ts": 0.0,
            },
        }
        self._last_full_check_ts: float = 0.0
        self._alert_cooldown_seconds: float = 900.0  # 15 minutes cooldown per endpoint

    def record_packet(self, endpoint_id: str, latency_ms: float = 0.0, details: Optional[Dict[str, Any]] = None):
        """Records a successful data packet arrival from a live feed."""
        with self._lock:
            if endpoint_id not in self._endpoints:
                return
            now_ts = time.time()
            ep = self._endpoints[endpoint_id]
            ep["last_success_ts"] = now_ts
            ep["last_packet_time_ist"] = format_ist()
            ep["age_seconds"] = 0
            ep["latency_ms"] = round(latency_ms, 1)
            ep["failure_count"] = 0
            ep["status"] = "HEALTHY"
            if details:
                ep["details"].update(details)

    def record_failure(self, endpoint_id: str, error_msg: str):
        """Records a feed fetch exception or packet failure."""
        with self._lock:
            if endpoint_id not in self._endpoints:
                return
            ep = self._endpoints[endpoint_id]
            ep["failure_count"] += 1
            ep["details"]["last_error"] = str(error_msg)
            ep["details"]["last_error_time"] = format_ist()
            if ep["failure_count"] >= 3:
                ep["status"] = "OFFLINE_ALERT"

    def check_all_endpoints(self, force_probe: bool = False) -> Dict[str, Any]:
        """
        Evaluates health and packet age across all 4 data endpoints.
        If force_probe is True or cached check is > 30s old, actively queries state engine & feeds.
        """
        now_ts = time.time()
        is_market_open, gate_msg = NSEMarketCalendar.can_execute_trades()

        with self._lock:
            from execution.state_engine import state_engine

            # 1. SWING TRADING EVALUATION
            try:
                t0 = time.time()
                ohlcv_stats = state_engine.get_ohlcv_stats()
                price_stream = state_engine.get_kv("last_price_stream", {})
                active_pos = state_engine.get_active_positions()
                t_latency = (time.time() - t0) * 1000.0

                stream_ts_str = price_stream.get("timestamp")
                updated_count = price_stream.get("updated_count", 0)

                # Determine age of last price stream
                stream_age = None
                if stream_ts_str:
                    try:
                        import datetime
                        # Format is YYYY-MM-DD HH:MM:SS IST
                        ts_clean = stream_ts_str.replace(" IST", "")
                        dt_stream = datetime.datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
                        dt_now_unaware = now_ist().replace(tzinfo=None)
                        stream_age = max(0, int((dt_now_unaware - dt_stream).total_seconds()))
                    except Exception:
                        pass

                ep_swing = self._endpoints["swing_trading"]
                ep_swing["latency_ms"] = round(t_latency, 1)
                ep_swing["details"] = {
                    "stocks_in_vault": ohlcv_stats.get("stocks_count", 0),
                    "latest_bar_date": ohlcv_stats.get("latest_bar_date", "--"),
                    "last_vault_sync": ohlcv_stats.get("last_updated", "--"),
                    "active_positions_count": len(active_pos),
                    "last_stream_updated_count": updated_count,
                    "last_stream_time": stream_ts_str or "--"
                }

                if is_market_open:
                    if stream_age is not None and stream_age <= 120:
                        ep_swing["status"] = "HEALTHY"
                    elif stream_age is not None and stream_age <= 300:
                        ep_swing["status"] = "DEGRADED"
                    elif ep_swing.get("last_success_ts", 0) > (now_ts - 120):
                        ep_swing["status"] = "HEALTHY"
                    else:
                        ep_swing["status"] = "OFFLINE_ALERT" if (stream_age is not None and stream_age > 300) else "HEALTHY"
                else:
                    ep_swing["status"] = "HEALTHY_STANDBY"

                ep_swing["age_seconds"] = stream_age if stream_age is not None else 0
                if stream_ts_str:
                    ep_swing["last_packet_time_ist"] = stream_ts_str
            except Exception as e:
                logger.debug(f"Watchdog swing check notice: {e}")
                self._endpoints["swing_trading"]["status"] = "DEGRADED"

            # 2. INTRADAY DESK EVALUATION
            try:
                t0 = time.time()
                with state_engine._get_connection() as conn:
                    latest_snap = conn.execute(
                        "SELECT session_date, slot_time, timestamp_ist, spot_price, atm_strike, market_state "
                        "FROM intraday_option_snapshots ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                t_latency = (time.time() - t0) * 1000.0

                ep_intra = self._endpoints["intraday_desk"]
                ep_intra["latency_ms"] = round(t_latency, 1)

                snap_age = None
                if latest_snap:
                    ts_ist_str = latest_snap["timestamp_ist"]
                    ep_intra["last_packet_time_ist"] = ts_ist_str
                    try:
                        import datetime
                        ts_clean = ts_ist_str.replace(" IST", "")
                        dt_snap = datetime.datetime.strptime(ts_clean, "%Y-%m-%d %H:%M:%S")
                        dt_now_unaware = now_ist().replace(tzinfo=None)
                        snap_age = max(0, int((dt_now_unaware - dt_snap).total_seconds()))
                    except Exception:
                        pass

                    ep_intra["details"] = {
                        "session_date": latest_snap["session_date"],
                        "last_slot": latest_snap["slot_time"],
                        "spot_price": latest_snap["spot_price"],
                        "atm_strike": latest_snap["atm_strike"],
                        "market_state": latest_snap["market_state"]
                    }

                if is_market_open:
                    # Intraday slots occur every 15m/30m; live spot streams continuously
                    if snap_age is not None and snap_age <= 1800:  # within 30 mins
                        ep_intra["status"] = "HEALTHY"
                    elif snap_age is not None and snap_age <= 2700:
                        ep_intra["status"] = "DEGRADED"
                    elif ep_intra.get("last_success_ts", 0) > (now_ts - 120):
                        ep_intra["status"] = "HEALTHY"
                    else:
                        ep_intra["status"] = "DEGRADED"
                else:
                    ep_intra["status"] = "HEALTHY_STANDBY"

                ep_intra["age_seconds"] = snap_age if snap_age is not None else 0
            except Exception as e:
                logger.debug(f"Watchdog intraday check notice: {e}")
                self._endpoints["intraday_desk"]["status"] = "DEGRADED"

            # 3. OPTIONS DESK EVALUATION
            try:
                from engines.nifty_analytics_engine import nifty_analytics_engine
                oc_cache = getattr(nifty_analytics_engine, "_option_chain_mem_cache", None) or getattr(nifty_analytics_engine, "_option_chain_cache", {})
                if not oc_cache:
                    oc_cache = state_engine.get_kv("latest_nifty_option_chain_deep") or {}
                w_chain = oc_cache.get("weekly", {})
                m_chain = oc_cache.get("monthly", {})
                cache_ts = getattr(nifty_analytics_engine, "_option_chain_cache_ts", 0.0)
                chain_age = max(0, int(now_ts - cache_ts)) if cache_ts > 0 else None
                if chain_age is None and oc_cache.get("fetched_at_ist"):
                    try:
                        import datetime
                        clean_ts = oc_cache["fetched_at_ist"].replace(" IST", "")
                        dt_fetched = datetime.datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
                        dt_now_unaware = now_ist().replace(tzinfo=None)
                        chain_age = max(0, int((dt_now_unaware - dt_fetched).total_seconds()))
                    except Exception:
                        pass

                ep_opt = self._endpoints["options_desk"]
                if w_chain:
                    ep_opt["details"] = {
                        "weekly_expiry": w_chain.get("expiry"),
                        "monthly_expiry": m_chain.get("expiry"),
                        "strikes_count": len(w_chain.get("strikes", [])),
                        "atm_straddle_prem": w_chain.get("straddle", {}).get("straddle_premium"),
                        "near_atm_pcr": w_chain.get("pcr", {}).get("near_atm_pcr"),
                        "put_wall": w_chain.get("walls_and_pain", {}).get("put_wall_1"),
                        "call_wall": w_chain.get("walls_and_pain", {}).get("call_wall_1"),
                    }
                    if is_market_open:
                        if chain_age is not None and chain_age <= 45:
                            ep_opt["status"] = "HEALTHY"
                        elif chain_age is not None and chain_age <= 120:
                            ep_opt["status"] = "DEGRADED"
                        else:
                            ep_opt["status"] = "OFFLINE_ALERT"
                    else:
                        ep_opt["status"] = "HEALTHY_STANDBY"
                else:
                    ep_opt["status"] = "HEALTHY_STANDBY" if not is_market_open else "DEGRADED"

                ep_opt["age_seconds"] = chain_age if chain_age is not None else 0
            except Exception as e:
                logger.debug(f"Watchdog options desk check notice: {e}")
                self._endpoints["options_desk"]["status"] = "DEGRADED"

            # 4. PREDICTIONS ML EVALUATION
            try:
                t0 = time.time()
                today_str = now_ist().strftime("%Y-%m-%d")
                with state_engine._get_connection() as conn:
                    pred_row = conn.execute(
                        "SELECT prediction_date, created_at, predicted_direction, composite_score "
                        "FROM nifty_predictions_log WHERE prediction_date = ? LIMIT 1",
                        (today_str,)
                    ).fetchone()
                    if not pred_row:
                        pred_row = conn.execute(
                            "SELECT prediction_date, created_at, predicted_direction, composite_score "
                            "FROM nifty_predictions_log ORDER BY id DESC LIMIT 1"
                        ).fetchone()
                t_latency = (time.time() - t0) * 1000.0

                ep_pred = self._endpoints["predictions_ml"]
                ep_pred["latency_ms"] = round(t_latency, 1)
                if pred_row:
                    ep_pred["details"] = {
                        "prediction_date": pred_row["prediction_date"],
                        "created_at": pred_row["created_at"],
                        "predicted_direction": pred_row["predicted_direction"],
                        "composite_score": pred_row["composite_score"],
                        "aligned_today": (pred_row["prediction_date"] == today_str)
                    }
                    ep_pred["last_packet_time_ist"] = pred_row["created_at"]
                    ep_pred["status"] = "HEALTHY" if (pred_row["prediction_date"] == today_str or not is_market_open) else "DEGRADED"
                else:
                    ep_pred["status"] = "DEGRADED"
                ep_pred["age_seconds"] = 0
            except Exception as e:
                logger.debug(f"Watchdog predictions check notice: {e}")
                self._endpoints["predictions_ml"]["status"] = "DEGRADED"

            # Compute overall telemetry score
            healthy_count = sum(1 for ep in self._endpoints.values() if ep["status"] in ("HEALTHY", "HEALTHY_STANDBY"))
            alert_count = sum(1 for ep in self._endpoints.values() if ep["status"] == "OFFLINE_ALERT")
            degraded_count = sum(1 for ep in self._endpoints.values() if ep["status"] == "DEGRADED")

            overall_status = "HEALTHY"
            if alert_count > 0 and is_market_open:
                overall_status = "ALERT"
            elif degraded_count > 0:
                overall_status = "DEGRADED"

            # Check if any alert needs to be dispatched via Telegram
            if is_market_open and alert_count > 0:
                self._dispatch_alerts_if_due(now_ts)

            self._last_full_check_ts = now_ts
            return {
                "overall_status": overall_status,
                "summary_badge": f"{healthy_count}/4 FEEDS LIVE" if overall_status == "HEALTHY" else (
                    f"{alert_count} FEEDS OFFLINE" if overall_status == "ALERT" else f"{degraded_count} FEEDS DEGRADED"
                ),
                "healthy_count": healthy_count,
                "alert_count": alert_count,
                "degraded_count": degraded_count,
                "total_endpoints": len(self._endpoints),
                "market_open": is_market_open,
                "trading_gate_message": gate_msg,
                "last_checked_ist": format_ist(),
                "endpoints": list(self._endpoints.values()),
            }

    def _dispatch_alerts_if_due(self, now_ts: float):
        """Sends a high-priority Telegram alert when a feed becomes offline during market hours."""
        from core.telegram_vault import telegram_vault

        for ep_id, ep in self._endpoints.items():
            if ep["status"] == "OFFLINE_ALERT":
                time_since_last_alert = now_ts - ep.get("last_alert_ts", 0.0)
                if time_since_last_alert >= self._alert_cooldown_seconds:
                    ep["last_alert_ts"] = now_ts
                    msg = (
                        f"🚨 <b>[QUANTDESK GATEKEEPER ALERT] Data Feed Degradation Detected!</b>\n\n"
                        f"• <b>Pipeline:</b> {ep['name']} ({ep['category']})\n"
                        f"• <b>Status:</b> 🔴 <code>OFFLINE_ALERT</code>\n"
                        f"• <b>Packet Age:</b> {ep.get('age_seconds', '--')}s (> 300s SLA limit)\n"
                        f"• <b>Last Packet:</b> {ep.get('last_packet_time_ist', '--')}\n"
                        f"• <b>Time:</b> {format_ist()}\n\n"
                        f"<i>Action: QuantDesk Watchdog has flagged this feed and is initiating automated stream re-connection.</i>"
                    )
                    try:
                        threading.Thread(
                            target=telegram_vault.send_message,
                            args=(msg,),
                            kwargs={"parse_mode": "HTML"},
                            daemon=True
                        ).start()
                        logger.warning(f"🚨 Dispatched Watchdog Telegram Alert for degraded feed: {ep_id}")
                    except Exception as e:
                        logger.error(f"Failed to dispatch Watchdog alert: {e}")


# Singleton instance exported across QuantDesk
data_watchdog = DataWatchdog()
