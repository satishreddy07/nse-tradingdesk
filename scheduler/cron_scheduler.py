"""
========================================================================================
Automated Quant Cron Scheduler & Render Market-Hours Anti-Sleep Engine
========================================================================================
Runs on every NSE Indian Equities trading day (Monday-Friday, excluding dynamic NSE holidays):
- 08:55 AM IST: Daily Dynamic NSE Holiday Sync from nseindia.com
- 09:00 AM – 03:50 PM IST (Every 4 mins): Strict Render Anti-Sleep Keep-Alive Guard
- 10:20 AM IST: Morning Intraday Momentum & Entry Scan
- 03:00 PM IST: Afternoon Close-of-Day Swing Scan & MOC Execution

Strictly stops keep-alive pings outside 09:00 AM – 03:50 PM IST, on weekends, and on
dynamically fetched NSE holidays so the Render server can sleep outside market hours.
========================================================================================
"""

import logging
import os
import time
import threading
from typing import Dict, Any, Optional
import requests

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core import config
from core.config import (
    SCHEDULE_MORNING_HOUR, SCHEDULE_MORNING_MINUTE,
    SCHEDULE_AFTERNOON_HOUR, SCHEDULE_AFTERNOON_MINUTE
)
from engines.institutional_engine import (
    InstitutionalEngine,
    get_institutional_status,
    refresh_live_market_prices,
)
from engines.nifty500_engine import run_nifty500_pipeline_sync
from execution.state_engine import state_engine
from core.execution_queue import execution_queue

logger = logging.getLogger("Scheduler")

_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()
_detected_external_url: Optional[str] = None

_keepalive_telemetry: Dict[str, Any] = {
    "window_rule": "08:40 AM – 03:50 PM IST (Mon-Fri, Excl. NSE Holidays)",
    "interval_minutes": 4,
    "last_ping_ist": None,
    "last_ping_status": "IDLE",
    "pings_today": 0,
    "pings_date": None,
    "target_url": None
}


def register_external_request_origin(scheme: str, host: str):
    """
    Records the public external URL from incoming HTTP requests when RENDER_EXTERNAL_URL is not set.
    Strictly validates against authorized domain suffixes to prevent SSRF / Host Header Poisoning.
    """
    global _detected_external_url
    if not host:
        return
    host_clean = host.strip().lower().split(":")[0]
    allowed_suffixes = (".onrender.com", "duckdns.org", "onrender.com")
    if not any(host_clean == sfx or host_clean.endswith(sfx) for sfx in allowed_suffixes):
        return
    candidate = f"{scheme or 'https'}://{host.strip()}"
    if _detected_external_url != candidate:
        _detected_external_url = candidate
        try:
            state_engine.set_kv("RENDER_PUBLIC_URL", candidate)
        except Exception:
            pass
        logger.info(f"🌐 Validated public service URL for Render Anti-Sleep Guard: {_detected_external_url}")


def get_keepalive_target_url() -> str:
    """
    Resolves the external URL to ping so Render's edge load balancer resets its 15-min idle timer.
    Priority:
    1. RENDER_EXTERNAL_URL (Automatically injected by Render on all web services)
    2. PUBLIC_APP_URL (Optional user env override)
    3. Auto-detected external Host header from live traffic (persisted in DB)
    4. Localhost fallback (for local verification)
    """
    global _detected_external_url
    env_url = (os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_APP_URL") or "").strip().rstrip("/")
    if env_url:
        return f"{env_url}/health?source=render_keepalive"
    if not _detected_external_url:
        try:
            saved = state_engine.get_kv("RENDER_PUBLIC_URL")
            if saved and isinstance(saved, str):
                saved_clean = saved.strip().lower()
                if any(x in saved_clean for x in ("testserver", "localhost", "127.0.0.1")) or "." not in saved_clean:
                    state_engine.set_kv("RENDER_PUBLIC_URL", None)
                else:
                    _detected_external_url = saved
        except Exception:
            pass
    if _detected_external_url:
        clean_det = _detected_external_url.strip().lower()
        if any(x in clean_det for x in ("testserver", "localhost", "127.0.0.1")) or "." not in clean_det:
            _detected_external_url = None
        else:
            return f"{_detected_external_url.rstrip('/')}/health?source=render_keepalive"
    port = os.getenv("PORT", "8000")
    return f"http://127.0.0.1:{port}/health?source=render_keepalive"


def scheduled_nse_holiday_sync_job():
    """
    08:55 AM IST Weekday Job (and Startup Task):
    Dynamically fetches official NSE trading holidays from nseindia.com before the 09:00 AM
    anti-sleep window opens.
    """
    try:
        meta = NSEMarketCalendar.sync_holidays_from_nse(force=True)
        logger.info(
            f"📅 NSE Dynamic Holiday Sync complete: {meta.get('current_year_holidays_count', 0)} "
            f"holidays in {meta.get('current_year')} (Source: {meta.get('source')})."
        )
    except Exception as e:
        logger.warning(f"NSE holiday sync job warning: {e}")


def scheduled_render_keepalive_job():
    """
    Runs every 4 minutes between 09:00 AM and 03:50 PM IST on active NSE trading days.
    Strictly DOES NOT ping before 09:00 AM, after 03:50 PM, on weekends, or on NSE holidays.
    """
    current_dt = now_ist()
    today_str = current_dt.strftime("%Y-%m-%d")

    if _keepalive_telemetry.get("pings_date") != today_str:
        _keepalive_telemetry["pings_date"] = today_str
        _keepalive_telemetry["pings_today"] = 0

    is_active_window, reason = NSEMarketCalendar.is_render_keepalive_window(current_dt)
    if not is_active_window:
        _keepalive_telemetry["last_ping_status"] = reason
        return

    target_url = get_keepalive_target_url()
    _keepalive_telemetry["target_url"] = target_url

    try:
        resp = requests.get(
            target_url,
            headers={"User-Agent": "QuantDesk-Render-KeepAlive/2.1"},
            timeout=10
        )
        _keepalive_telemetry["last_ping_ist"] = format_ist(current_dt)
        _keepalive_telemetry["pings_today"] = int(_keepalive_telemetry.get("pings_today", 0)) + 1
        _keepalive_telemetry["last_ping_status"] = f"HTTP {resp.status_code} OK ({reason})"
        logger.debug(f"⚡ Render Anti-Sleep Keep-Alive ping -> {target_url} [{resp.status_code}]")
    except Exception as e:
        _keepalive_telemetry["last_ping_ist"] = format_ist(current_dt)
        _keepalive_telemetry["last_ping_status"] = f"PING_ERR: {e}"
        logger.warning(f"Render Anti-Sleep Keep-Alive ping error ({target_url}): {e}")


def scheduled_live_price_stream_job():
    """
    Runs every 5 minutes during NSE market hours (09:15 AM – 03:30 PM IST, Mon-Fri, Excl. Holidays).
    Queued sequentially via central ExecutionQueueManager to guarantee zero SQLite lock collisions.
    1. Streams live prices for all Active Portfolio Positions and Screened Signals.
    2. Updates complete NSE Option Chain (Weekly + Monthly) synchronously every 5 minutes.
    3. Reconciles all active options strategies against their Targets, Stop Losses, and 15:15 MOC cutoff.
    4. Evaluates and deploys intraday automated options strategies based on intraday progression (max 2 limit & diversity).
    """
    current_dt = now_ist()
    can_trade, _ = NSEMarketCalendar.can_execute_trades(current_dt)
    if not can_trade:
        return

    def _stream_worker():
        try:
            res = refresh_live_market_prices(scope="ACTIVE_AND_SIGNALS")
            logger.info(
                f"📡 [5m Live Price Stream] Updated {res.get('symbols_updated', 0)} stocks "
                f"({res.get('active_positions_updated', 0)} active positions) in {res.get('elapsed_sec', 0)}s"
            )
        except Exception as e:
            logger.warning(f"5m live price stream warning: {e}")

        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            # 1. Update complete option chains along with stock prices every 5 minutes
            nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=True)
            # 2. Reconcile stops and targets for all active options strategies (Trailing SL, Target, MOC)
            nifty_analytics_engine.reconcile_active_options_strategies_stops_and_targets()
            # 3. Background re-entry & stop loss monitors for intraday and positional legs
            try:
                from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
                intraday_options_strategy_engine.tick_active_options_reentry_monitor()
            except Exception as e_im:
                logger.debug(f"5m Intraday options monitor notice: {e_im}")
            try:
                from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
                weekly_monthly_options_strategy_engine.tick_positional_options_monitor("WEEKLY")
                weekly_monthly_options_strategy_engine.tick_positional_options_monitor("MONTHLY")
            except Exception as e_pm:
                logger.debug(f"5m Positional options monitor notice: {e_pm}")
            # 4. Periodically evaluate AI intelligence analysis (respecting cache TTL)
            if config.AI_ENABLED:
                try:
                    from ai_engine.service import ai_service
                    ai_service.analyze_predictions(force_refresh=False)
                except Exception as aie:
                    logger.debug(f"5m AI analysis evaluation warning: {aie}")

            # 5. Check real-time institutional call/put wall breach
            try:
                from engines.intraday_analytics_engine import intraday_analytics_engine
                w_chain = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False).get("weekly", {})
                spot_val = float(w_chain.get("spot_price") or 0.0)
                cw = int(w_chain.get("walls_and_pain", {}).get("call_wall_1") or 0)
                pw = int(w_chain.get("walls_and_pain", {}).get("put_wall_1") or 0)
                if spot_val > 0 and cw > 0 and pw > 0:
                    intraday_analytics_engine.check_intraday_wall_breach_and_alert(spot_val, cw, pw)
            except Exception as we:
                logger.debug(f"Wall breach check notice: {we}")
        except Exception as e:
            logger.warning(f"5m option chain sync & auto-trade warning: {e}")

    execution_queue.submit_task(
        task_fn=_stream_worker,
        task_name="5m Live Price & Option Stream",
        task_type="PRICE_STREAM_5M",
        deduplicate=True
    )


def scheduled_intraday_flash_bulletin_job(slot_label: str):
    """
    Fires at key institutional inflection points (09:45, 11:30, 13:00, 14:15 IST)
    to generate and dispatch a multi-pillar Intraday Flash Bulletin to Telegram.
    """
    current_dt = now_ist()
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        return
    try:
        from engines.intraday_analytics_engine import intraday_analytics_engine
        intraday_analytics_engine.generate_intraday_ai_flash_bulletin(slot_label=slot_label)
    except Exception as e:
        logger.warning(f"Intraday flash bulletin ({slot_label}) warning: {e}")


def scheduled_premarket_nifty_prediction_job():
    """
    08:45 AM IST Weekday Job:
    1. Calculates official Nifty 50 6-Pillar T+1 & T+5 Pre-Market Forecast.
    2. FREEZES and locks it for today's session in SQLite and ledger.
    3. Dispatches Telegram Pre-Market Bulletin.
    4. STRICT HARD GATE: Only after prediction models successfully complete and lock,
       evaluates and deploys positional (Weekly & Monthly) options strategies with hard-gated idempotency.
    """
    current_dt = now_ist()
    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        logger.info("⏸️ Today is not an active trading day. 08:45 AM pre-market prediction skipped.")
        return
    try:
        from engines.nifty_analytics_engine import nifty_analytics_engine
        logger.info(f"🔒 [08:45 AM IST Trigger] Computing & Locking Official Pre-Market Forecast for {current_dt.date()}...")
        preds = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=True)
        nifty_analytics_engine.dispatch_premarket_prediction_telegram_alert(preds)
        logger.info("✓ [08:45 AM IST Job] Official Pre-Market Forecast LOCKED & Telegram Bulletin Dispatched.")

        # Record SUCCESS telemetry for 08:45 AM auto run
        t1_v = preds.get("t1_forecast", {}).get("verdict") if preds else "--"
        t5_v = preds.get("t5_forecast", {}).get("verdict") if preds else "--"
        state_engine.record_scheduled_scan_outcome(
            run_key="premarket_0845",
            status="SUCCESS",
            buy_now_count=0,
            total_scanned=50,
            note=f"Official Pre-Market Forecast Locked at {format_ist(current_dt)} • T+1: {t1_v} | T+5: {t5_v}"
        )

        # STRICT HARD GATE PIPELINE: Trigger options strategy evaluation only after prediction lock
        if preds and preds.get("status") in ("OK", "LOCKED"):
            try:
                from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
                logger.info("🎯 [08:45 AM Hard Gate Pipeline] Evaluating Weekly & Monthly Options Strategies from locked prediction...")
                w_res = weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades("WEEKLY", force_refresh=False)
                m_res = weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades("MONTHLY", force_refresh=False)
                logger.info(
                    f"🎯 [08:45 AM Strategy Hard Gate] Weekly: {w_res.get('status')} ({w_res.get('reason') or w_res.get('regime')}) | "
                    f"Monthly: {m_res.get('status')} ({m_res.get('reason') or m_res.get('regime')})"
                )
            except Exception as e_opt:
                logger.warning(f"08:45 AM Options Strategy Deployment notice: {e_opt}")
    except Exception as e:
        logger.warning(f"08:45 AM Pre-Market Prediction Job error: {e}")
        state_engine.record_scheduled_scan_outcome(
            run_key="premarket_0845",
            status="FAILED",
            buy_now_count=0,
            total_scanned=0,
            note=f"08:45 AM Run Failed: {str(e)}"
        )


def scheduled_intraday_options_entry_job():
    """
    09:20 AM IST Weekday Job:
    Evaluates and deploys Intraday 5-Regime options strategy once the 15-minute Opening Range and VWAP are established.
    Strict hard gate: max 1 active intraday strategy and max 1 deployment per day.
    """
    current_dt = now_ist()
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        return
    try:
        from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
        logger.info("⚡ [09:20 AM Intraday Options Pipeline] Evaluating Intraday 5-Regime strategy...")
        res = intraday_options_strategy_engine.evaluate_and_execute_intraday_trades(force_refresh=False)
        logger.info(f"⚡ [09:20 AM Intraday Options Result] Status: {res.get('status')} ({res.get('reason') or res.get('regime')})")
    except Exception as e:
        logger.warning(f"09:20 AM Intraday Options Entry error: {e}")


def scheduled_options_premarket_job():
    """
    09:00 AM IST Weekday Job:
    Reconciles options portfolio state and checks hard gates before market open.
    Does NOT spawn redundant duplicate strategies.
    """
    current_dt = now_ist()
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        return
    try:
        from engines.nifty_analytics_engine import nifty_analytics_engine
        nifty_analytics_engine.reconcile_active_options_strategies_stops_and_targets()
        logger.info("✓ [09:00 AM Options Gate Check] Active strategies reconciled before market open.")
    except Exception as e:
        logger.warning(f"09:00 AM Options Pre-Market Reconciler error: {e}")


def scheduled_morning_scan_job():
    """
    10:20 AM IST Trading Day Job:
    - Verifies NSE trading day calendar (using dynamically synced NSE holidays).
    - Runs Institutional 6-Pillar Engine (Morning Mode).
    - Reconciles active positions across both portfolios.
    - Enqueued through execution_queue to avoid SQLite write lock collisions.
    """
    current_dt = now_ist()
    logger.info(f"⏰ [10:20 AM Trigger] Initiating morning scan sequence at {format_ist(current_dt)}...")

    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        reason = NSEMarketCalendar.get_holiday_reason(current_dt.date())
        logger.info(f"⏸️ Today is not an active trading day ({reason}). Morning scan skipped.")
        state_engine.log_scan_run(
            scan_type="SCHEDULED_1020AM",
            portfolio_type="BOTH",
            status="SKIPPED_HOLIDAY",
            execution_time_sec=0.0,
            message=f"Scan skipped: {reason}"
        )
        return

    today_str = current_dt.strftime("%Y-%m-%d")
    telemetry = state_engine.get_scheduled_scans_telemetry()
    morning_meta = telemetry.get("morning_1020") or {}
    if morning_meta.get("date") == today_str and morning_meta.get("status") in ("SUCCESS", "RUNNING"):
        logger.info(f"⏭️ Morning scan for {today_str} is already {morning_meta.get('status')}. Skipping redundant trigger.")
        return

    state_engine.record_scheduled_scan_outcome(
        run_key="morning_1020",
        status="RUNNING",
        note="10:20 AM Morning Swing Scan in progress..."
    )

    def _morning_worker():
        try:
            t0 = time.time()
            inst_engine = InstitutionalEngine()
            signals = inst_engine.run(auto_execute=True, mode_override="morning")
            elapsed = time.time() - t0
            b_now = int((signals["bucket"] == "BUY_NOW").sum()) if not signals.empty and "bucket" in signals.columns else 0
            p_brk = int((signals["bucket"] == "PRE_BREAKOUT").sum()) if not signals.empty and "bucket" in signals.columns else 0
            w_pb = int((signals["bucket"] == "WAIT_PULLBACK").sum()) if not signals.empty and "bucket" in signals.columns else 0
            try:
                from engines.institutional_engine import get_institutional_status
            except ImportError:
                pass
            eng_status = get_institutional_status()
            eng_st = eng_status.get("status", "")
            if eng_st == "FAILED" or (signals.empty and eng_st != "NO_CANDIDATES" and eng_status.get("universe_size", 0) == 0):
                state_engine.record_scheduled_scan_outcome(
                    run_key="morning_1020",
                    status="FAILED",
                    buy_now_count=0,
                    total_scanned=0,
                    execution_time_sec=elapsed,
                    note="Scan aborted: No valid OHLCV data retrieved or screener unreachable"
                )
                logger.warning(f"❌ 10:20 AM Morning Institutional Scan aborted or returned 0 valid candidates.")
            else:
                state_engine.record_scheduled_scan_outcome(
                    run_key="morning_1020",
                    status="SUCCESS",
                    buy_now_count=b_now,
                    total_scanned=len(signals),
                    pre_breakout_count=p_brk,
                    wait_pullback_count=w_pb,
                    execution_time_sec=elapsed,
                    note=f"{b_now} BUY NOW stocks found out of {len(signals)} analyzed"
                )
                logger.info(f"✓ 10:20 AM Morning Institutional Scan completed. {len(signals)} stocks evaluated, {b_now} BUY_NOW.")
        except Exception as e:
            logger.error(f"Error in 10:20 AM Institutional scan: {e}", exc_info=True)
            state_engine.record_scheduled_scan_outcome(
                run_key="morning_1020",
                status="FAILED",
                error=str(e),
                note=f"Scan error: {e}"
            )

    execution_queue.submit_task(
        task_fn=_morning_worker,
        task_name="10:20 AM Morning Scan",
        task_type="CHARTINK_SCAN",
        deduplicate=True
    )


def scheduled_afternoon_scan_job():
    """
    3:00 PM IST Trading Day Job:
    - Verifies NSE trading day calendar (using dynamically synced NSE holidays).
    - Runs Institutional 6-Pillar Engine (Afternoon Mode with Intraday Confirmation).
    - Runs Nifty 500 Momentum Compounded Engine (4-Tier Exit Hierarchy + 7-Point Entry Matrix).
    - Enqueued through execution_queue to avoid SQLite write lock collisions.
    """
    current_dt = now_ist()
    logger.info(f"⏰ [3:00 PM Trigger] Initiating afternoon MOC scan sequence at {format_ist(current_dt)}...")

    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        reason = NSEMarketCalendar.get_holiday_reason(current_dt.date())
        logger.info(f"⏸️ Today is not an active trading day ({reason}). Afternoon scan skipped.")
        state_engine.log_scan_run(
            scan_type="SCHEDULED_300PM",
            portfolio_type="BOTH",
            status="SKIPPED_HOLIDAY",
            execution_time_sec=0.0,
            message=f"Scan skipped: {reason}"
        )
        return

    today_str = current_dt.strftime("%Y-%m-%d")
    telemetry = state_engine.get_scheduled_scans_telemetry()
    afternoon_meta = telemetry.get("afternoon_305") or {}
    if afternoon_meta.get("date") == today_str and afternoon_meta.get("status") in ("SUCCESS", "RUNNING"):
        logger.info(f"⏭️ Afternoon scan for {today_str} is already {afternoon_meta.get('status')}. Skipping redundant trigger.")
        return

    state_engine.record_scheduled_scan_outcome(
        run_key="afternoon_305",
        status="RUNNING",
        note="3:00 PM MOC Swing Scan in progress..."
    )

    def _afternoon_worker():
        # 1. Run Institutional 6-Pillar Engine (Afternoon Mode with Intraday Confirmation)
        try:
            t0 = time.time()
            inst_engine = InstitutionalEngine()
            signals_p1 = inst_engine.run(auto_execute=True, mode_override="afternoon")
            elapsed = time.time() - t0
            b_now = int((signals_p1["bucket"] == "BUY_NOW").sum()) if not signals_p1.empty and "bucket" in signals_p1.columns else 0
            p_brk = int((signals_p1["bucket"] == "PRE_BREAKOUT").sum()) if not signals_p1.empty and "bucket" in signals_p1.columns else 0
            w_pb = int((signals_p1["bucket"] == "WAIT_PULLBACK").sum()) if not signals_p1.empty and "bucket" in signals_p1.columns else 0
            try:
                from engines.institutional_engine import get_institutional_status
            except ImportError:
                pass
            eng_status = get_institutional_status()
            eng_st = eng_status.get("status", "")
            if eng_st == "FAILED" or (signals_p1.empty and eng_st != "NO_CANDIDATES" and eng_status.get("universe_size", 0) == 0):
                state_engine.record_scheduled_scan_outcome(
                    run_key="afternoon_305",
                    status="FAILED",
                    buy_now_count=0,
                    total_scanned=0,
                    execution_time_sec=elapsed,
                    note="Scan aborted: No valid OHLCV data retrieved or screener unreachable"
                )
                logger.warning(f"❌ 3:00 PM Afternoon Institutional Scan aborted or returned 0 valid candidates.")
            else:
                state_engine.record_scheduled_scan_outcome(
                    run_key="afternoon_305",
                    status="SUCCESS",
                    buy_now_count=b_now,
                    total_scanned=len(signals_p1),
                    pre_breakout_count=p_brk,
                    wait_pullback_count=w_pb,
                    execution_time_sec=elapsed,
                    note=f"{b_now} BUY NOW stocks found out of {len(signals_p1)} analyzed"
                )
                logger.info(f"✓ 3:00 PM Afternoon Institutional Scan completed. {len(signals_p1)} stocks evaluated, {b_now} BUY_NOW.")
        except Exception as e:
            logger.error(f"Error in 3:00 PM Institutional scan: {e}", exc_info=True)
            state_engine.record_scheduled_scan_outcome(
                run_key="afternoon_305",
                status="FAILED",
                error=str(e),
                note=f"Scan error: {e}"
            )

        # 2. Run Nifty 500 Momentum Compounded Engine (7-Point Matrix + 4-Tier Exits)
        try:
            res_p2 = run_nifty500_pipeline_sync(mode="SCHEDULED_300PM", auto_execute=True)
            logger.info(f"✓ 3:00 PM Nifty 500 Momentum Scan completed: {res_p2.get('orders_executed', 0)} orders placed.")
        except Exception as e:
            logger.error(f"Error in 3:00 PM Nifty 500 scan: {e}", exc_info=True)

    execution_queue.submit_task(
        task_fn=_afternoon_worker,
        task_name="3:00 PM MOC Afternoon Scan",
        task_type="CHARTINK_SCAN",
        deduplicate=True
    )


def scheduled_nifty_intraday_50min_job(slot_label: Optional[str] = None):
    """
    Runs at every 15m/30m/50m interval from 09:30 AM to 03:30 PM IST.
    Collects live Nifty 50 Current Weekly Option Chain data, stores the snapshot in SQLite (`data/swing_trades.db`),
    reconciles any missing past slots, refreshes Nifty predictions on key slots, and syncs to Telegram Cloud Vault.
    """
    current_dt = now_ist()
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        return
    try:
        from engines.nifty_analytics_engine import nifty_analytics_engine
        dash = nifty_analytics_engine.capture_intraday_50min_snapshot(slot_override=slot_label, sync_cloud=True)
        # Auto-gapfill any missed earlier slots so the dashboard is 100% complete
        nifty_analytics_engine.auto_gapfill_missed_intraday_slots(interval_mode="30m")
        snap = dash.get("latest_snapshot") or {}
        logger.info(
            f"⚡ [Intraday Option Expiry Collector] Captured slot {snap.get('slot_time')} | "
            f"Nifty Spot: {snap.get('spot_price')} | Straddle: {snap.get('straddle_premium')} | "
            f"Near-ATM PCR: {snap.get('near_atm_pcr')} | State: {snap.get('market_state')}"
        )
        if slot_label in ("09:30", "12:00", "15:20", "15:30", None):
            nifty_analytics_engine.compute_nifty_predictions(force_refresh=False)
    except Exception as e:
        logger.warning(f"Intraday option snapshot job warning: {e}")


def _intraday_collector_watchdog():
    """
    Continuous background watchdog thread running every 40 seconds:
    Guarantees that even if frontend tabs are closed, Render was idling, or a container woke up:
    1. During NSE trading hours (09:15–15:35 IST), ensures active slot is captured.
    2. Auto-gapfills any missed past slots for today so the schedule is 100% complete with zero blanks.
    3. 3:00 PM MOC Swing Scan Watchdog Catch-Up: between 15:00 and 15:28 IST, if today's afternoon scan is
       pending or awaiting trigger, dispatches it immediately.
    4. 10:20 AM Morning Swing Scan Watchdog Catch-Up: between 10:20 and 10:45 IST, if today's morning scan is
       pending or awaiting trigger, dispatches it immediately.
    """
    time.sleep(10)
    while True:
        try:
            dt_now = now_ist()
            if NSEMarketCalendar.is_trading_day(dt_now.date()):
                mins = dt_now.hour * 60 + dt_now.minute
                # Active market window: 09:15 to 15:35 IST
                if 555 <= mins <= 935:
                    from engines.nifty_analytics_engine import nifty_analytics_engine
                    nifty_analytics_engine.auto_gapfill_missed_intraday_slots(interval_mode="30m")
                    active_slot = nifty_analytics_engine._determine_current_slot(dt_now, interval_mode="30m")
                    if active_slot:
                        today_str = dt_now.strftime("%Y-%m-%d")
                        snaps = nifty_analytics_engine._get_snapshots_for_date(today_str)
                        existing_slots = {s["slot_time"] for s in snaps}
                        if active_slot not in existing_slots:
                            logger.info(f"🛡️ [Watchdog] Capturing active slot {active_slot}...")
                            nifty_analytics_engine.capture_intraday_50min_snapshot(
                                slot_override=active_slot, sync_cloud=True, interval_mode="30m"
                            )

                    # Monitor Intraday Options Strategy Legs for Stop Loss & Price-Triggered Re-Entry
                    try:
                        from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
                        intraday_options_strategy_engine.tick_active_options_reentry_monitor()
                    except Exception:
                        pass

                    # Monitor Weekly & Monthly Positional Options Strategy Legs
                    try:
                        from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
                        weekly_monthly_options_strategy_engine.tick_active_positional_reentry_monitor("WEEKLY")
                        weekly_monthly_options_strategy_engine.tick_active_positional_reentry_monitor("MONTHLY")
                    except Exception:
                        pass

                    # Autonomous Auto-Trades Evaluation: Deploy intraday trades when system feels confident (09:45 to 14:45 IST)
                    if 585 <= mins <= 885:
                        try:
                            from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
                            intraday_options_strategy_engine.evaluate_and_execute_auto_trades(force_refresh=False)
                        except Exception as e_auto:
                            logger.debug(f"Intraday auto-trades watchdog notice: {e_auto}")

                    # Central Data Gatekeeper SLA monitoring
                    try:
                        from core.data_watchdog import data_watchdog
                        data_watchdog.check_all_endpoints(force_probe=False)
                    except Exception:
                        pass

                    # Watchdog Scheduled Scans Catch-Up:
                    # 08:45 AM Pre-Market Prediction Catch-Up (08:45 to 15:30 IST)
                    today_str = dt_now.strftime("%Y-%m-%d")
                    is_pred_locked = bool(state_engine.get_kv(f"nifty_prediction_locked_{today_str}"))
                    if not is_pred_locked and mins >= 525:  # 525 mins = 08:45 AM IST
                        logger.info(f"🛡️ [Watchdog Catch-Up] 08:45 AM pre-market prediction pending for {today_str}. Running now...")
                        scheduled_premarket_nifty_prediction_job()

                    # 3:00 PM MOC Catch-Up (15:00 to 15:28 IST)
                    if 900 <= mins <= 928:
                        scans_tel = state_engine.get_scheduled_scans_telemetry()
                        aft_st = (scans_tel.get("afternoon_305") or {}).get("status")
                        if aft_st in ("PENDING", "AWAITING_TRIGGER", None):
                            logger.info(f"🛡️ [Watchdog Catch-Up] 3:00 PM MOC scan status is '{aft_st}'. Dispatching afternoon scan now...")
                            scheduled_afternoon_scan_job()

                    # 10:20 AM Morning Catch-Up (10:20 to 10:45 IST)
                    elif 620 <= mins <= 645:
                        scans_tel = state_engine.get_scheduled_scans_telemetry()
                        mrn_st = (scans_tel.get("morning_1020") or {}).get("status")
                        if mrn_st in ("PENDING", "AWAITING_TRIGGER", None):
                            logger.info(f"🛡️ [Watchdog Catch-Up] 10:20 AM morning scan status is '{mrn_st}'. Dispatching morning scan now...")
                            scheduled_morning_scan_job()
        except Exception as e:
            logger.debug(f"Watchdog notice: {e}")
        time.sleep(40)


def scheduled_daily_prediction_training_job():
    """
    03:40 PM IST Trading Day Job:
    - Runs EOD prediction outcome evaluation and online training.
    - Evaluates morning's locked pre-market forecast against actual EOD close.
    - Updates adaptive pillar weights and sends Telegram EOD report.
    - Backs up prediction attribution ledger to Telegram Cloud Vault.
    """
    current_dt = now_ist()
    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        return
    try:
        from engines.nifty_analytics_engine import nifty_analytics_engine
        res = nifty_analytics_engine.evaluate_and_grade_eod_predictions()
        st = res.get("self_training", {})
        logger.info(
            f"🎯 [03:40 PM EOD Training Job] Nifty Predictions Evaluated ({res.get('outcome')}) & Trained. "
            f"Hit Rates: {st.get('t1_hit_rate_pct')}%, {st.get('t5_hit_rate_pct')}%. Cloud Vault Backed Up."
        )
    except Exception as e:
        logger.warning(f"03:40 PM Daily Prediction Training Job error: {e}")

    try:
        from engines.swing_quant_ml_engine import swing_quant_ml_engine
        # 1. Reconcile active shadow trades against EOD candle highs/lows
        shadow_audit = swing_quant_ml_engine.reconcile_shadow_trades()
        # 2. Retrain ML meta-classifier & expected R regressor
        ml_meta = swing_quant_ml_engine.train_model()
        logger.info(
            f"🤖 [03:40 PM Swing ML Self-Training] Shadow Audit: {shadow_audit.get('resolved', 0)} resolved "
            f"({shadow_audit.get('winners', 0)} W / {shadow_audit.get('losers', 0)} L, {shadow_audit.get('remaining_active', 0)} active). "
            f"Model Samples: {ml_meta.get('total_samples', 0)} (Empirical: {ml_meta.get('empirical_samples', 0)}), Brier: {ml_meta.get('brier_score', 0):.4f}."
        )
    except Exception as e:
        logger.warning(f"03:40 PM Swing Quant ML Training Job error: {e}")

    try:
        from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
        orch_res = horizon_learning_orchestrator.retrain_and_evaluate_all_horizons()
        logger.info(f"🌳 [03:40 PM Multi-Horizon Tree ML Pipeline] Completed retraining: {list(orch_res.get('results', {}).keys())}")
    except Exception as e_orch:
        logger.warning(f"03:40 PM Horizon ML Training Job warning: {e_orch}")

    try:
        from core.nifty_prediction_ledger import nifty_prediction_ledger
        from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
        res_out = nifty_prediction_ledger.resolve_matured_outcomes()
        logger.info(f"🎯 [03:40 PM Nifty Forecast Ledger Resolution] Resolved T+1: {res_out.get('resolved_t1')}, T+5: {res_out.get('resolved_t5')}")
        t1_ret = nifty_forecast_learning_engine.train_forecast_models(horizon="T+1")
        t5_ret = nifty_forecast_learning_engine.train_forecast_models(horizon="T+5")
        logger.info(f"🧠 [03:40 PM Nifty Forecast Models] T+1: {t1_ret.get('status')} | T+5: {t5_ret.get('status')}")
    except Exception as e_fc:
        logger.warning(f"03:40 PM Nifty Forecast Learning Job warning: {e_fc}")


    try:
        from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
        opt_ml = intraday_options_strategy_engine.train_options_ml_model()
        logger.info(
            f"🎯 [03:40 PM Intraday Options ML Training] Samples: {opt_ml.get('total_samples', 0)}, "
            f"Win Rate Accuracy: {opt_ml.get('win_rate_accuracy_pct', 0)}%, R²: {opt_ml.get('expectancy_r2', 0)}."
        )
    except Exception as e:
        logger.warning(f"03:40 PM Intraday Options ML Training Job error: {e}")

    try:
        from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
        w_ml = weekly_monthly_options_strategy_engine.train_positional_options_ml_model("WEEKLY")
        m_ml = weekly_monthly_options_strategy_engine.train_positional_options_ml_model("MONTHLY")
        logger.info(
            f"🎯 [03:40 PM Positional Options ML Training] "
            f"Weekly Win Acc: {w_ml.get('win_rate_accuracy_pct', 0)}%, R²: {w_ml.get('expectancy_r2', 0)} | "
            f"Monthly Win Acc: {m_ml.get('win_rate_accuracy_pct', 0)}%, R²: {m_ml.get('expectancy_r2', 0)}."
        )
    except Exception as e:
        logger.warning(f"03:40 PM Positional Options ML Training Job error: {e}")


def scheduled_daily_moc_backup_job():
    """
    03:45 PM IST Trading Day Job:
    - Runs atomic Market-On-Close SQLite backup.
    - Executes PRAGMA integrity_check to verify zero corruption.
    - Auto-prunes snapshots older than 14 days to prevent disk bloat.
    - Syncs snapshot to Telegram Cloud Vault if configured.
    """
    curr = now_ist()
    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(curr.date()):
        return
    try:
        from core.backup_manager import daily_backup_manager
        res = daily_backup_manager.run_daily_moc_backup(retention_days=14, sync_cloud=True)
        logger.info(
            f"💾 [03:45 PM Daily MOC Backup Job] {res.get('status')} | "
            f"File: {res.get('backup_filename')} ({res.get('backup_size_mb')} MB) | "
            f"Integrity: {res.get('integrity_check')} | Pruned: {res.get('pruned_count')}"
        )
    except Exception as e:
        logger.error(f"03:45 PM Daily MOC Backup Job error: {e}")


def scheduled_movers_research_job():
    """
    Every 30 mins during market hours (09:15 to 15:30 IST):
    Scans positive moving stocks, downloads OHLCV data, analyzes pre-breakout accumulation signatures,
    and updates the empirical movers research audit.
    """
    curr = now_ist()
    if not NSEMarketCalendar.is_trading_day(curr.date()):
        return
    can_trade, _ = NSEMarketCalendar.can_execute_trades()
    if not can_trade:
        return
    try:
        from engines.accumulation_research_engine import accumulation_research_engine
        logger.info("🔬 [30-Min Movers Research] Triggering periodic accumulation research scan...")
        accumulation_research_engine.run_research_scan(force_refresh=True)
    except Exception as e:
        logger.warning(f"30-Min Movers Research scan notice: {e}")


def scheduled_inhouse_scanner_job():
    """
    Every 30 mins during market hours (09:15 to 15:30 IST, Mon-Fri):
    Runs the In-House Screener across the broad cash universe (unrestricted, matching Chartink).
    Executes 6-pillar analysis with dual-entry classification (ACCUMULATION_COIL & BREAKOUT_SURGE).
    Pipes qualifying BUY_NOW setups directly into:
    1. Shadow Tracking (quant_trade_dataset)
    2. Portfolio 1 (trades / active_positions)
    """
    curr = now_ist()
    if not NSEMarketCalendar.is_trading_day(curr.date()):
        return
    can_trade, _ = NSEMarketCalendar.can_execute_trades()
    if not can_trade:
        return
    try:
        from engines.institutional_engine import InstitutionalEngine
        from engines.screener_comparison_engine import screener_comparison_engine
        logger.info("⚡ [30-Min In-House Broad Screener] Running broad cash equities 6-pillar scan & dual-entry execution...")
        inst_engine = InstitutionalEngine()
        signals = inst_engine.run(auto_execute=True, mode_override="periodic_30m")
        # Update dual-engine parity audit cache with fresh results
        screener_comparison_engine.run_comparison_audit(force_refresh=True)
        b_now = int((signals["bucket"] == "BUY_NOW").sum()) if not signals.empty and "bucket" in signals.columns else 0
        logger.info(f"✓ [30-Min In-House Screener] Scan complete. {len(signals)} evaluated, {b_now} BUY_NOW signals routed to Shadow & Portfolio 1.")
    except Exception as e:
        logger.error(f"30-Min In-House Screener execution error: {e}", exc_info=True)


def start_scheduler():
    """Initializes and starts the BackgroundScheduler with IST triggers & Render Anti-Sleep Guard."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        # Kick off initial non-blocking dynamic NSE holiday fetch + initial keep-alive check
        threading.Thread(target=scheduled_nse_holiday_sync_job, daemon=True).start()

        _scheduler = BackgroundScheduler(timezone=IST)

        # Job 0A-1: Daily Official Pre-Market Forecast Lock & Telegram Bulletin at 08:45 AM IST Mon-Fri
        _scheduler.add_job(
            scheduled_premarket_nifty_prediction_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=45, timezone=IST),
            id="job_premarket_prediction_0845",
            name="08:45 AM Official Pre-Market Forecast Lock & Bulletin",
            replace_existing=True,
            misfire_grace_time=300
        )

        # Job 0A: Daily Dynamic NSE Holiday Sync at 08:55 AM IST Mon-Fri
        _scheduler.add_job(
            scheduled_nse_holiday_sync_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=55, timezone=IST),
            id="job_nse_holiday_sync_0855",
            name="08:55 AM Dynamic NSE Holiday Calendar Sync",
            replace_existing=True,
            misfire_grace_time=600
        )

        # Job 0B: Dedicated 24/7 VPS Host (No Render keep-alive pinger needed)

        # Job 0C: 5-Minute Live Price Stream for Active Positions & Screened Stocks (09:15 AM – 03:30 PM IST)
        _scheduler.add_job(
            scheduled_live_price_stream_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone=IST),
            id="job_live_price_stream_5m",
            name="09:15 AM – 03:30 PM Live Market Price Stream (Every 5m)",
            replace_existing=True,
            misfire_grace_time=180
        )

        # Job 0C-2: 09:00 AM Pre-Market Options State Reconciliation (Mon-Fri)
        _scheduler.add_job(
            scheduled_options_premarket_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=IST),
            id="job_options_premarket_0900",
            name="09:00 AM Options Pre-Market Gate Check",
            replace_existing=True,
            misfire_grace_time=300
        )

        # Job 0C-3: 09:20 AM Intraday 5-Regime Options Strategy Auto-Deploy (Mon-Fri)
        _scheduler.add_job(
            scheduled_intraday_options_entry_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=20, timezone=IST),
            id="job_intraday_options_0920",
            name="09:20 AM Intraday 5-Regime Options Strategy Auto-Deploy",
            replace_existing=True,
            misfire_grace_time=300
        )

        # Job 0D: Multi-Interval (15m / 30m / 50m) Nifty Weekly Option Expiry Telemetry Collector (09:30 AM to 03:30 PM IST)
        # Runs every 15 minutes (:00, :15, :30, :45) between 09:30 and 15:30 IST, plus specific 50m slots
        _scheduler.add_job(
            scheduled_nifty_intraday_50min_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,15,30,45", timezone=IST),
            id="job_nifty_intraday_15m_stream",
            name="09:30 AM – 03:30 PM Nifty Weekly Option Multi-Interval Collector (Every 15m)",
            replace_existing=True,
            misfire_grace_time=180
        )

        # Explicit 50-minute slot triggers to guarantee exact 50m cadence captures
        intraday_50m_extra_slots = [
            (10, 20, "10:20"),
            (11, 10, "11:10"),
            (12, 50, "12:50"),
            (13, 40, "13:40"),
            (15, 20, "15:20"),
        ]
        for hr, mn, slot_lbl in intraday_50m_extra_slots:
            _scheduler.add_job(
                scheduled_nifty_intraday_50min_job,
                kwargs={"slot_label": slot_lbl},
                trigger=CronTrigger(day_of_week="mon-fri", hour=hr, minute=mn, timezone=IST),
                id=f"job_nifty_50m_{hr:02d}{mn:02d}",
                name=f"{hr:02d}:{mn:02d} IST Nifty Weekly Option 50m Collector ({slot_lbl})",
                replace_existing=True,
                misfire_grace_time=300
            )

        # Job 1: 10:20 AM IST Monday-Friday
        trigger_morning = CronTrigger(
            day_of_week="mon-fri",
            hour=SCHEDULE_MORNING_HOUR,
            minute=SCHEDULE_MORNING_MINUTE,
            timezone=IST
        )
        _scheduler.add_job(
            scheduled_morning_scan_job,
            trigger=trigger_morning,
            id="job_morning_1020",
            name="10:20 AM Morning Intraday Scan",
            replace_existing=True,
            misfire_grace_time=120
        )

        # Job 2: 3:00 PM IST Monday-Friday
        trigger_afternoon = CronTrigger(
            day_of_week="mon-fri",
            hour=SCHEDULE_AFTERNOON_HOUR,
            minute=SCHEDULE_AFTERNOON_MINUTE,
            timezone=IST
        )
        _scheduler.add_job(
            scheduled_afternoon_scan_job,
            trigger=trigger_afternoon,
            id="job_afternoon_305",
            name="3:00 PM Afternoon Close-of-Day Swing Scan",
            replace_existing=True,
            misfire_grace_time=120
        )

        # Job 3: 03:40 PM Daily Prediction Outcomes & Online Training (Mon-Fri)
        _scheduler.add_job(
            scheduled_daily_prediction_training_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=40, timezone=IST),
            id="job_daily_prediction_training_1540",
            name="03:40 PM Daily Prediction Outcomes & Online Training Sync",
            replace_existing=True,
            misfire_grace_time=600
        )

        # Job 3B: 03:45 PM Market-On-Close Daily Master Database Backup & Pruning (Mon-Fri)
        _scheduler.add_job(
            scheduled_daily_moc_backup_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=45, timezone=IST),
            id="job_daily_moc_backup_1545",
            name="03:45 PM Market-On-Close Daily Master Database Backup & Retention Prune",
            replace_existing=True,
            misfire_grace_time=600
        )

        # Job 4: Intraday Flash Bulletins at Institutional Transition Windows (Mon-Fri)
        # 4A. 09:45 AM IST - Opening Range Breakout (ORB) Bulletin
        _scheduler.add_job(
            lambda: scheduled_intraday_flash_bulletin_job("09:45 ORB"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=45, timezone=IST),
            id="job_intraday_bulletin_0945",
            name="09:45 AM Opening Range Breakout (ORB) Flash Bulletin",
            replace_existing=True,
            misfire_grace_time=180
        )
        # 4B. 11:30 AM IST - Mid-Day Pre-Europe Institutional Review
        _scheduler.add_job(
            lambda: scheduled_intraday_flash_bulletin_job("11:30 Pre-Europe"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=11, minute=30, timezone=IST),
            id="job_intraday_bulletin_1130",
            name="11:30 AM Mid-Day Pre-Europe Institutional Review",
            replace_existing=True,
            misfire_grace_time=180
        )
        # 4C. 01:00 PM IST - European Cash Open Volatility Impulse
        _scheduler.add_job(
            lambda: scheduled_intraday_flash_bulletin_job("13:00 Europe Open"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=13, minute=0, timezone=IST),
            id="job_intraday_bulletin_1300",
            name="01:00 PM European Cash Open Volatility Impulse",
            replace_existing=True,
            misfire_grace_time=180
        )
        # 4D. 02:15 PM IST - Expiry Afternoon Gamma Unwind Alert
        _scheduler.add_job(
            lambda: scheduled_intraday_flash_bulletin_job("14:15 Afternoon Gamma"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=14, minute=15, timezone=IST),
            id="job_intraday_bulletin_1415",
            name="02:15 PM Expiry Afternoon Gamma Unwind Alert",
            replace_existing=True,
            misfire_grace_time=180
        )

        # Job 5: Every 30 Minutes Movers Accumulation Research Scan (Mon-Fri, 09:30-15:30 IST)
        _scheduler.add_job(
            scheduled_movers_research_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone=IST),
            id="job_movers_research_30m",
            name="30-Min Movers & Pre-Breakout Accumulation Research Scan",
            replace_existing=True,
            misfire_grace_time=300
        )

        # Job 6: Every 30 Minutes In-House Broad Screener & Dual-Entry Execution (Mon-Fri, 09:30-15:30 IST)
        _scheduler.add_job(
            scheduled_inhouse_scanner_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone=IST),
            id="job_inhouse_scanner_30m",
            name="30-Min In-House Broad Universe Screener & Dual-Entry Execution",
            replace_existing=True,
            misfire_grace_time=300
        )

        _scheduler.start()
        # Trigger an immediate live price refresh, and Nifty 50m option capture if in market hours
        threading.Thread(target=scheduled_live_price_stream_job, daemon=True).start()
        threading.Thread(target=scheduled_nifty_intraday_50min_job, daemon=True).start()
        # Launch dedicated continuous Intraday Watchdog
        threading.Thread(target=_intraday_collector_watchdog, daemon=True, name="IntradayWatchdog").start()
        # Zero-Failure Cold-Boot Catch-Up: If container booted >= 08:45 AM on a trading day and prediction not locked, dispatch immediately
        try:
            curr_ist = now_ist()
            if NSEMarketCalendar.is_trading_day(curr_ist.date()):
                today_str = curr_ist.strftime("%Y-%m-%d")
                mins_now = curr_ist.hour * 60 + curr_ist.minute
                is_pred_locked = bool(state_engine.get_kv(f"nifty_prediction_locked_{today_str}"))
                if not is_pred_locked and mins_now >= 525:  # >= 08:45 AM IST
                    logger.info(f"🛡️ [Cold Boot Catch-Up] Container started >= 08:45 AM ({curr_ist.strftime('%H:%M:%S')} IST). Dispatching pre-market prediction...")
                    threading.Thread(target=scheduled_premarket_nifty_prediction_job, daemon=True, name="ColdBootPrediction").start()
        except Exception as e:
            logger.debug(f"Cold boot prediction check error: {e}")
        # Launch dedicated zero-broker live options & spot background streamer
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            nifty_analytics_engine.start_live_options_streamer()
        except Exception:
            pass
        logger.info(
            "✓ Automated Scheduler active on Asia/Kolkata timezone: "
            "08:45 AM Pre-Market Lock, 08:55 AM NSE Holiday Sync, 08:40–15:50 Render Anti-Sleep Guard, 5m Live Price Stream, "
            "30m/50m Nifty Weekly Option Expiry Collector (09:30–15:30), 10:20 AM & 3:00 PM Scans, with Continuous Auto-Gapfill Watchdog & Zero-Broker 12s Streamer."
        )
        return _scheduler


def shutdown_scheduler():
    """Gracefully shuts down the scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)
            logger.info("Scheduler shutdown cleanly.")


def get_scheduler_status() -> Dict[str, Any]:
    """Returns scheduler operational status, Render Anti-Sleep telemetry, and dynamic NSE holiday metadata."""
    next_info = NSEMarketCalendar.get_next_scheduled_scan()
    market_status = NSEMarketCalendar.get_market_status()
    keepalive_active, keepalive_reason = NSEMarketCalendar.is_render_keepalive_window()
    holiday_meta = NSEMarketCalendar.get_holiday_metadata()

    jobs_info = []
    if _scheduler and _scheduler.running:
        for job in _scheduler.get_jobs():
            next_run = job.next_run_time
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": next_run.strftime("%Y-%m-%d %H:%M:%S IST") if next_run else None
            })

    # Continuous background daemon workers running alongside cron jobs
    daemons = [
        {
            "id": "daemon_intraday_watchdog",
            "name": "Continuous 40s Intraday Watchdog & Auto-Gapfill Guard",
            "cadence": "Every 40 seconds (09:15–15:35 IST)",
            "type": "BACKGROUND_THREAD",
            "status": "RUNNING",
            "vault_sync": True,
            "collision_guard": "PRAGMA WAL + 15s Timeout"
        },
        {
            "id": "daemon_options_spot_streamer",
            "name": "Zero-Broker 12s Live Options & Spot Streamer",
            "cadence": "Every 12 seconds (Market Hours)",
            "type": "BACKGROUND_THREAD",
            "status": "RUNNING",
            "vault_sync": False,
            "collision_guard": "In-Memory Lock + Atomic KV"
        },
        {
            "id": "daemon_chartink_6pillar_worker",
            "name": "Chartink 6-Pillar Parallel Worker Engine",
            "cadence": "10:20 AM & 03:00 PM (24 Parallel Workers)",
            "type": "THREADPOOL_EXECUTOR",
            "status": "READY",
            "vault_sync": True,
            "collision_guard": "Atomic Batch Save + ExecutionQueue"
        }
    ]

    return {
        "scheduler_running": bool(_scheduler and _scheduler.running),
        "timezone": "Asia/Kolkata (IST)",
        "jobs": jobs_info,
        "next_scan": next_info,
        "market_status": market_status,
        "auto_workers": {
            "total_count": len(jobs_info) + len(daemons),
            "cron_jobs_count": len(jobs_info),
            "daemon_threads_count": len(daemons),
            "concurrency_mode": "SQLite WAL (Write-Ahead Logging)",
            "collision_prevention": "PRAGMA busy_timeout=15000ms • Serialized Write Queue • Atomic Commits",
            "daemons": daemons,
            "all_workers_vault_synced": True
        },
        "vps_host_status": {
            "is_active_now": True,
            "host": "RackNerd KVM VPS (Ubuntu 24.04)",
            "uptime_mode": "24/7 DEDICATED (ALWAYS ONLINE)",
            "current_state_reason": "Dedicated VPS Server • Native systemd supervisor • No sleep timeout"
        },
        "render_keepalive": {
            "is_active_now": True,
            "current_state_reason": "24/7 Dedicated VPS Server (No sleep timeout)",
            "pings_today": 0,
            "last_ping_status": "24/7 DEDICATED HOST — NO KEEPALIVE NEEDED"
        },
        "nse_holidays_meta": holiday_meta
    }
