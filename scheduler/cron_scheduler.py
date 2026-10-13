"""
========================================================================================
Automated Quant Cron Scheduler & Render Market-Hours Anti-Sleep Engine
========================================================================================
Runs on every NSE Indian Equities trading day (Monday-Friday, excluding dynamic NSE holidays):
- 08:55 AM IST: Daily Dynamic NSE Holiday Sync from nseindia.com
- 09:00 AM – 03:50 PM IST (Every 4 mins): Strict Render Anti-Sleep Keep-Alive Guard
- 10:20 AM IST: Morning Intraday Momentum & Entry Scan
- 03:05 PM IST: Afternoon Close-of-Day Swing Scan & MOC Execution

Strictly stops keep-alive pings outside 09:00 AM – 03:50 PM IST, on weekends, and on
dynamically fetched NSE holidays so the Render server can sleep outside market hours.
========================================================================================
"""

import logging
import os
import threading
from typing import Dict, Any, Optional
import requests

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.data_quality import IST, now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.config import (
    SCHEDULE_MORNING_HOUR, SCHEDULE_MORNING_MINUTE,
    SCHEDULE_AFTERNOON_HOUR, SCHEDULE_AFTERNOON_MINUTE
)
from engines.institutional_engine import InstitutionalEngine
from engines.nifty500_engine import run_nifty500_pipeline_sync
from execution.state_engine import state_engine

logger = logging.getLogger("Scheduler")

_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()
_detected_external_url: Optional[str] = None

_keepalive_telemetry: Dict[str, Any] = {
    "window_rule": "09:00 AM – 03:50 PM IST (Mon-Fri, Excl. NSE Holidays)",
    "interval_minutes": 4,
    "last_ping_ist": None,
    "last_ping_status": "IDLE",
    "pings_today": 0,
    "pings_date": None,
    "target_url": None
}


def register_external_request_origin(scheme: str, host: str):
    """
    Dynamically records the public external URL (e.g. https://quant-trading-desk.onrender.com)
    from incoming HTTP requests when RENDER_EXTERNAL_URL is not explicitly set.
    """
    global _detected_external_url
    if not host:
        return
    host_clean = host.strip().lower()
    if any(local_h in host_clean for local_h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return
    candidate = f"{scheme or 'https'}://{host.strip()}"
    if _detected_external_url != candidate:
        _detected_external_url = candidate
        logger.info(f"🌐 Auto-detected public service URL for Render Anti-Sleep Guard: {_detected_external_url}")


def get_keepalive_target_url() -> str:
    """
    Resolves the external URL to ping so Render's edge load balancer resets its 15-min idle timer.
    Priority:
    1. RENDER_EXTERNAL_URL (Automatically injected by Render on all web services)
    2. PUBLIC_APP_URL (Optional user env override)
    3. Auto-detected external Host header from live traffic
    4. Localhost fallback (for local verification)
    """
    env_url = (os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_APP_URL") or "").strip().rstrip("/")
    if env_url:
        return f"{env_url}/health?source=render_keepalive"
    if _detected_external_url:
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


def scheduled_morning_scan_job():
    """
    10:20 AM IST Trading Day Job:
    - Verifies NSE trading day calendar (using dynamically synced NSE holidays).
    - Runs Institutional 6-Pillar Engine (Morning Mode).
    - Reconciles active positions across both portfolios.
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

    try:
        inst_engine = InstitutionalEngine()
        signals = inst_engine.run(auto_execute=True, mode_override="morning")
        logger.info(f"✓ 10:20 AM Morning Institutional Scan completed. {len(signals)} stocks evaluated.")
    except Exception as e:
        logger.error(f"Error in 10:20 AM Institutional scan: {e}", exc_info=True)


def scheduled_afternoon_scan_job():
    """
    3:05 PM IST Trading Day Job:
    - Verifies NSE trading day calendar (using dynamically synced NSE holidays).
    - Runs Institutional 6-Pillar Engine (Afternoon Mode with Intraday Confirmation).
    - Runs Nifty 500 Momentum Compounded Engine (4-Tier Exit Hierarchy + 7-Point Entry Matrix).
    """
    current_dt = now_ist()
    logger.info(f"⏰ [3:05 PM Trigger] Initiating afternoon MOC scan sequence at {format_ist(current_dt)}...")

    NSEMarketCalendar.sync_holidays_from_nse(force=False)
    if not NSEMarketCalendar.is_trading_day(current_dt.date()):
        reason = NSEMarketCalendar.get_holiday_reason(current_dt.date())
        logger.info(f"⏸️ Today is not an active trading day ({reason}). Afternoon scan skipped.")
        state_engine.log_scan_run(
            scan_type="SCHEDULED_305PM",
            portfolio_type="BOTH",
            status="SKIPPED_HOLIDAY",
            execution_time_sec=0.0,
            message=f"Scan skipped: {reason}"
        )
        return

    # 1. Run Institutional 6-Pillar Engine (Afternoon Mode with Intraday Confirmation)
    try:
        inst_engine = InstitutionalEngine()
        signals_p1 = inst_engine.run(auto_execute=True, mode_override="afternoon")
        logger.info(f"✓ 3:05 PM Afternoon Institutional Scan completed. {len(signals_p1)} stocks evaluated.")
    except Exception as e:
        logger.error(f"Error in 3:05 PM Institutional scan: {e}", exc_info=True)

    # 2. Run Nifty 500 Momentum Compounded Engine (7-Point Matrix + 4-Tier Exits)
    try:
        res_p2 = run_nifty500_pipeline_sync(mode="SCHEDULED_305PM", auto_execute=True)
        logger.info(f"✓ 3:05 PM Nifty 500 Momentum Scan completed: {res_p2.get('orders_executed', 0)} orders placed.")
    except Exception as e:
        logger.error(f"Error in 3:05 PM Nifty 500 scan: {e}", exc_info=True)


def start_scheduler():
    """Initializes and starts the BackgroundScheduler with IST triggers & Render Anti-Sleep Guard."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        # Kick off initial non-blocking dynamic NSE holiday fetch + initial keep-alive check
        threading.Thread(target=scheduled_nse_holiday_sync_job, daemon=True).start()

        _scheduler = BackgroundScheduler(timezone=IST)

        # Job 0A: Daily Dynamic NSE Holiday Sync at 08:55 AM IST Mon-Fri
        _scheduler.add_job(
            scheduled_nse_holiday_sync_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour=8, minute=55, timezone=IST),
            id="job_nse_holiday_sync_0855",
            name="08:55 AM Dynamic NSE Holiday Calendar Sync",
            replace_existing=True,
            misfire_grace_time=600
        )

        # Job 0B: Strict Render Anti-Sleep Keep-Alive Pinger (Every 4 mins, 09:00 AM – 03:50 PM IST Mon-Fri)
        _scheduler.add_job(
            scheduled_render_keepalive_job,
            trigger=CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/4", timezone=IST),
            id="job_render_keepalive",
            name="09:00 AM – 03:50 PM Render Anti-Sleep Guard (Every 4m)",
            replace_existing=True,
            misfire_grace_time=120
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
            misfire_grace_time=300
        )

        # Job 2: 3:05 PM IST Monday-Friday
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
            name="3:05 PM Afternoon Close-of-Day Swing Scan",
            replace_existing=True,
            misfire_grace_time=300
        )

        _scheduler.start()
        # Trigger an immediate keep-alive check if server booted inside 09:00-15:50 IST
        threading.Thread(target=scheduled_render_keepalive_job, daemon=True).start()
        logger.info(
            "✓ Automated Scheduler active on Asia/Kolkata timezone: "
            "08:55 AM NSE Holiday Sync, 09:00–15:50 Render Anti-Sleep Guard, 10:20 AM & 3:05 PM Scans."
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

    return {
        "scheduler_running": bool(_scheduler and _scheduler.running),
        "timezone": "Asia/Kolkata (IST)",
        "jobs": jobs_info,
        "next_scan": next_info,
        "market_status": market_status,
        "render_keepalive": {
            **_keepalive_telemetry,
            "is_active_now": keepalive_active,
            "current_state_reason": keepalive_reason,
            "target_url": get_keepalive_target_url()
        },
        "nse_holidays_meta": holiday_meta
    }
