"""
========================================================================================
Automated Quant Cron Scheduler
========================================================================================
Runs on every NSE Indian Equities trading day:
- 10:20 AM IST: Morning Intraday Momentum & Entry Scan
- 03:05 PM IST: Afternoon Close-of-Day Swing Scan & MOC Execution

Excludes weekends (Sat, Sun) and official NSE Exchange Holidays.
Includes manual on-demand triggers and status monitoring.
========================================================================================
"""

import logging
import threading
from typing import Dict, Any, Optional

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


def scheduled_morning_scan_job():
    """
    10:20 AM IST Trading Day Job:
    - Verifies NSE trading day calendar.
    - Runs Institutional 6-Pillar Engine (Morning Mode).
    - Reconciles active positions across both portfolios.
    """
    current_dt = now_ist()
    logger.info(f"⏰ [10:20 AM Trigger] Initiating morning scan sequence at {format_ist(current_dt)}...")

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
        # Run Institutional 6-Pillar Engine (Morning Mode with Auto-Execution)
        inst_engine = InstitutionalEngine()
        signals = inst_engine.run(auto_execute=True, mode_override="morning")
        logger.info(f"✓ 10:20 AM Morning Institutional Scan completed. {len(signals)} stocks evaluated.")
    except Exception as e:
        logger.error(f"Error in 10:20 AM Institutional scan: {e}", exc_info=True)


def scheduled_afternoon_scan_job():
    """
    3:05 PM IST Trading Day Job:
    - Verifies NSE trading day calendar.
    - Runs Institutional 6-Pillar Engine (Afternoon Mode with Intraday Confirmation).
    - Runs Nifty 500 Momentum Compounded Engine (4-Tier Exit Hierarchy + 7-Point Entry Matrix).
    """
    current_dt = now_ist()
    logger.info(f"⏰ [3:05 PM Trigger] Initiating afternoon MOC scan sequence at {format_ist(current_dt)}...")

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
    """Initializes and starts the BackgroundScheduler with IST triggers."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        _scheduler = BackgroundScheduler(timezone=IST)

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
        logger.info(f"✓ Automated Scheduler active on Asia/Kolkata timezone: 10:20 AM & 3:05 PM Mon-Fri.")
        return _scheduler


def shutdown_scheduler():
    """Gracefully shuts down the scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)
            logger.info("Scheduler shutdown cleanly.")


def get_scheduler_status() -> Dict[str, Any]:
    """Returns scheduler operational status and next job run times."""
    next_info = NSEMarketCalendar.get_next_scheduled_scan()
    market_status = NSEMarketCalendar.get_market_status()

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
        "market_status": market_status
    }
