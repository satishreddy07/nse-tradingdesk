"""
========================================================================================
FastAPI REST API & Web Application
========================================================================================
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.config import P1_BASE_CAPITAL, P2_BASE_CAPITAL, P2_MAX_SLOTS
from engines.institutional_engine import (
    InstitutionalEngine, get_institutional_status, PORTFOLIO_TYPE as P1_TYPE
)
from engines.nifty500_engine import (
    run_nifty500_pipeline_async, get_nifty500_status, get_live_compounded_equity,
    PORTFOLIO_TYPE as P2_TYPE
)
from execution.state_engine import state_engine
from scheduler.cron_scheduler import (
    start_scheduler, shutdown_scheduler, get_scheduler_status
)

logger = logging.getLogger("WebAPI")

app = FastAPI(
    title="Quant Trading Desk",
    description="Institutional NSE Equities & Nifty 500 Momentum Swing Desk",
    version="2.0.0"
)

# Template setup
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def on_startup():
    logger.info("Initializing Quant Trading Desk services...")
    start_scheduler()


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down Quant Trading Desk services...")
    shutdown_scheduler()


# --------------------------------------------------------------------------
# HEALTH & FRONTEND ROUTES
# --------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": format_ist()}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Quant Trading Desk</h1><p>index.html not found</p>")


# --------------------------------------------------------------------------
# SYSTEM STATUS & CALENDAR
# --------------------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    scheduler_info = get_scheduler_status()
    curr_time = format_ist()
    p1_status = get_institutional_status()
    p2_status = get_nifty500_status()

    return {
        "status": "ONLINE",
        "current_time_ist": curr_time,
        "market_status": scheduler_info["market_status"],
        "next_scan": scheduler_info["next_scan"],
        "scheduler_running": scheduler_info["scheduler_running"],
        "jobs": scheduler_info["jobs"],
        "portfolio_1_quick": {
            "universe_size": p1_status.get("universe_size", 0),
            "buy_now_count": p1_status.get("buy_now_count", 0),
            "active_positions": p1_status.get("active_count", 0),
            "status": p1_status.get("status", "IDLE")
        },
        "portfolio_2_quick": {
            "macro_regime": p2_status.get("macro_regime", "UNKNOWN"),
            "compounded_equity": p2_status.get("compounded_equity", P2_BASE_CAPITAL),
            "active_slots": p2_status.get("active_slots", 0),
            "available_slots": p2_status.get("available_slots", P2_MAX_SLOTS),
            "status": p2_status.get("status", "IDLE")
        }
    }


@app.get("/api/calendar")
def get_market_calendar():
    return {
        "current_date": now_ist().strftime("%Y-%m-%d"),
        "is_trading_day": NSEMarketCalendar.is_trading_day(),
        "market_status": NSEMarketCalendar.get_market_status(),
        "holidays": NSEMarketCalendar.NSE_HOLIDAYS
    }


# --------------------------------------------------------------------------
# PORTFOLIO DATA ROUTES
# --------------------------------------------------------------------------
@app.get("/api/portfolio/{portfolio_type}/summary")
def get_portfolio_summary(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    stats = state_engine.get_portfolio_stats(portfolio_type=p_type)

    if p_type == P2_TYPE:
        curr_eq, tot_eq, active_count = get_live_compounded_equity()
        p2_state = get_nifty500_status()
        return {
            "portfolio_type": p_type,
            "title": "Nifty 500 Momentum Compounded Engine",
            "base_capital": P2_BASE_CAPITAL,
            "current_equity": round(curr_eq, 2),
            "dynamic_compounding_active": True,
            "active_slots": active_count,
            "max_slots": P2_MAX_SLOTS,
            "available_slots": max(0, P2_MAX_SLOTS - active_count),
            "slot_cap_inr": round(curr_eq / P2_MAX_SLOTS, 2),
            "risk_per_trade_inr": round(curr_eq * 0.015, 2),
            "macro_regime": p2_state.get("macro_regime", "UNKNOWN"),
            "bench_ltp": p2_state.get("bench_ltp"),
            "bench_ema50": p2_state.get("bench_ema50"),
            "stats": stats
        }
    else:
        p1_state = get_institutional_status()
        return {
            "portfolio_type": p_type,
            "title": "Institutional 6-Pillar Engine",
            "base_capital": P1_BASE_CAPITAL,
            "current_equity": round(P1_BASE_CAPITAL + stats.get("total_realized_pnl", 0.0) + stats.get("unrealized_pnl", 0.0), 2),
            "active_positions": stats.get("active_count", 0),
            "open_risk": stats.get("open_risk", 0.0),
            "universe_size": p1_state.get("universe_size", 0),
            "buy_now_count": p1_state.get("buy_now_count", 0),
            "pre_breakout_count": p1_state.get("pre_breakout_count", 0),
            "wait_pullback_count": p1_state.get("wait_pullback_count", 0),
            "stats": stats
        }


@app.get("/api/portfolio/{portfolio_type}/positions")
def get_portfolio_positions(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    positions = state_engine.get_active_positions(portfolio_type=p_type)

    for p in positions:
        ep = float(p["entry_price"])
        cp = float(p.get("current_price") or ep)
        sl = float(p["stop_loss"])
        qty = int(p["quantity"])
        p["unrealized_pnl"] = round((cp - ep) * qty, 2)
        p["return_pct"] = round(((cp - ep) / ep) * 100.0, 2)
        p["stop_distance_pct"] = round(((cp - sl) / cp) * 100.0, 2)
        r_unit = max(0.01, ep - float(p.get("original_stop_loss") or sl))
        p["current_r"] = round((cp - ep) / r_unit, 2)

    return positions


@app.get("/api/portfolio/{portfolio_type}/history")
def get_portfolio_history(portfolio_type: str, limit: int = 200):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_trade_history(portfolio_type=p_type, limit=limit)


@app.get("/api/portfolio/{portfolio_type}/signals")
def get_portfolio_signals(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_latest_signals(portfolio_type=p_type)


# --------------------------------------------------------------------------
# SCAN TRIGGER & EXECUTION
# --------------------------------------------------------------------------
class TriggerScanRequest(BaseModel):
    scan_type: str = "ALL"  # 'MORNING_1020AM', 'AFTERNOON_305PM', 'INSTITUTIONAL', 'NIFTY500', 'ALL'
    auto_execute: bool = False


@app.get("/sw.js")
def service_worker_cleanup():
    from fastapi.responses import Response
    js = "self.addEventListener('install', () => self.skipWaiting()); self.addEventListener('activate', () => self.registration.unregister());"
    return Response(content=js, media_type="application/javascript")


@app.get("/api/scan/progress")
def get_scan_progress():
    return state_engine.get_scan_progress()


@app.post("/api/scan/trigger")
def trigger_scan(req: TriggerScanRequest):
    import threading
    from engines.nifty500_engine import run_nifty500_pipeline_sync

    s_type = req.scan_type.upper()
    curr_prog = state_engine.get_scan_progress()
    if curr_prog.get("is_running"):
        return {
            "status": "ALREADY_RUNNING",
            "message": f"Scan is currently in progress: {curr_prog.get('stage')}",
            "progress": curr_prog
        }

    state_engine.start_scan_progress(
        scan_type=s_type,
        initial_stage=f"Initiating live {s_type} scan..."
    )

    def run_scan_worker():
        logger.info(f"⚡ Live scan started: {s_type} (Auto-Execute: {req.auto_execute})")
        try:
            if s_type in ("MORNING_1020AM", "INSTITUTIONAL", "ALL"):
                inst = InstitutionalEngine()
                mode = "morning" if "MORNING" in s_type else "afternoon"
                inst.run(auto_execute=req.auto_execute, mode_override=mode)

            if s_type in ("AFTERNOON_305PM", "NIFTY500", "ALL"):
                run_nifty500_pipeline_sync(mode="MANUAL_TRIGGER", auto_execute=req.auto_execute)
        except Exception as e:
            logger.error(f"Error during live scan {s_type}: {e}", exc_info=True)
            state_engine.finish_scan_progress(f"Scan encountered error: {str(e)}")

    t = threading.Thread(target=run_scan_worker, daemon=True)
    t.start()

    return {
        "status": "STARTED",
        "message": f"Live scan '{s_type}' started! Fetching latest stocks...",
        "auto_execute": req.auto_execute,
        "timestamp": format_ist(),
        "progress": state_engine.get_scan_progress()
    }


@app.get("/api/scan/logs")
def get_scan_logs(limit: int = 50):
    return state_engine.get_recent_scan_logs(limit=limit)



# --------------------------------------------------------------------------
# MANUAL TRADE OPERATIONS
# --------------------------------------------------------------------------
class ClosePositionRequest(BaseModel):
    symbol: str
    exit_price: Optional[float] = None
    exit_reason: str = "MANUAL_DISCRETIONARY"
    notes: str = ""


@app.post("/api/position/close")
def close_active_position(req: ClosePositionRequest):
    sym = req.symbol.strip().upper()
    active_rows = state_engine.get_active_positions()
    matched = [p for p in active_rows if p["symbol"] == sym]

    if not matched:
        raise HTTPException(status_code=404, detail=f"Active position for {sym} not found.")

    pos = matched[0]
    exit_px = req.exit_price or float(pos.get("current_price") or pos["entry_price"])

    res = state_engine.close_position(
        symbol=sym,
        exit_price=exit_px,
        exit_reason=req.exit_reason,
        notes=req.notes
    )
    if not res:
        raise HTTPException(status_code=500, detail="Failed to close position.")

    return {"status": "SUCCESS", "closed_trade": res}


# --------------------------------------------------------------------------
# SAMPLE SEED / REHYDRATION ENDPOINT
# --------------------------------------------------------------------------
@app.post("/api/seed/demo-data")
def seed_demo_data():
    """Generates initial sample data for instant exploration."""
    now_str = format_ist()

    # Portfolio 1 sample closed trade
    state_engine.close_position("TRENT", exit_price=5400.0, exit_reason="T2_FULL_TARGET", notes="Initial demo trade")
    
    # Portfolio 1 active position
    state_engine.open_position(
        symbol="DIXON",
        entry_price=12400.0,
        quantity=8,
        stop_loss=11900.0,
        target=13150.0,
        target_2=13900.0,
        notes="6-Pillar Score 88.5 | Tier A",
        sl_reason="INITIAL_SL",
        sector="CAPGOODS",
        portfolio_type="INSTITUTIONAL_6PILLAR",
        entry_reason="Minervini Stage 2 + Pocket Pivot breakout"
    )

    # Portfolio 2 active position
    state_engine.open_position(
        symbol="BEL",
        entry_price=312.0,
        quantity=32,
        stop_loss=296.0,
        target=344.0,
        target_2=360.0,
        notes="7-Point Matrix Qualified | RS Leader",
        sl_reason="INITIAL_SL",
        sector="NIFTY500",
        portfolio_type="NIFTY500_MOMENTUM",
        entry_reason="EMA20 pullback bounce + Volume expansion"
    )

    return {"status": "SUCCESS", "message": "Demo data populated."}
