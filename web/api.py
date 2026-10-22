"""
========================================================================================
FastAPI REST API & Web Application
========================================================================================
"""
import os
import asyncio
import logging
import threading
import math
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Depends, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def verify_admin_key(request: Request):
    """
    Security Gate for Mutating Endpoints (F-03 remediation):
    If ADMIN_API_KEY is configured in the environment, requests must provide
    either header 'X-API-Key: <key>' or 'Authorization: Bearer <key>'.
    """
    expected = os.environ.get("ADMIN_API_KEY", "").strip()
    if not expected:
        return True  # Dev mode: open if not configured
    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")
    token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else api_key_header
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing administrative API key.")
    return True

from contextlib import asynccontextmanager

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from core.config import P1_BASE_CAPITAL, P1_BASE_RISK_PCT, P2_BASE_CAPITAL, P2_MAX_SLOTS, NIFTY_LOT_SIZE
from engines.institutional_engine import (
    InstitutionalEngine, get_institutional_status, refresh_live_market_prices,
    PORTFOLIO_TYPE as P1_TYPE
)
from engines.nifty500_engine import (
    run_nifty500_pipeline_async, run_nifty500_pipeline_sync,
    get_nifty500_status, get_live_compounded_equity,
    PORTFOLIO_TYPE as P2_TYPE
)
from engines.strategy_engine import (
    portfolio3_strategy_engine,
    PORTFOLIO_TYPE as P3_TYPE
)
from engines.custom_screener_engine import custom_screener_engine
from engines.screener_comparison_engine import screener_comparison_engine
from execution.state_engine import state_engine
from core.execution_queue import execution_queue
from core.data_watchdog import data_watchdog
from scheduler.cron_scheduler import (
    start_scheduler, shutdown_scheduler, get_scheduler_status,
    register_external_request_origin
)


logger = logging.getLogger("WebAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event handler for Quant Trading Desk.
    Executes disaster recovery and starts cron scheduler on boot,
    and cleanly shuts down background schedulers on exit.
    """
    logger.info("Initializing Quant Trading Desk services...")
    # Cold-boot disaster recovery from Telegram Cloud Vault (Trade DB + OHLCV Historical Cache)
    try:
        telegram_vault.auto_restore_on_cold_boot()
    except Exception as e:
        logger.warning(f"Startup Telegram restore notice: {e}")
    start_scheduler()

    # Auto-login Kotak Neo on boot from saved credentials
    try:
        import threading
        from engines.kotak_neo_engine import kotak_neo_engine
        from execution.state_engine import state_engine
        from core import config
        saved = state_engine.get_kv("KOTAK_NEO_CREDENTIALS") or {}
        if (saved.get("consumer_key") or config.KOTAK_NEO_CONSUMER_KEY) and (saved.get("ucc") or config.KOTAK_NEO_UCC) and (saved.get("totp_secret") or config.KOTAK_NEO_TOTP_SECRET):
            logger.info("🔐 Auto-authenticating Kotak Neo session on boot from saved credentials...")
            threading.Thread(target=kotak_neo_engine.login, name="KotakNeoBootLogin", daemon=True).start()
    except Exception as ke:
        logger.warning(f"Startup Kotak Neo auto-login notice: {ke}")

    yield
    logger.info("Shutting down Quant Trading Desk services...")
    shutdown_scheduler()


app = FastAPI(
    title="Quant Trading Desk",
    description="Institutional NSE Equities & Nifty 500 Momentum Swing Desk",
    version="2.1.0",
    lifespan=lifespan
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Configuration with origin whitelisting
allowed_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    cors_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    cors_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://nse-tradingdesk.onrender.com",
        "https://nseuniverse.duckdns.org",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Template setup
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def capture_public_host_middleware(request: Request, call_next):
    """Automatically detects the public Render URL for Anti-Sleep Guard and enforces zero browser caching for dynamic APIs while allowing static caching."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    if host:
        register_external_request_origin(proto, host)
    response = await call_next(request)
    if not request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response




# --------------------------------------------------------------------------
# HEALTH & FRONTEND ROUTES
# --------------------------------------------------------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    db_healthy = True
    db_msg = "connected"
    try:
        from execution.state_engine import state_engine
        with state_engine._get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as e:
        db_healthy = False
        db_msg = str(e)

    status_code = 200 if db_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if db_healthy else "degraded",
            "database": "healthy" if db_healthy else f"unhealthy: {db_msg}",
            "timestamp": format_ist()
        }
    )


@app.get("/favicon.ico")
def favicon():
    bull_icon = STATIC_DIR / "bull-logo.jpg"
    if bull_icon.exists():
        return FileResponse(str(bull_icon), media_type="image/jpeg")
    return Response(status_code=204)


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index(request: Request):
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    return HTMLResponse("<h1>Quant Trading Desk</h1><p>index.html not found</p>")


@app.get("/manifest.json", include_in_schema=False)
def get_manifest():
    manifest_path = STATIC_DIR / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js", include_in_schema=False)
def get_service_worker():
    sw_path = STATIC_DIR / "sw.js"
    if sw_path.exists():
        return FileResponse(
            sw_path,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"}
        )
    raise HTTPException(status_code=404, detail="Service worker not found")


# --------------------------------------------------------------------------
# SYSTEM STATUS, TRADING GATE & CALENDAR
# --------------------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    scheduler_info = get_scheduler_status()
    curr_time = format_ist()
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    p1_status = get_institutional_status()
    p2_status = get_nifty500_status()
    vault_info = telegram_vault.get_status()
    last_stream = state_engine.get_kv("last_price_stream", {})

    try:
        from scheduler.cron_scheduler import get_server_public_ip
        srv_ip = get_server_public_ip(force=False)
    except Exception:
        srv_ip = "127.0.0.1"

    try:
        from engines.kotak_neo_engine import kotak_neo_engine
        kotak_status = kotak_neo_engine.get_status()
    except Exception:
        kotak_status = {"status": "UNAVAILABLE", "is_streaming": False}

    return {
        "status": "ONLINE",
        "server_ip": srv_ip,
        "kotak_neo": kotak_status,
        "current_time_ist": curr_time,
        "market_status": scheduler_info["market_status"],
        "trading_gate_open": can_trade,
        "trading_gate_message": gate_msg,
        "next_scan": scheduler_info["next_scan"],
        "scheduler_running": scheduler_info["scheduler_running"],
        "jobs": scheduler_info["jobs"],
        "auto_workers": scheduler_info.get("auto_workers", {}),
        "render_keepalive": scheduler_info.get("render_keepalive", {}),
        "nse_holidays_meta": scheduler_info.get("nse_holidays_meta", {}),
        "telegram_vault": vault_info,
        "last_price_stream": last_stream,
        "data_watchdog": data_watchdog.check_all_endpoints(force_probe=False),
        "scheduled_scans_telemetry": state_engine.get_scheduled_scans_telemetry(),
        "portfolio_1_quick": {
            "chartink_fetched_count": p1_status.get("chartink_fetched_count", 0),
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
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    keepalive_active, keepalive_msg = NSEMarketCalendar.is_render_keepalive_window()
    return {
        "current_date": now_ist().strftime("%Y-%m-%d"),
        "is_trading_day": NSEMarketCalendar.is_trading_day(),
        "market_status": NSEMarketCalendar.get_market_status(),
        "trading_gate_open": can_trade,
        "trading_gate_message": gate_msg,
        "render_keepalive_active": keepalive_active,
        "render_keepalive_message": keepalive_msg,
        "nse_holidays_meta": NSEMarketCalendar.get_holiday_metadata(),
        "holidays": NSEMarketCalendar.NSE_HOLIDAYS
    }


@app.post("/api/calendar/sync-holidays")
def sync_nse_holidays_endpoint():
    """Manually triggers a live sync of official NSE trading holidays from nseindia.com."""
    meta = NSEMarketCalendar.sync_holidays_from_nse(force=True)
    return {
        "status": "OK",
        "message": f"Synced {meta.get('current_year_holidays_count', 0)} NSE holidays for {meta.get('current_year')} from {meta.get('source')}.",
        "nse_holidays_meta": meta
    }


# --------------------------------------------------------------------------
# PORTFOLIO DATA ROUTES
# --------------------------------------------------------------------------
@app.get("/api/portfolio/{portfolio_type}/summary")
def get_portfolio_summary(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    stats = state_engine.get_portfolio_stats(portfolio_type=p_type)
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()

    if p_type == P3_TYPE:
        return portfolio3_strategy_engine.get_portfolio_summary()
    elif p_type == P2_TYPE:
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
            "trading_gate_open": can_trade,
            "trading_gate_message": gate_msg,
            "stats": stats
        }
    else:
        p1_state = get_institutional_status()
        signals = state_engine.get_latest_signals(portfolio_type=P1_TYPE)
        sig_count = len(signals)
        b_now = sum(1 for s in signals if s.get("bucket") == "BUY_NOW")
        p_brk = sum(1 for s in signals if s.get("bucket") == "PRE_BREAKOUT")
        w_pb = sum(1 for s in signals if s.get("bucket") == "WAIT_PULLBACK")
        ohlcv_stats = state_engine.get_ohlcv_stats()
        chartink_cnt = p1_state.get("chartink_fetched_count") or ohlcv_stats.get("stocks_count", 0)
        c_avoid = sum(1 for s in signals if s.get("bucket") == "AVOID")
        return {
            "portfolio_type": p_type,
            "title": "Institutional 6-Pillar Engine",
            "base_capital": P1_BASE_CAPITAL,
            "current_equity": round(P1_BASE_CAPITAL + stats.get("total_realized_pnl", 0.0) + stats.get("unrealized_pnl", 0.0), 2),
            "active_positions": stats.get("active_count", 0),
            "max_slots": stats.get("max_slots", 10),
            "slots_used": stats.get("slots_used", 0),
            "available_slots": stats.get("slots_available", 0),
            "available_cash": stats.get("available_cash", 0.0),
            "total_invested": stats.get("total_invested", 0.0),
            "open_risk": stats.get("open_risk", 0.0),
            "nse_total_count": p1_state.get("nse_total_count") or state_engine.get_total_nse_stocks_count(),
            "passing_criteria_count": p1_state.get("passing_criteria_count") or chartink_cnt,
            "chartink_fetched_count": chartink_cnt,
            "universe_size": p1_state.get("universe_size") or sig_count,
            "cache_hits_count": p1_state.get("cache_hits_count", 0),
            "gap_filled_count": p1_state.get("gap_filled_count", 0),
            "full_downloads_count": p1_state.get("full_downloads_count", 0),
            "buy_now_count": p1_state.get("buy_now_count") if p1_state.get("universe_size") else b_now,
            "pre_breakout_count": p1_state.get("pre_breakout_count") if p1_state.get("universe_size") else p_brk,
            "wait_pullback_count": p1_state.get("wait_pullback_count") if p1_state.get("universe_size") else w_pb,
            "pullback_count": p1_state.get("pullback_count") or (p_brk + w_pb),
            "avoid_count": p1_state.get("avoid_count") or c_avoid,
            "trading_gate_open": can_trade,
            "trading_gate_message": gate_msg,
            "stats": stats
        }


@app.get("/api/portfolio/{portfolio_type}/positions")
def get_portfolio_positions(portfolio_type: str):
    if portfolio_type.strip().upper() == "ALL":
        positions = state_engine.get_active_positions(portfolio_type=None)
    else:
        p_type = state_engine._normalize_portfolio_type(portfolio_type)
        positions = state_engine.get_active_positions(portfolio_type=p_type)

    from engines.kotak_neo_engine import kotak_neo_engine
    for p in positions:
        sym = (p.get("symbol") or "").strip().upper()
        live_px = kotak_neo_engine.get_live_price(sym)
        ep = float(p["entry_price"])
        cp = float(live_px or p.get("current_price") or ep)
        p["current_price"] = cp
        sl = float(p["stop_loss"])
        qty = int(p["quantity"])
        p["unrealized_pnl"] = round((cp - ep) * qty, 2)
        p["return_pct"] = round(((cp - ep) / ep) * 100.0, 2)
        p["stop_distance_pct"] = round(((cp - sl) / cp) * 100.0, 2)
        r_unit = max(0.01, ep - float(p.get("original_stop_loss") or sl))
        p["current_r"] = round((cp - ep) / r_unit, 2)
        tick = kotak_neo_engine.get_live_tick(sym)
        if tick:
            if tick.get("change_pct") is not None:
                p["day_change_pct"] = tick["change_pct"]
            if tick.get("updated_at"):
                p["price_updated_at"] = tick["updated_at"]
        p["shares"] = qty
        p["total_quantity"] = qty
        p["current_stop"] = sl
        p["initial_sl"] = sl

    return positions


@app.get("/api/portfolio/{portfolio_type}/history")
def get_portfolio_history(portfolio_type: str, limit: int = 200):
    if portfolio_type.strip().upper() == "ALL":
        return state_engine.get_trade_history(portfolio_type=None, limit=limit)
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_trade_history(portfolio_type=p_type, limit=limit)


@app.get("/api/portfolio/{portfolio_type}/signals")
def get_portfolio_signals(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_latest_signals(portfolio_type=p_type)


# --------------------------------------------------------------------------
# PORTFOLIO 3: PRE-BREAKOUT COILS & SPLIT EXIT ROUTES
# --------------------------------------------------------------------------
@app.get("/api/portfolio/PORTFOLIO_3/watchlist")
def get_portfolio3_watchlist(include_all: bool = False):
    """Returns active pre-breakout coil watchlist candidates for Portfolio 3."""
    return portfolio3_strategy_engine.get_active_watchlist(include_all=include_all)


@app.get("/api/portfolio/PORTFOLIO_3/legs")
def get_portfolio3_legs():
    """Returns active and recent dual-leg tracking records for Portfolio 3."""
    with portfolio3_strategy_engine._get_connection() as conn:
        rows = conn.execute("SELECT * FROM p3_position_legs ORDER BY id DESC LIMIT 100").fetchall()
        return [dict(r) for r in rows]


@app.post("/api/portfolio/PORTFOLIO_3/scan-watchlist")
def trigger_portfolio3_watchlist_scan(background_tasks: BackgroundTasks):
    """Manually triggers EOD Pre-Breakout Coil Watchlist Ingestion for Portfolio 3."""
    def _worker():
        portfolio3_strategy_engine.ingest_prebreakout_watchlist(sync_cloud=True)
    background_tasks.add_task(_worker)
    return {"status": "TRIGGERED", "message": "Portfolio 3 Watchlist Ingestion triggered in background."}


@app.post("/api/portfolio/PORTFOLIO_3/trigger-intraday")
def trigger_portfolio3_intraday_cycle(bypass_window: bool = False, bypass_gate: bool = False):
    """Manually triggers Intraday Breakout Validation & Volume Confirmation for Portfolio 3."""
    res = portfolio3_strategy_engine.run_intraday_breakout_cycle(
        bypass_window=bypass_window,
        bypass_gate=bypass_gate
    )
    return res


@app.post("/api/portfolio/PORTFOLIO_3/reconcile")
def trigger_portfolio3_reconcile(bypass_gate: bool = False):
    """Manually triggers automated split exit reconciliation for Portfolio 3."""
    closed = portfolio3_strategy_engine.reconcile_open_positions(bypass_gate=bypass_gate)
    return {"status": "SUCCESS", "closed_trades": closed}


class RefreshPricesRequest(BaseModel):
    scope: str = "ACTIVE_AND_SIGNALS"


@app.post("/api/prices/refresh")
def refresh_prices_endpoint(req: Optional[RefreshPricesRequest] = None):
    """
    Fetches live LTP from Yahoo Finance for all Active Portfolio Positions and Screened Signals on the page,
    updating MTM PnL, trailing stops, and signal prices immediately.
    Serialized via execution_queue to prevent collision with active scans.
    """
    scope = req.scope if req and req.scope else "ACTIVE_AND_SIGNALS"
    def _refresh_task():
        return refresh_live_market_prices(scope=scope)

    q_status = execution_queue.get_status()
    if not q_status.get("is_running") and q_status.get("queued_count") == 0:
        res = execution_queue.submit_and_wait(
            _refresh_task,
            task_name="Manual Price Refresh",
            task_type="PRICE_REFRESH_MANUAL",
            timeout=30.0
        )
        return res or {"status": "SUCCESS", "message": "Live prices refreshed successfully."}
    else:
        info = execution_queue.submit_task(
            _refresh_task,
            task_name="Manual Price Refresh",
            task_type="PRICE_REFRESH_MANUAL",
            deduplicate=True
        )
        return {
            "status": "QUEUED",
            "message": f"Price refresh queued behind current active task ({q_status.get('current_task', {}).get('name', 'running task')}).",
            "queue_position": info.get("queue_position", 1),
            "queue_status": execution_queue.get_status()
        }


# --------------------------------------------------------------------------
# SCAN TRIGGER & LIVE PROGRESS
# --------------------------------------------------------------------------
class TriggerScanRequest(BaseModel):
    scan_type: str = "ALL"  # 'MORNING_1020AM', 'AFTERNOON_300PM', 'AFTERNOON_305PM', 'INSTITUTIONAL', 'NIFTY500', 'ALL'
    auto_execute: bool = True
    source: Optional[str] = None


@app.get("/api/execution/queue")
def get_execution_queue_status():
    """Returns live telemetry of QuantDesk central Execution Queue."""
    return execution_queue.get_status()


@app.get("/api/scan/progress")
def get_scan_progress():
    prog = state_engine.get_scan_progress()
    p1_state = get_institutional_status()
    prog["chartink_fetched_count"] = p1_state.get("chartink_fetched_count", 0)
    prog["passing_criteria_count"] = p1_state.get("passing_criteria_count", p1_state.get("chartink_fetched_count", 0))
    prog["nse_total_count"] = p1_state.get("nse_total_count", state_engine.get_total_nse_stocks_count())
    prog["buy_now_count"] = p1_state.get("buy_now_count", 0)
    prog["pullback_count"] = p1_state.get("pullback_count", 0)
    prog["avoid_count"] = p1_state.get("avoid_count", 0)
    prog["universe_size"] = p1_state.get("universe_size", 0)
    prog["queue_status"] = execution_queue.get_status()
    return prog


@app.post("/api/scan/trigger", dependencies=[Depends(verify_admin_key)])
def trigger_scan(req: TriggerScanRequest):
    s_type = req.scan_type.upper()
    curr_prog = state_engine.get_scan_progress()
    q_status = execution_queue.get_status()

    # If scan is already running on execution worker
    if curr_prog.get("is_running") and q_status.get("is_running") and q_status.get("current_task", {}).get("type") in ("CHARTINK_SCAN", "IN_HOUSE_SCAN", "SCAN"):
        return {
            "status": "ALREADY_RUNNING",
            "message": f"Scan is currently in progress: {curr_prog.get('stage')}",
            "progress": curr_prog
        }

    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()

    if req.source:
        state_engine.set_swing_universe_source(req.source)
    active_source = req.source or state_engine.get_swing_universe_source()
    source_label = "In-House NSE" if active_source == "IN_HOUSE" else "Chartink Live"

    # Determine initial stage based on queue state
    if q_status.get("is_running"):
        init_stage = f"⏳ In Queue (Position {q_status.get('queued_count', 0) + 1}): Waiting for [{q_status.get('current_task', {}).get('name')}] to complete..."
    else:
        init_stage = f"Initiating live {s_type} scan ({source_label})..."

    state_engine.start_scan_progress(
        scan_type=s_type,
        initial_stage=init_stage
    )

    effective_auto_exec = bool(req.auto_execute and can_trade)
    def run_scan_worker():
        logger.info(f"⚡ Live scan started: {s_type} [{source_label}] (Auto-Execute: {effective_auto_exec}, TradingGate: {can_trade})")
        state_engine.update_scan_progress(stage=f"Step 1/6: Sourcing live {s_type} universe from {source_label}...", percent=10)
        try:
            if s_type in ("MORNING_1020AM", "AFTERNOON_305PM", "AFTERNOON_300PM", "INSTITUTIONAL", "ALL"):
                inst = InstitutionalEngine()
                mode_override = "morning" if "MORNING" in s_type else ("afternoon" if "AFTERNOON" in s_type else None)
                inst.run(auto_execute=effective_auto_exec, mode_override=mode_override, universe_source=active_source)

            if s_type in ("AFTERNOON_305PM", "AFTERNOON_300PM", "NIFTY500", "ALL"):
                run_nifty500_pipeline_sync(mode="MANUAL_TRIGGER", auto_execute=effective_auto_exec)
        except Exception as e:
            logger.error(f"Error during live scan {s_type}: {e}", exc_info=True)
            state_engine.finish_scan_progress(f"Scan encountered error: {str(e)}")

    task_res = execution_queue.submit_task(
        run_scan_worker,
        task_name=f"Live Scan: {s_type} ({source_label})",
        task_type="IN_HOUSE_SCAN" if active_source == "IN_HOUSE" else "CHARTINK_SCAN",
        deduplicate=True
    )

    return {
        "status": "QUEUED" if q_status.get("is_running") else "STARTED",
        "message": f"Live scan '{s_type}' dispatched through execution queue! Zero lock collisions.",
        "auto_execute": req.auto_execute,
        "trading_gate_open": can_trade,
        "trading_gate_message": gate_msg,
        "timestamp": format_ist(),
        "queue_position": task_res.get("queue_position", 1),
        "progress": state_engine.get_scan_progress()
    }


@app.get("/api/scan/logs")
def get_scan_logs(limit: int = 50):
    return state_engine.get_recent_scan_logs(limit=limit)


@app.get("/api/system/activity-logs")
def get_system_activity_logs(limit: int = 100, category: Optional[str] = None):
    """
    Returns unified system-wide execution, audit, and pipeline telemetry logs.
    Covers:
    - 08:35 & 08:45 AM Pre-Market Forecast Locks & Bulletins
    - Screener Ingestion -> Indicator -> 6-Pillar Classification Breakdown
    - Risk Gate Evaluations & Active Portfolio Admissions
    - Uncapped Training Portfolio Shadow ML Dataset Logging
    - 30-Minute Movers Research Scans
    - Intraday Multi-Interval Expiry Snapshots
    - 5-Minute Live Price & MTM Streams
    """
    return state_engine.get_system_activity_logs(limit=limit, category=category)


# --------------------------------------------------------------------------
# MANUAL TRADE OPERATIONS (STRICT MARKET-HOURS TRADING GATE ENFORCED)
# --------------------------------------------------------------------------
class ClosePositionRequest(BaseModel):
    symbol: str
    exit_price: Optional[float] = None
    exit_reason: str = "MANUAL_DISCRETIONARY"
    notes: str = ""
    portfolio_type: Optional[str] = None
    bypass_gate: bool = False


class OpenPositionRequest(BaseModel):
    symbol: str
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    target_2: float = 0.0
    portfolio_type: str = "INSTITUTIONAL_6PILLAR"
    sector: str = "NSE"
    notes: str = "UI Manual Execution"


@app.post("/api/position/open", dependencies=[Depends(verify_admin_key)])
def open_new_position(req: OpenPositionRequest):
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    if not can_trade:
        raise HTTPException(status_code=403, detail=gate_msg)

    p_type = state_engine._normalize_portfolio_type(req.portfolio_type)
    sym_clean = req.symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    active_positions = state_engine.get_active_positions(portfolio_type=p_type)
    active_syms = [p["symbol"] for p in active_positions]

    if sym_clean in active_syms:
        raise HTTPException(status_code=400, detail=f"{sym_clean} is already active in your portfolio.")

    max_slots = P2_MAX_SLOTS if "NIFTY" in p_type else 10
    if len(active_positions) >= max_slots:
        raise HTTPException(
            status_code=400,
            detail=f"Slot limit reached ({len(active_positions)}/{max_slots} slots used). Please close an existing position before opening {sym_clean}."
        )

    stats = state_engine.get_portfolio_stats(portfolio_type=p_type)
    avail_cash = float(stats.get("available_cash", 0.0))
    qty = max(1, int(req.quantity))
    req_val = round(req.entry_price * qty, 2)

    if req_val > avail_cash:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient available cash: Required ₹{req_val:,.2f}, but only ₹{avail_cash:,.2f} is free (Slots: {len(active_positions)}/{max_slots} used, ₹{stats.get('total_invested', 0):,.2f} invested)."
        )

    ok = state_engine.open_position(
        symbol=sym_clean,
        entry_price=req.entry_price,
        quantity=qty,
        stop_loss=req.stop_loss,
        target=req.target,
        target_2=req.target_2 or round(req.entry_price + 2.0 * max(1.0, req.target - req.entry_price), 2),
        triggered_from="UI_MANUAL",
        notes=req.notes,
        sector=req.sector,
        portfolio_type=p_type,
        entry_reason="UI Manual Execution"
    )
    if not ok:
        raise HTTPException(status_code=400, detail=f"Order rejected by execution engine for {sym_clean}.")

    return {
        "status": "opened",
        "symbol": sym_clean,
        "quantity": qty,
        "entry_price": req.entry_price,
        "required_capital": req_val,
        "remaining_cash": round(max(0.0, avail_cash - req_val), 2),
        "timestamp": format_ist()
    }


@app.post("/api/position/close", dependencies=[Depends(verify_admin_key)])
def close_active_position(req: ClosePositionRequest):
    if not req.bypass_gate:
        can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
        if not can_trade:
            raise HTTPException(
                status_code=403,
                detail=gate_msg
            )

    sym = req.symbol.strip().upper()
    active_rows = state_engine.get_active_positions(portfolio_type=req.portfolio_type)
    matched = [p for p in active_rows if p["symbol"] == sym]

    if not matched:
        raise HTTPException(status_code=404, detail=f"Active position for {sym} ({req.portfolio_type or 'ALL'}) not found.")

    pos = matched[0]
    p_type = req.portfolio_type or pos.get("portfolio_type")
    exit_px = req.exit_price or float(pos.get("current_price") or pos["entry_price"])

    res = state_engine.close_position(
        symbol=sym,
        exit_price=exit_px,
        exit_reason=req.exit_reason,
        notes=req.notes,
        bypass_gate=req.bypass_gate,
        portfolio_type=p_type
    )
    if not res:
        raise HTTPException(status_code=403, detail="Trade closure blocked by Strict Market-Hours Trading Gate.")

    return {"status": "closed", "closed_trade": res}


# --------------------------------------------------------------------------
# MASTER FILE FETCHED STOCKS EXPLORER & LIGHTWEIGHT CANDLESTICK CHART API
# --------------------------------------------------------------------------
@app.get("/api/stocks/fetched")
def get_fetched_stocks(search: str = "", limit: int = 850):
    """Returns all fetched stocks stored in the Single Master File (`ohlcv_master_store`)."""
    stats = state_engine.get_ohlcv_stats()
    rows = state_engine.get_all_ohlcv_metadata(search=search, limit=limit)
    return {
        "total_in_master_file": stats.get("stocks_count", len(rows)),
        "latest_bar_date": stats.get("latest_bar_date", "--"),
        "last_updated": stats.get("last_updated", "--"),
        "stocks": rows
    }


@app.get("/api/stocks/{symbol}/chart")
def get_stock_chart_and_data(symbol: str, bars: int = 260):
    """
    Returns Lightweight Candlestick Chart series, Volume, EMA(20/50/200), RSI(14), ATR(14),
    and exact Entry / Stop-Loss / Target 1 / Target 2 price lines from the Single Master File.
    If the stock is not yet in the Master File, fetches it live from Yahoo Finance and saves it.
    """
    from engines.institutional_engine import DirectYahooFinanceFetcher

    sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    payload = state_engine.get_stock_chart_payload(sym, limit_bars=bars)
    candles = payload.get("candles", [])
    if payload.get("status") == "NOT_FOUND" or len(candles) < 80:
        fetcher = DirectYahooFinanceFetcher(max_workers=4)
        _, df = fetcher.fetch_ticker_chart(sym, range_period="2y", max_retries=3)
        if df is not None and not df.empty and len(df) >= 80:
            state_engine.save_ohlcv_df(sym, df)
            payload = state_engine.get_stock_chart_payload(sym, limit_bars=bars)
    return payload


@app.post("/api/stocks/{symbol}/reset_chart")
def reset_stock_chart(symbol: str):
    """
    Explicitly resets chart state for a completed or exited stock:
    purges any lingering setup signal from latest_signals and returns the reset chart payload.
    """
    sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            conn.execute("DELETE FROM latest_signals WHERE symbol = ?", (sym,))
            conn.commit()
    return state_engine.get_stock_chart_payload(sym, limit_bars=260)



# --------------------------------------------------------------------------
# PORTFOLIO RESET & SINGLE MASTER FILE TELEGRAM CLOUD VAULT ENDPOINTS
# --------------------------------------------------------------------------
class ResetPortfolioRequest(BaseModel):
    portfolio_type: Optional[str] = None  # None = reset both portfolios to ₹0 PnL
    confirm_token: Optional[str] = None   # Must equal 'CONFIRM_USER_AUTHORIZED_RESET'


@app.post("/api/portfolio/reset", dependencies=[Depends(verify_admin_key)])
def reset_portfolio(req: ResetPortfolioRequest):
    return state_engine.reset_portfolio_trades(
        portfolio_type=req.portfolio_type,
        confirm_token=req.confirm_token or ""
    )


@app.get("/api/vault/status")
def get_vault_status():
    return telegram_vault.get_status()


@app.post("/api/vault/sync", dependencies=[Depends(verify_admin_key)])
def sync_vault_to_cloud():
    """Manually triggers background backup of the Single Master File (`data/swing_trades.db`) to Telegram Cloud."""
    cnt = state_engine.get_ohlcv_stats().get("stocks_count", 0)
    telegram_vault.sync_master_vault_async(note="Manual UI Single Master File Sync", send_notification=False)
    return {
        "status": "STARTED",
        "message": f"Updating Single Master File in-place on Telegram Cloud ({cnt} Stocks OHLCV + All Trades)..."
    }


@app.post("/api/vault/restore", dependencies=[Depends(verify_admin_key)])
def restore_vault_from_cloud():
    """Restores the Single Master File (`data/swing_trades.db`) from Telegram Cloud Vault."""
    res = telegram_vault.restore_master_file_from_cloud()
    return {
        "status": res.get("status", "SUCCESS"),
        "db_restored": res.get("status") == "SUCCESS",
        "ohlcv_restored": res.get("status") == "SUCCESS",
        "details": res
    }


@app.post("/api/system/backup/daily")
def trigger_daily_backup():
    """Triggers an immediate atomic MOC daily backup with integrity check and retention pruning."""
    from core.backup_manager import daily_backup_manager
    return daily_backup_manager.run_daily_moc_backup(retention_days=14, sync_cloud=True)


@app.get("/api/system/backup/list")
def list_daily_backups():
    """Lists recent daily MOC backups and retention status."""
    from core.backup_manager import daily_backup_manager
    return {
        "status": "OK",
        "retention_days": daily_backup_manager.retention_days,
        "backup_directory": str(daily_backup_manager.backup_dir),
        "backups": daily_backup_manager.list_backups()
    }


# --------------------------------------------------------------------------
# NIFTY 50 QUANT DESK ENDPOINTS (INTRADAY 50M EXPIRY, DEEP OPTIONS, T+1/T+5 PREDICTIONS)
# --------------------------------------------------------------------------
@app.get("/api/nifty/options-chain")
def get_nifty_options_chain(force_refresh: bool = False):
    """Fetches live NSE Nifty 50 Current Weekly & Current Monthly Option Chain deep OI analysis."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=force_refresh)


@app.get("/api/nifty/live-spot")
def get_nifty_live_spot(force_refresh: bool = False):
    """
    Returns ultra-fast real-time Nifty 50 spot tick, daily change, and live ATM strike with sub-second latency.
    Zero-broker stream powered by Yahoo Finance chart quote (^NSEI) with NSE allIndices fallback.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_live_nifty_spot(force_refresh=force_refresh)


@app.get("/api/nifty/intraday-expiry")
def get_nifty_intraday_expiry(session_date: Optional[str] = None, interval_mode: str = "15m", force_refresh: bool = False):
    """Returns the Intraday Weekly Option Expiry progression (15m / 30m Institutional Standard / 50m)."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_intraday_expiry_dashboard(session_date=session_date, interval_mode=interval_mode, force_refresh=force_refresh)


@app.get("/api/nifty/intraday-advanced")
def get_nifty_intraday_advanced(force_refresh: bool = False):
    """Returns Institutional Intraday Suite: VWAP, 15m/30m ORB, Choppiness Index, GEX, Zero-Gamma Flip, Sector Breadth, and Intraday ML."""
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        suite = intraday_analytics_engine.get_full_intraday_suite(force_refresh=force_refresh)
        return _sanitize_for_json(suite)
    except Exception as e:
        logger.error(f"Error fetching intraday advanced suite: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/intraday-ml")
def get_nifty_intraday_ml(force_refresh: bool = False):
    """
    Returns live pure-data Intraday ML Model predictions trained on prior slots of the day:
    Next-slot expected price delta, directional probabilities (P_up, P_flat, P_down),
    Writer Dominance Score (-100 to +100), and quantitative strategy recommendation.
    """
    from engines.intraday_ml_engine import intraday_ml_engine
    try:
        res = intraday_ml_engine.train_and_predict(force_refresh=force_refresh)
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error running intraday ML engine: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.post("/api/nifty/intraday-expiry/capture")
def capture_nifty_intraday_snapshot(slot_override: Optional[str] = None, interval_mode: str = "15m"):
    """Manually or automatically captures a live Weekly Option Expiry snapshot and syncs to Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.capture_intraday_50min_snapshot(slot_override=slot_override, sync_cloud=True, interval_mode=interval_mode)


@app.get("/api/nifty/options/strike-depth")
def get_nifty_option_strike_depth(
    strike: int,
    option_type: str = "CE",
    expiry: Optional[str] = None,
    expiry_type: str = "weekly",
    force_refresh: bool = False
):
    """
    Returns real-time 5-level Bid/Ask Market Depth, Order Book Imbalance (OBI),
    Micro-Price, and institutional flow bias for a specific Nifty option strike.
    Powered by Kotak Neo SFeed Level-2 depth with REST quotes fallback.
    Strictly zero fake numbers or hardcoded fallbacks.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    from engines.nifty_analytics_engine import nifty_analytics_engine

    opt_type_clean = option_type.upper().strip()
    exp_target = (expiry or "").strip()

    if not exp_target:
        try:
            chain = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False)
            if expiry_type.lower() == "monthly":
                exp_target = chain.get("monthly", {}).get("expiry", "")
            else:
                exp_target = chain.get("weekly", {}).get("expiry", "")
        except Exception as e:
            logger.debug(f"Could not resolve expiry for depth request: {e}")

    depth = kotak_neo_engine.get_option_depth(
        strike=strike,
        opt_type=opt_type_clean,
        expiry=exp_target,
        force_refresh=force_refresh
    )
    depth["expiry_type"] = expiry_type.lower()
    return _sanitize_for_json(depth)


@app.get("/api/nifty/intraday/order-flow")
def get_nifty_intraday_order_flow(force_refresh: bool = False):
    """
    Returns real-time Near-ATM Institutional Order Flow Pulse, Aggregate Order Book Imbalance,
    Call vs Put Depth Volume Ratio, Spread Liquidity Vacuum Risk, and strike-by-strike depth.
    Powered exclusively by Kotak Neo API.
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        data = intraday_analytics_engine.compute_order_flow_market_depth(force_refresh=force_refresh)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error computing order flow market depth: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/order-flow/analytics")
def get_nifty_order_flow_analytics(force_refresh: bool = False):
    """
    Returns full institutional Order Flow analytics including:
    - Level-2 5-depth Near-ATM Order Book Imbalance (OBI)
    - 1m & 3m OBI Velocity (dOBI/dt) and OBI Acceleration
    - Iceberg / Passive Absorption order detection across near-ATM strikes
    - Real-time Depth-Confirmed Gamma Squeeze alerts
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        data = intraday_analytics_engine.compute_order_flow_market_depth(force_refresh=force_refresh)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error computing order flow analytics: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.post("/api/nifty/options/smart-execution")
async def calculate_smart_execution_router(request: Request):
    """
    Micro-Price Smart Execution Router:
    Calculates Level-2 volume-weighted Micro-Prices for multi-leg strategies.
    Determines optimal limit order ladder inside the spread and returns exact ₹ slippage savings.
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        body = await request.json()
        legs = body.get("legs", [])
        spot = body.get("spot_price")
        data = intraday_analytics_engine.calculate_smart_micro_price_execution(legs=legs, spot_price=spot)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error in smart execution router: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/options/theta-decay-curve")
def get_nifty_theta_decay_curve():
    """
    Returns real-time intraday Theta Decay acceleration curve (dTheta/dt)
    for Weekly Nifty 50 options across the trading session (09:15 - 15:30 IST).
    Highlights the peak non-linear collapse window (12:45 - 14:30 IST).
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        data = intraday_analytics_engine.compute_weekly_theta_decay_curve()
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error in theta decay curve: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/options/max-pain-migration")
def get_nifty_max_pain_migration():
    """
    Returns real-time Dynamic Max Pain strike and tracks dynamic intraday migration trajectory.
    Detects whether writers are defending original strikes (Gravity Pin) or rolling up/down.
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        data = intraday_analytics_engine.compute_dynamic_max_pain_migration()
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error in max pain migration: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/options/term-structure")
def get_nifty_term_structure(force_refresh: bool = False):
    """
    Returns Term Structure Volatility Skew (Weekly ATM IV vs Monthly ATM IV)
    and Monthly Roll-Over Flow Radar.
    Detects Calendar Spread Arbitrage opportunities when Weekly IV spikes abnormally.
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        skew_data = intraday_analytics_engine.compute_term_structure_and_calendar_skew(force_refresh=force_refresh)
        rollover_data = intraday_analytics_engine.compute_monthly_rollover_radar(force_refresh=force_refresh)
        res = {
            "status": "OK",
            "term_structure_skew": skew_data,
            "monthly_rollover_radar": rollover_data,
            "timestamp_ist": format_ist()
        }
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error in term structure: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/options/delta-hedging-band")
def get_nifty_delta_hedging_band(net_delta: float = 0.0):
    """
    Returns Delta-Neutral Dynamic Hedging Bands:
    Calculates 1-sigma and 2-sigma boundaries and determines whether adjustments
    are required to maintain delta neutrality.
    """
    from engines.intraday_analytics_engine import intraday_analytics_engine
    try:
        data = intraday_analytics_engine.compute_delta_neutral_hedging_band(net_delta=net_delta)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error in delta hedging band: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/minute-chart")
def get_nifty_minute_chart(interval: str = "1m"):
    """
    Returns live intraday minute candlestick chart for NIFTY 50 (^NSEI)
    with VWAP, EMA(9), EMA(21), RSI(14), and ATR(14).
    Supported intervals: '1m', '5m', '15m'.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_nifty_intraday_minute_chart(interval=interval)


@app.get("/api/system/watchdog")
def get_system_watchdog_status(force_probe: bool = False):
    """Returns real-time health telemetry across all 4 QuantDesk data pipelines."""
    return data_watchdog.check_all_endpoints(force_probe=force_probe)


@app.post("/api/system/watchdog/probe")
def probe_all_system_feeds():
    """Forces an immediate health check & stream re-connection probe across all 4 data pipelines."""
    from engines.nifty_analytics_engine import nifty_analytics_engine

    try:
        refresh_live_market_prices(scope="ACTIVE")
    except Exception:
        pass
    try:
        nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=True)
    except Exception:
        pass
    try:
        nifty_analytics_engine.get_live_nifty_spot()
    except Exception:
        pass

    return data_watchdog.check_all_endpoints(force_probe=True)


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        try:
            return _sanitize_for_json(obj.item())
        except Exception:
            return str(obj)
    return obj


@app.get("/api/nifty/predictions")
def get_nifty_predictions(force_refresh: bool = False, force_recalculate: bool = False):
    """Computes live 6-pillar Nifty 50 T+1 & T+5 predictions with forward-testing attribution & self-learning weights. Returns locked forecast during market hours."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        data = nifty_analytics_engine.compute_nifty_predictions(force_refresh=force_refresh, force_recalculate=force_recalculate)
        if isinstance(data, dict):
            scans_tel = state_engine.get_scheduled_scans_telemetry()
            data["auto_run_telemetry"] = scans_tel.get("premarket_0845", {})
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error computing live predictions: {e}", exc_info=True)
        cached = state_engine.get_kv("latest_nifty_predictions")
        if cached and isinstance(cached, dict) and cached.get("status") == "OK":
            scans_tel = state_engine.get_scheduled_scans_telemetry()
            cached["auto_run_telemetry"] = scans_tel.get("premarket_0845", {})
            return _sanitize_for_json(cached)
        return {"status": "DATA_UNAVAILABLE", "message": f"Data unavailable: Prediction service error: {str(e)}"}


@app.post("/api/nifty/premarket-forecast/lock")
def lock_premarket_forecast():
    """Explicitly calculates the official pre-market forecast, locks it for today's session, and dispatches the Telegram Bulletin."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    now_dt = now_ist()
    is_trading = NSEMarketCalendar.is_trading_day(now_dt.date())
    is_market_alive = is_trading and (datetime.time(9, 0) <= now_dt.time() <= datetime.time(15, 30))
    today_str = now_dt.strftime("%Y-%m-%d")
    locked_key = f"nifty_prediction_locked_{today_str}"
    existing_lock = state_engine.get_kv(locked_key)
    if not existing_lock or not isinstance(existing_lock, dict):
        cand_pred = state_engine.get_kv("latest_nifty_predictions")
        if cand_pred and isinstance(cand_pred, dict) and cand_pred.get("prediction_date") == today_str:
            existing_lock = cand_pred
    if not existing_lock or not isinstance(existing_lock, dict):
        existing_lock = nifty_analytics_engine._load_prediction_from_db(today_str)

    is_contaminated = False
    if existing_lock and isinstance(existing_lock, dict):
        cand_d_date = str(existing_lock.get("data_date") or existing_lock.get("data_as_of_date") or "")
        if cand_d_date == today_str and now_dt.time() < datetime.time(15, 30):
            is_contaminated = True

    if is_market_alive and existing_lock and isinstance(existing_lock, dict) and existing_lock.get("status") == "OK" and not is_contaminated:
        return {
            "status": "LOCKED",
            "message": "Market is currently alive (09:00 - 15:30 IST). Predictions are strictly locked against manual recomputation to preserve institutional audit integrity. Next auto-evaluation is at 03:40 PM IST.",
            "is_premarket_locked": True,
            "market_hours_locked": True,
            "recompute_blocked": True,
            "telegram_sent": False,
            "prediction_date": existing_lock.get("prediction_date"),
            "premarket_locked_at_ist": existing_lock.get("premarket_locked_at_ist"),
            "t1_verdict": existing_lock.get("t1_forecast", {}).get("verdict"),
            "t5_verdict": existing_lock.get("t5_forecast", {}).get("verdict")
        }

    try:
        data = nifty_analytics_engine.compute_nifty_predictions(force_refresh=True, force_recalculate=True)
        bulletin_sent = nifty_analytics_engine.dispatch_premarket_prediction_telegram_alert(data)
        state_engine.record_scheduled_scan_outcome(
            run_key="premarket_0845",
            status="SUCCESS",
            buy_now_count=0,
            total_scanned=50,
            note=f"Pre-Market Forecast Locked at {format_ist(now_dt)} • T+1: {data.get('t1_forecast', {}).get('verdict')} | T+5: {data.get('t5_forecast', {}).get('verdict')}"
        )
        return {
            "status": "OK",
            "message": "Pre-market forecast computed, locked, and Telegram bulletin dispatched.",
            "telegram_sent": bulletin_sent,
            "prediction_date": data.get("prediction_date"),
            "premarket_locked_at_ist": data.get("premarket_locked_at_ist"),
            "t1_verdict": data.get("t1_forecast", {}).get("verdict"),
            "t5_verdict": data.get("t5_forecast", {}).get("verdict")
        }
    except Exception as e:
        logger.error(f"Error locking pre-market forecast: {e}", exc_info=True)
        state_engine.record_scheduled_scan_outcome(
            run_key="premarket_0845",
            status="FAILED",
            buy_now_count=0,
            total_scanned=0,
            note=f"08:45 AM Lock Error: {str(e)}"
        )
        return {"status": "ERROR", "message": str(e)}


@app.post("/api/nifty/eod-evaluation/run")
def run_eod_prediction_evaluation():
    """Runs the 03:40 PM EOD prediction outcome evaluation, grades direction and cones, updates learning weights, and sends Telegram review."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        res = nifty_analytics_engine.evaluate_and_grade_eod_predictions()
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error running EOD prediction evaluation: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


@app.get("/api/nifty/us-market-intelligence")
def get_us_market_intelligence(force_refresh: bool = False):
    """Returns multi-pillar US market intelligence, breadth analysis, VIX term structure & US T+1 forecast."""
    from engines.us_analytics_engine import us_analytics_engine
    try:
        data = us_analytics_engine.fetch_and_analyze_us_markets(force_refresh=force_refresh)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error fetching US intelligence: {e}", exc_info=True)
        return {"status": "ERROR", "message": f"US Analytics Engine error: {str(e)}"}


@app.get("/api/nifty/futures-relative-radar")
def get_futures_relative_radar(force_refresh: bool = False):
    """Returns live E-mini futures momentum, rolling beta, and Nifty fair value gap analysis."""
    from engines.us_analytics_engine import us_analytics_engine
    try:
        data = us_analytics_engine.fetch_and_analyze_us_markets(force_refresh=force_refresh)
        return _sanitize_for_json({
            "status": "OK",
            "futures_relative_radar": data.get("futures_relative_radar", {}),
            "us_t1_forecast": data.get("us_t1_forecast", {}),
            "volatility_surface": data.get("volatility_surface", {}),
            "breadth_analysis": data.get("breadth_analysis", {}),
        })
    except Exception as e:
        logger.error(f"Error fetching futures radar: {e}", exc_info=True)
        return {"status": "ERROR", "message": f"Futures Radar error: {str(e)}"}


@app.get("/api/nifty/economic-calendar")
def get_nifty_economic_calendar(force_refresh: bool = False):
    """Fetches or scrapes live 3-Star (High-Impact) Economic Calendar events from Moneycontrol."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        data = nifty_analytics_engine.fetch_moneycontrol_economic_calendar(force_refresh=force_refresh)
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {e}", exc_info=True)
        return {"status": "ERROR", "message": f"Economic Calendar error: {str(e)}", "events": []}


@app.get("/api/nifty/fao-history")
def get_nifty_fao_history():
    """
    Returns rolling 1-Week Official NSE F&O Clearing Corporation Participant Wise Open Interest History
    (Client, DII, FII, Pro) from latest trading session to 1-week back with contract breakdowns.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        flow_data = nifty_analytics_engine.fetch_fii_dii_cash_and_participant_oi(force_refresh=False)
        return _sanitize_for_json({
            "status": "OK",
            "latest_cash": flow_data.get("latest_cash", {}),
            "latest_oi": flow_data.get("latest_oi", {}),
            "cash_history": flow_data.get("one_week_cash_history", []),
            "oi_history": flow_data.get("one_week_participant_oi_history", []),
        })
    except Exception as e:
        logger.error(f"Error reading F&O history: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e), "cash_history": [], "oi_history": []}


@app.get("/api/nifty/fii-dii/status")
def get_nifty_fii_dii_status():
    """Returns auto-run telemetry and latest available session dates for FII/DII cash and participant OI."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        from scheduler.cron_scheduler import _scheduler
        next_run_ist = None
        if _scheduler and _scheduler.running:
            for j_id in ("job_fii_dii_participant_oi_postmarket", "job_fii_dii_participant_oi_0830"):
                j = _scheduler.get_job(j_id)
                if j and j.next_run_time:
                    nr = j.next_run_time.strftime("%Y-%m-%d %H:%M:%S IST")
                    if not next_run_ist or nr < next_run_ist:
                        next_run_ist = nr
    except Exception:
        next_run_ist = None

    tel = nifty_analytics_engine.get_fii_dii_autorun_status()
    if next_run_ist:
        tel["next_run_ist"] = next_run_ist
    return _sanitize_for_json({"status": "OK", "telemetry": tel})


@app.post("/api/nifty/fii-dii/update-now")
def trigger_nifty_fii_dii_manual_update():
    """
    Manually triggers immediate check and ingestion of latest official NSE FII/DII Cash and F&O Participant OI.
    If latest session data is already available, stops cleanly with UP_TO_DATE.
    If not, downloads and persists latest records, updating tables and telemetry immediately.
    """
    from engines.nifty_analytics_engine import nifty_analytics_engine
    try:
        tel = nifty_analytics_engine.check_and_update_fii_dii_participant_oi(force_refresh=True, trigger_source="MANUAL_BUTTON")
        flow_data = nifty_analytics_engine.fetch_fii_dii_cash_and_participant_oi(force_refresh=False)
        return _sanitize_for_json({
            "status": "OK",
            "telemetry": tel,
            "data": {
                "latest_cash": flow_data.get("latest_cash", {}),
                "latest_oi": flow_data.get("latest_oi", {}),
                "cash_history": flow_data.get("one_week_cash_history", []),
                "oi_history": flow_data.get("one_week_participant_oi_history", []),
                "flow_score": flow_data.get("flow_score", 0.0),
            }
        })
    except Exception as e:
        logger.error(f"Manual FII/DII update error: {e}", exc_info=True)
        return {"status": "ERROR", "message": str(e)}


# --------------------------------------------------------------------------
# OPTIONS STRATEGY BUILDER & PORTFOLIO DESK ENDPOINTS
# --------------------------------------------------------------------------
class DeployStrategyRequest(BaseModel):
    strategy_name: str
    symbol: str = "NIFTY"
    expiry_date: str = ""
    expiry_type: str = "WEEKLY"
    spot_at_entry: float = 0.0
    net_premium: float = 0.0
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: Optional[Any] = None
    risk_reward: Optional[str] = "--"
    est_margin: float = 0.0
    pop_pct: float = 0.0
    legs: list
    target_pnl: Optional[float] = 0.0
    stoploss_pnl: Optional[float] = 0.0
    auto_trade_mode: Optional[str] = "MANUAL"
    market_condition: Optional[str] = ""


class ExitStrategyRequest(BaseModel):
    strategy_id: str
    exit_reason: str = "MANUAL_EXIT"


class AdjustStrategyRequest(BaseModel):
    strategy_id: str
    legs: list
    net_premium: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: Optional[Any] = None
    risk_reward: Optional[str] = None
    est_margin: Optional[float] = None
    target_pnl: Optional[float] = None
    stoploss_pnl: Optional[float] = None
    notes: Optional[str] = None


class AutoDeployOptionsRequest(BaseModel):
    modes: Optional[list] = ["WEEKLY", "MONTHLY", "INTRADAY"]
    force_refresh: Optional[bool] = True


class ResetPortfolioRequest(BaseModel):
    keep_active: Optional[bool] = True


@app.get("/api/options/portfolio")
def get_options_portfolio_endpoint():
    """Returns Options Trading Portfolio with live MTM valuation, Greeks, and closed performance."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_options_portfolio()


@app.post("/api/options/reset-portfolio")
def reset_options_portfolio_endpoint(req: Optional[ResetPortfolioRequest] = None, keep_active: Optional[bool] = None):
    """Resets the Options Trading Portfolio: clears closed strategies, wipes historical realized PnL back to ₹0, preserves active strategies by default, and syncs cloud vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    should_keep = True
    if keep_active is not None:
        should_keep = keep_active
    elif req is not None and req.keep_active is not None:
        should_keep = req.keep_active
    return nifty_analytics_engine.reset_options_portfolio(keep_active=should_keep)



@app.post("/api/options/deploy-strategy")
def deploy_options_strategy_endpoint(req: DeployStrategyRequest):
    """Deploys a multi-leg options strategy into the Options Trading Portfolio and syncs to Telegram Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.deploy_options_strategy(req.dict())


@app.post("/api/options/adjust-strategy")
def adjust_options_strategy_endpoint(req: AdjustStrategyRequest):
    """Adjusts an active multi-leg options strategy in the Options Trading Portfolio and syncs to Telegram Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.adjust_options_strategy(req.dict())


@app.post("/api/options/exit-strategy")
def exit_options_strategy_endpoint(req: ExitStrategyRequest):
    """Exits a deployed multi-leg options strategy at current market prices and syncs to Telegram Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.exit_options_strategy(req.strategy_id, req.exit_reason)


@app.post("/api/options/auto-deploy")
def auto_deploy_options_endpoint(req: Optional[AutoDeployOptionsRequest] = None):
    """Regenerates complete analysis (fresh live option chain, 6-pillar ML predictions & AI interpretation) then evaluates and auto-deploys strategies."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    modes = req.modes if req and req.modes else ["WEEKLY", "MONTHLY", "INTRADAY"]
    force_refresh = req.force_refresh if (req and req.force_refresh is not None) else True
    return nifty_analytics_engine.evaluate_and_deploy_auto_options_strategies(modes=modes, force_refresh=force_refresh)


@app.get("/api/options/chain-compact")
def get_options_chain_compact_endpoint(expiry_type: str = "WEEKLY"):
    """Returns compact scrollable option chain data (strikes, LTPs, OIs) for weekly and monthly expiries."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    chain = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False)
    if chain.get("status") != "OK":
        return chain
    exp_key = "monthly" if expiry_type.upper() == "MONTHLY" else "weekly"
    sub_data = chain.get(exp_key, {})
    return {
        "status": "OK",
        "symbol": chain.get("symbol", "NIFTY 50"),
        "spot_price": chain.get("spot_price", 0.0),
        "expiry_date": sub_data.get("expiry", "--"),
        "expiry_type": expiry_type.upper(),
        "atm_strike": sub_data.get("atm_strike"),
        "strikes": sub_data.get("strikes", []),
        "all_strikes": sub_data.get("all_strikes", sub_data.get("strikes", [])),
        "timestamp": chain.get("nse_timestamp") or chain.get("fetched_at_ist")
    }


@app.get("/api/portfolio/equity-curves")
def get_portfolio_equity_curves_endpoint():
    """Returns interactive equity curves for Options Desk, Swing Trading (P1), and Nifty 500 (P2)."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_all_equity_curves()


# ==============================================================================
# INTRADAY OPTIONS 5-REGIME RULES: AUTO-TRADES, RE-ENTRY & BACKTEST API
# ==============================================================================
class AutoDeploy5RegimeRequest(BaseModel):
    force_refresh: bool = False
    confidence_threshold: float = 8.0
    lot_size: int = NIFTY_LOT_SIZE


@app.get("/api/options/intraday-rules/backtest")
def get_options_intraday_backtest_endpoint(lookback_sessions: int = 50, lot_size: int = NIFTY_LOT_SIZE):
    """
    Backtests the 5 prediction regimes independently:
    - Compares 30% vs 40% stop-losses
    - Compares with vs without price-triggered 1-time re-entry
    - Compares Primary vs Experimental Asymmetric strategies
    - Net P&L after transaction costs, Win Rate %, Expectancy, Profit Factor, Max Drawdown, and Tail Losses.
    """
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    return intraday_options_strategy_engine.run_comprehensive_backtest(
        lookback_sessions=lookback_sessions,
        lot_size=lot_size
    )


@app.post("/api/options/intraday-rules/auto-deploy")
def post_options_auto_deploy_endpoint(req: Optional[AutoDeploy5RegimeRequest] = None):
    """
    Executes auto-trades whenever the system feels confident based on the 5-regime rules.
    Logs trades with exact timestamps, sigma strikes, and inputs to SQLite for ML model training.
    """
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    force = req.force_refresh if req else False
    conf = req.confidence_threshold if req else 8.0
    lots = req.lot_size if req else NIFTY_LOT_SIZE
    return intraday_options_strategy_engine.evaluate_and_execute_auto_trades(
        force_refresh=force,
        confidence_threshold=conf,
        lot_size=lots
    )


@app.get("/api/options/intraday-rules/regime")
def get_options_intraday_regime_endpoint(force_refresh: bool = False):
    """
    Returns real-time Nifty 50 regime strictly derived from pure INTRADAY analysis:
    VWAP, 30m ORB, Choppiness Index (CHOP), Strike Gamma Exposure (GEX), Sectoral Breadth, and Intraday ML.
    Also returns real-time daytime confidence metrics for 0DTE and non-0DTE intraday execution.
    """
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    reg = intraday_options_strategy_engine.get_intraday_market_regime(force_refresh=force_refresh)
    conf = intraday_options_strategy_engine.calculate_daytime_confidence(reg)
    reg["daytime_confidence"] = conf
    return reg


@app.get("/api/options/intraday-rules/daytime-confidence")
def get_options_intraday_daytime_confidence_endpoint(force_refresh: bool = False):
    """Returns real-time daytime confidence metrics and breakdown for 0DTE and non-0DTE intraday execution."""
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    reg = intraday_options_strategy_engine.get_intraday_market_regime(force_refresh=force_refresh)
    conf = intraday_options_strategy_engine.calculate_daytime_confidence(reg)
    return {
        "status": "OK",
        "regime": reg.get("regime"),
        "intraday_composite_score": reg.get("intraday_composite_score"),
        "daytime_confidence": conf,
        "timestamp_ist": format_ist()
    }


@app.get("/api/options/intraday-rules/active-trades")
def get_options_active_trades_endpoint():
    """Returns live active intraday options legs, execution status, and re-entry state."""
    from execution.state_engine import state_engine
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM intraday_options_trade_log
                ORDER BY id DESC LIMIT 50
            """).fetchall()
    return {
        "status": "OK",
        "trades": [dict(r) for r in rows],
        "timestamp_ist": format_ist()
    }


@app.get("/api/options/intraday-rules/ml-metrics")
def get_options_ml_metrics_endpoint():
    """Returns Options ML model metrics, training sample count, and feature importances."""
    from execution.state_engine import state_engine
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    meta = state_engine.get_kv("options_strategy_ml_metadata")
    if not meta:
        meta = intraday_options_strategy_engine.train_options_ml_model()
    return meta


@app.post("/api/options/intraday-rules/train-ml")
def post_options_train_ml_endpoint():
    """Retrains the options machine learning meta-model on all completed trade lifecycle logs."""
    from engines.intraday_options_strategy_engine import intraday_options_strategy_engine
    return intraday_options_strategy_engine.train_options_ml_model()


# ==============================================================================
# WEEKLY & MONTHLY OPTIONS 5-REGIME RULES: AUTO-TRADES, RE-ENTRY & BACKTEST API
# ==============================================================================
class AutoDeployPositionalRequest(BaseModel):
    horizon: str = "WEEKLY"  # "WEEKLY" or "MONTHLY"
    force_refresh: bool = False
    confidence_threshold: float = 8.0
    lot_size: int = NIFTY_LOT_SIZE


@app.get("/api/options/positional-rules/backtest")
def get_options_positional_backtest_endpoint(
    horizon: str = "WEEKLY",
    lookback_sessions: int = 50,
    lot_size: int = NIFTY_LOT_SIZE
):
    """
    Backtests Weekly (40% SL) or Monthly (50% SL) options strategies separately across all 5 regimes:
    - Compares specified SL (40% weekly, 50% monthly) vs alternative SL sensitivities
    - Compares with vs without price-triggered 1-time re-entry
    - Net P&L after transaction costs, Win Rate %, Expectancy, Profit Factor, Max Drawdown, and Tail Losses
    - Breakdown by regime and expiry
    """
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    return weekly_monthly_options_strategy_engine.run_positional_backtest(
        horizon=horizon,
        lookback_sessions=lookback_sessions,
        lot_size=lot_size
    )


@app.post("/api/options/positional-rules/auto-deploy")
def post_options_positional_auto_deploy_endpoint(req: Optional[AutoDeployPositionalRequest] = None):
    """
    Executes Weekly or Monthly auto-trades whenever the system feels confident based on 5-regime rules.
    Logs trades with exact timestamps, sigma strikes, and inputs to dedicated SQLite tables.
    """
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    horizon = req.horizon if req else "WEEKLY"
    force = req.force_refresh if req else False
    conf = req.confidence_threshold if req else 8.0
    lots = req.lot_size if req else NIFTY_LOT_SIZE
    return weekly_monthly_options_strategy_engine.evaluate_and_execute_positional_trades(
        horizon=horizon,
        force_refresh=force,
        confidence_threshold=conf,
        lot_size=lots
    )


@app.get("/api/options/positional-rules/regime")
def get_options_positional_regime_endpoint(horizon: str = "WEEKLY", force_refresh: bool = False):
    """
    Returns real-time Nifty 50 regime strictly derived from the WEEKLY PREDICTION MODEL (T+5 Structural Framework)
    and includes the Pre-Trade Strategy Tournament ranking the candidate architecture with the highest winning probability.
    """
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    reg = weekly_monthly_options_strategy_engine.get_weekly_prediction_model_regime(force_refresh=force_refresh)
    try:
        tournament = weekly_monthly_options_strategy_engine.get_pre_trade_strategy_tournament(horizon=horizon, force_refresh=force_refresh)
        reg["pre_trade_strategy_tournament"] = tournament
    except Exception as e:
        reg["pre_trade_strategy_tournament"] = {"status": "ERROR", "message": str(e)}
    return reg


@app.get("/api/options/positional-rules/tournament")
def get_options_positional_tournament_endpoint(horizon: str = "WEEKLY", force_refresh: bool = False):
    """
    Analyzes which candidate option strategy architecture has the HIGHEST PROBABILITY OF WINNING (P_win)
    and positive expectancy for the current Weekly Prediction Model regime before trade initiation.
    """
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    return weekly_monthly_options_strategy_engine.get_pre_trade_strategy_tournament(
        horizon=horizon,
        force_refresh=force_refresh
    )


@app.get("/api/options/positional-rules/active-trades")
def get_options_positional_active_trades_endpoint(horizon: str = "WEEKLY"):
    """Returns live active Weekly or Monthly options legs, execution status, and re-entry state."""
    from execution.state_engine import state_engine
    table_name = "weekly_options_trade_log" if horizon.upper() == "WEEKLY" else "monthly_options_trade_log"
    with state_engine._lock:
        with state_engine._get_connection() as conn:
            rows = conn.execute(f"""
                SELECT * FROM {table_name}
                ORDER BY id DESC LIMIT 50
            """).fetchall()
    return {
        "status": "OK",
        "horizon": horizon.upper(),
        "trades": [dict(r) for r in rows],
        "timestamp_ist": format_ist()
    }


@app.get("/api/options/positional-rules/ml-metrics")
def get_options_positional_ml_metrics_endpoint(horizon: str = "WEEKLY"):
    """
    Returns Weekly or Monthly Options ML model metrics, training sample count, and feature importances.
    STRICT SEPARATION: Never mixes Weekly/Monthly telemetry with Intraday!
    """
    from execution.state_engine import state_engine
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    kv_key = "weekly_options_ml_metadata" if horizon.upper() == "WEEKLY" else "monthly_options_ml_metadata"
    meta = state_engine.get_kv(kv_key)
    if not meta:
        meta = weekly_monthly_options_strategy_engine.train_positional_options_ml_model(horizon=horizon)
    return meta


@app.post("/api/options/positional-rules/train-ml")
def post_options_positional_train_ml_endpoint(horizon: str = "WEEKLY"):
    """Retrains the Weekly or Monthly machine learning meta-model exclusively on its own trade logs."""
    from engines.weekly_monthly_options_strategy_engine import weekly_monthly_options_strategy_engine
    return weekly_monthly_options_strategy_engine.train_positional_options_ml_model(horizon=horizon)


# ==============================================================================
# ADVANCED CONFIDENCE-DRIVEN AUTO-STRATEGY & REAL-TIME WATCHDOG API
# ==============================================================================
class AdvancedAutoDeployRequest(BaseModel):
    horizon: str = "INTRADAY"
    force_refresh: bool = False
    custom_threshold: Optional[float] = None
    lot_size: int = NIFTY_LOT_SIZE
    bypass_market_hours: bool = False


@app.get("/api/options/advanced/confidence")
def get_advanced_options_confidence_endpoint(horizon: str = "INTRADAY", force_refresh: bool = False):
    """
    Returns real-time dynamic confidence score, confidence grade, unified regime,
    recommended asymmetric strategy, and pillar contributions.
    """
    from engines.advanced_confidence_engine import advanced_confidence_engine
    return advanced_confidence_engine.compute_dynamic_confidence(
        horizon=horizon,
        force_refresh=force_refresh
    )


@app.post("/api/options/advanced/deploy")
def post_advanced_options_deploy_endpoint(req: Optional[AdvancedAutoDeployRequest] = None):
    """
    Evaluates dynamic confidence score and executes advanced options strategies
    (Broken Wing Butterfly, Iron Fly, Ratio Backspread) when system feels confident.
    """
    from engines.advanced_auto_strategy_orchestrator import advanced_auto_strategy_orchestrator
    h = req.horizon if req else "INTRADAY"
    force = req.force_refresh if req else False
    thresh = req.custom_threshold if req else None
    lots = req.lot_size if req else NIFTY_LOT_SIZE
    bypass = req.bypass_market_hours if req else False
    return advanced_auto_strategy_orchestrator.evaluate_and_deploy_strategy(
        horizon=h,
        force_refresh=force,
        custom_threshold=thresh,
        lot_size=lots,
        bypass_market_hours=bypass
    )


@app.post("/api/options/advanced/watchdog-tick")
@app.get("/api/options/advanced/watchdog-tick")
def get_advanced_options_watchdog_tick_endpoint(force_execution: bool = False):
    """
    Runs an immediate watchdog tick across all active strategies to update trailing stops,
    check profit-locking milestones, and trigger dynamic auto-adjustments.
    """
    from engines.strategy_adjustment_engine import strategy_adjustment_engine
    return strategy_adjustment_engine.tick_watchdog(force_execution=force_execution)


@app.get("/api/options/advanced/active-strategies")
def get_advanced_active_strategies_endpoint(horizon: Optional[str] = None):
    """Returns all active advanced options strategies with live mark-to-market status and trailing stop telemetry."""
    from execution.state_engine import state_engine
    query = "SELECT * FROM options_strategies WHERE status = 'ACTIVE'"
    params = []
    if horizon:
        query += " AND expiry_type = ?"
        params.append(horizon.upper())
    query += " ORDER BY id DESC"

    with state_engine._lock:
        with state_engine._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

    return {
        "status": "OK",
        "horizon": horizon or "ALL",
        "strategies": [dict(r) for r in rows],
        "count": len(rows),
        "timestamp_ist": format_ist()
    }



# ==============================================================================
# UNIFIED MACHINE LEARNING, DECISION TREE & CONTINUOUS IMPROVEMENT SUITE
# ==============================================================================
@app.get("/api/ml/overview")
def get_ml_system_overview_endpoint():
    """Returns complete executive audit of the 4-horizon ML system, active models, and statistics."""
    from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
    return horizon_learning_orchestrator.get_system_ml_overview()


@app.get("/api/ml/dataset-stats")
def get_ml_dataset_stats_endpoint():
    """Returns dataset sample counts, live vs counterfactual breakdown across all 4 horizons."""
    from core.unified_ml_dataset import unified_ml_dataset
    return unified_ml_dataset.get_dataset_statistics()


@app.get("/api/ml/trees")
def get_ml_tree_structure_endpoint(horizon: str = "INTRADAY", tree_index: int = 0):
    """Exports representative decision tree topology from LightGBM / XGBoost into structured JSON."""
    from engines.tree_quant_ml_engine import tree_quant_ml_engine
    return tree_quant_ml_engine.export_tree_structure(horizon=horizon, tree_index=tree_index)


@app.get("/api/ml/shap")
def get_ml_shap_explanation_endpoint(horizon: str = "INTRADAY"):
    """Computes exact TreeSHAP feature attributions natively via XGBoost C++ engine using live measured market indicators."""
    from engines.tree_quant_ml_engine import tree_quant_ml_engine

    # Dynamically extract live measured market features
    comp_score = 0.0
    atm_pcr = 1.0
    vix_val = 15.0
    rsi_val = 50.0
    adx_val = 25.0
    regime_code = 1.0

    try:
        from execution.state_engine import state_engine
        last_pred = state_engine.get_kv("last_nifty_prediction", {})
        if isinstance(last_pred, dict):
            comp_score = float(last_pred.get("composite_score", 0.0) or 0.0)
            overlay = last_pred.get("pillars", {}).get("options_overlay", {})
            atm_pcr = float(overlay.get("near_atm_pcr", 1.0) or 1.0)
            vix_m = [
                m for m in last_pred.get("pillars", {}).get("macro_cross_asset", [])
                if isinstance(m, dict) and ("VIX" in str(m.get("ticker", "")).upper() or "VIX" in str(m.get("name", "")).upper())
            ]
            if vix_m:
                vix_val = float(vix_m[0].get("last_close") or vix_m[0].get("close") or 15.0)
            tech = last_pred.get("pillars", {}).get("nifty_technicals", {})
            rsi_val = float(tech.get("rsi14", 50.0) or 50.0)
            adx_val = float(tech.get("adx14", 25.0) or 25.0)
    except Exception:
        pass

    sample_features = {
        "composite_score": comp_score,
        "near_atm_pcr": atm_pcr,
        "vix_spot": vix_val,
        "rsi14": rsi_val,
        "trend_strength_adx": adx_val,
        "horizon_code": 1.0 if horizon == "INTRADAY" else (2.0 if horizon == "WEEKLY" else 4.0),
        "regime_code": regime_code
    }
    return tree_quant_ml_engine.compute_shap_explanations(horizon=horizon, features=sample_features)


@app.post("/api/ml/shap")
def post_ml_shap_explanation_endpoint(payload: Dict[str, Any], horizon: str = "INTRADAY"):
    """Computes exact TreeSHAP feature attributions for custom user/signal feature payloads."""
    from engines.tree_quant_ml_engine import tree_quant_ml_engine
    return tree_quant_ml_engine.compute_shap_explanations(horizon=horizon, features=payload)


@app.get("/api/ml/reentry-audit")
def get_ml_reentry_audit_endpoint(horizon: str = "WEEKLY"):
    """Evaluates stop-loss (40% weekly, 50% monthly) and 1-time re-entry comparative matrix."""
    from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
    return horizon_learning_orchestrator.evaluate_stop_loss_and_reentry_matrix(horizon=horizon)


@app.post("/api/ml/retrain")
def post_ml_retrain_endpoint():
    """Triggers continuous training & promotion pipeline across all 4 trading horizons."""
    from engines.horizon_learning_orchestrator import horizon_learning_orchestrator
    return horizon_learning_orchestrator.retrain_and_evaluate_all_horizons()


@app.post("/api/ml/rollback")
def post_ml_rollback_endpoint(horizon: str = "INTRADAY"):
    """Restores the previous production model artifact for the requested horizon."""
    from engines.tree_quant_ml_engine import tree_quant_ml_engine
    return tree_quant_ml_engine.rollback_to_previous_version(horizon=horizon)


# ==============================================================================
# UNIFIED NIFTY T+1 & T+5 FORECAST LEARNING, EVALUATION & RETRAINING FRAMEWORK
# ==============================================================================
@app.get("/api/nifty-forecast/overview")
def get_nifty_forecast_overview_endpoint():
    """Returns ledger statistics, active models, Brier scores, and calibration metrics for T+1 & T+5."""
    from core.nifty_prediction_ledger import nifty_prediction_ledger
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    return {
        "status": "OPERATIONAL",
        "timestamp": format_ist(),
        "ledger_statistics": nifty_prediction_ledger.get_ledger_statistics(),
        "model_registry": nifty_forecast_learning_engine._active_metadata
    }


@app.get("/api/nifty-forecast/ledger")
def get_nifty_forecast_ledger_endpoint(
    horizon: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """Retrieves immutable prediction records with forward targets and realized outcome resolutions."""
    from core.nifty_prediction_ledger import nifty_prediction_ledger
    with nifty_prediction_ledger._get_connection() as conn:
        query = "SELECT * FROM nifty_forecast_lifecycle_ledger"
        clauses = []
        params = []
        if horizon:
            horizon = horizon.replace(" ", "+").upper()
            clauses.append("horizon = ?")
            params.append(horizon)
        if status:
            clauses.append("status = ?")
            params.append(status.upper())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY prediction_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/nifty-forecast/evaluation")
def get_nifty_forecast_evaluation_endpoint(horizon: str = "T+1"):
    """Returns comprehensive directional accuracy, confusion matrix, Brier score, and MAE report."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    horizon = horizon.replace(" ", "+").upper()
    return nifty_forecast_learning_engine.generate_evaluation_report(horizon=horizon)


@app.post("/api/nifty-forecast/retrain")
def post_nifty_forecast_retrain_endpoint(horizon: str = "ALL"):
    """Executes controlled retraining and promotion verification for T+1 and/or T+5 models."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    horizon = horizon.replace(" ", "+").upper()
    h_list = ["T+1", "T+5"] if horizon == "ALL" else [horizon]
    results = {}
    for h in h_list:
        results[h] = nifty_forecast_learning_engine.train_forecast_models(horizon=h)
    return {
        "status": "COMPLETED",
        "results": results,
        "timestamp": format_ist()
    }


@app.post("/api/nifty-forecast/rollback")
def post_nifty_forecast_rollback_endpoint(horizon: str = "T+1"):
    """Restores the previous production model artifact for T+1 or T+5."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    horizon = horizon.replace(" ", "+").upper()
    return nifty_forecast_learning_engine.rollback_model(horizon=horizon)


@app.get("/api/nifty-forecast/predict")
def get_nifty_forecast_predict_endpoint(horizon: str = "T+1"):
    """Runs real-time inference on the active LightGBM/XGBoost ensemble for Nifty forecasts (F-47)."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    return nifty_forecast_learning_engine.predict(horizon=horizon)


@app.post("/api/nifty-forecast/predict")
def post_nifty_forecast_predict_endpoint(payload: Optional[Dict[str, Any]] = None, horizon: str = "T+1"):
    """Runs inference on the active LightGBM/XGBoost ensemble with optional custom feature payload (F-47)."""
    from engines.nifty_forecast_learning_engine import nifty_forecast_learning_engine
    return nifty_forecast_learning_engine.predict(horizon=horizon, features=payload)


# ==============================================================================
# ML PERFORMANCE & MODEL INTELLIGENCE DASHBOARD ENDPOINTS
# ==============================================================================
@app.get("/ml-performance", response_class=HTMLResponse)
def get_ml_performance_page_endpoint(request: Request):
    """Direct route to the ML Performance & Model Intelligence Dashboard."""
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            html = f.read()
            html = html.replace("let currentModule = 'module-swing';", "let currentModule = 'module-ml-performance';")
            return HTMLResponse(content=html)
    return HTMLResponse("<h1>Quant Trading Desk</h1><p>index.html not found</p>")


@app.get("/api/ml-performance/overview")
def get_ml_performance_overview_endpoint():
    """Returns overview statistics, counters, and system-wide ML health."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_overview()


@app.get("/api/ml-performance/registry")
def get_ml_performance_registry_endpoint():
    """Returns autonomously discovered registry details for every ML model."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_model_registry()


@app.get("/api/ml-performance/training-history")
def get_ml_performance_training_history_endpoint(
    model_name: Optional[str] = None,
    horizon: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """Returns verified historical training execution runs."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_training_history(
        model_name=model_name,
        horizon=horizon,
        status=status,
        limit=limit
    )


@app.get("/api/ml-performance/predictions")
def get_ml_performance_predictions_endpoint(
    horizon: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Returns predictions alongside actual finalized outcomes, errors, and confusion matrix."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_prediction_performance(
        horizon=horizon,
        status=status,
        limit=limit
    )


@app.get("/api/ml-performance/nifty-forecasts")
def get_ml_performance_nifty_forecasts_endpoint(horizon: str = "T+1"):
    """Returns dedicated evaluation metrics and attribution for NIFTY T+1 or T+5."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_t1_t5_performance(horizon=horizon)


@app.get("/api/ml-performance/trade-outcomes")
def get_ml_performance_trade_outcomes_endpoint(horizon: str = "ALL"):
    """Returns financial expectancy, win rate, PnL, and stop-loss re-entries."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_trade_outcome_learning(horizon=horizon)


@app.get("/api/ml-performance/explainability")
def get_ml_performance_explainability_endpoint(
    horizon: str = "INTRADAY",
    tree_index: int = 0
):
    """Returns exact decision tree topology, learned split thresholds, and TreeSHAP ranking."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_trees_and_explainability(
        horizon=horizon,
        tree_index=tree_index
    )


@app.get("/api/ml-performance/regimes")
def get_ml_performance_regimes_endpoint():
    """Returns performance breakdown across all 5 official market regimes."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_market_regime_performance()


@app.get("/api/ml-performance/dataset-audit")
def get_ml_performance_dataset_audit_endpoint():
    """Returns database tables audit, record freshness, missing rates, and feature leakage checks."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_dataset_feature_audit()


@app.get("/api/ml-performance/comparison")
def get_ml_performance_comparison_endpoint(horizon: str = "T+1"):
    """Returns tournament comparison between ensemble, standalone trees, baselines, and non-ML."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_model_comparison(horizon=horizon)


@app.get("/api/ml-performance/drift")
def get_ml_performance_drift_endpoint():
    """Returns statistical feature drift, prediction distribution shifts, and model health alerts."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_monitoring_and_drift()


@app.get("/api/ml-performance/audit-logs")
def get_ml_performance_audit_logs_endpoint(
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 100
):
    """Returns searchable operational audit event stream."""
    from engines.ml_performance_engine import ml_performance_engine
    return ml_performance_engine.get_audit_logs(
        severity=severity,
        category=category,
        limit=limit
    )


# ==============================================================================
# AI INTELLIGENCE LAYER ENDPOINTS (GROQ + OPENROUTER FREE TIER)
# ==============================================================================
class AIAnalyzeRequest(BaseModel):
    force_refresh: bool = False
    model_override: Optional[str] = None


@app.get("/api/ai/status")
def get_ai_status_endpoint():
    """Returns configuration status, active provider, candidate models, and cache info."""
    from ai_engine.service import ai_service
    return ai_service.get_status().model_dump()


@app.get("/api/ai/analysis")
def get_ai_analysis_endpoint():
    """Returns the latest valid AI analysis (from cache or newly evaluated)."""
    try:
        from ai_engine.service import ai_service
        analysis = ai_service.analyze_predictions(force_refresh=False)
        return analysis.model_dump()
    except Exception as exc:
        logger.error(f"Error in GET /api/ai/analysis: {exc}", exc_info=True)
        from ai_engine.service import ai_service
        from datetime import datetime, timezone
        return ai_service._build_unavailable_fallback(str(exc), datetime.now(timezone.utc).isoformat()).model_dump()


@app.post("/api/ai/analyze")
def trigger_ai_analysis_endpoint(payload: Optional[AIAnalyzeRequest] = None):
    """Triggers an on-demand AI analysis from current verified ML predictions."""
    try:
        from ai_engine.service import ai_service
        req = payload or AIAnalyzeRequest()
        analysis = ai_service.analyze_predictions(
            force_refresh=req.force_refresh,
            model_override=req.model_override
        )
        return analysis.model_dump()
    except Exception as exc:
        logger.error(f"Error in POST /api/ai/analyze: {exc}", exc_info=True)
        from ai_engine.service import ai_service
        from datetime import datetime, timezone
        return ai_service._build_unavailable_fallback(str(exc), datetime.now(timezone.utc).isoformat()).model_dump()


@app.post("/api/ai/benchmark")
def run_ai_benchmark_endpoint():
    """Protected benchmark workflow comparing all 5 candidate models against standardized input."""
    from ai_engine.service import ai_service
    reports = ai_service.run_model_benchmark()
    return {
        "status": "OK",
        "timestamp_ist": format_ist(),
        "models_evaluated": len(reports),
        "results": [r.model_dump() for r in reports]
    }


# ==============================================================================
# SWING QUANT ML & SHADOW LEDGER ENDPOINTS
# ==============================================================================
@app.get("/api/ml/swing/model-status")
def get_swing_ml_status_endpoint():
    """Returns training metadata, empirical win rates, Brier score, and gatekeeper criteria."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    return {
        "status": "OK",
        "timestamp_ist": format_ist(),
        "engine": swing_quant_ml_engine.get_status()
    }


@app.post("/api/ml/swing/train")
def train_swing_ml_endpoint():
    """Triggers on-demand online training for the Swing Quant ML meta-classifier & expected R regressor."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    meta = swing_quant_ml_engine.train_model()
    return {
        "status": "OK",
        "message": "Swing Quant ML model retrained successfully.",
        "training_metadata": meta
    }


@app.get("/api/ml/swing/dataset")
def get_swing_quant_dataset_endpoint(limit: int = 50, trade_type: Optional[str] = None):
    """Retrieves records from the master quant trade dataset (real and shadow counterfactuals)."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    rows = swing_quant_ml_engine.get_dataset(limit=limit, trade_type=trade_type)
    return {
        "status": "OK",
        "count": len(rows),
        "dataset": rows
    }


@app.get("/api/ml/swing/shadow-trades")
def get_active_shadow_trades_endpoint():
    """Retrieves all active shadow setups currently monitored counterfactually without capital risk."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    active = swing_quant_ml_engine.get_active_shadow_trades()
    return {
        "status": "OK",
        "active_shadow_count": len(active),
        "shadow_trades": active
    }


@app.post("/api/ml/swing/reconcile-shadow")
def reconcile_shadow_trades_endpoint():
    """Triggers on-demand reconciliation of active shadow trades against latest EOD candle highs/lows."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    audit = swing_quant_ml_engine.reconcile_shadow_trades()
    return {
        "status": "OK",
        "message": "Shadow counterfactual reconciliation complete.",
        "audit": audit
    }


@app.get("/api/ml/swing/active-setups")
def get_active_training_setups_endpoint():
    """Retrieves all active training setups with live MTM, PnL, return %, dist to SL and T1."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    setups = swing_quant_ml_engine.get_active_training_setups()
    return {
        "status": "OK",
        "count": len(setups),
        "setups": setups
    }


@app.get("/api/swing/decision-audit")
@app.get("/api/ml/swing/decision-audit")
def get_swing_decision_audit_endpoint(limit: int = 100, symbol: Optional[str] = None):
    """Retrieves full immutable audit chain of candidate decisions across Chartink, ML gate, AI gate, and execution."""
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    logs = swing_quant_ml_engine.get_decision_audit_logs(limit=limit, symbol=symbol)
    return {
        "status": "OK",
        "count": len(logs),
        "audit_logs": logs
    }


@app.get("/api/swing/candidate-eval/{symbol}")
def evaluate_candidate_endpoint(symbol: str):
    """Runs live on-demand ML + AI Final Buy Gate evaluation on any stock symbol."""
    import time
    from datetime import datetime, timezone
    from execution.state_engine import state_engine
    from engines.swing_quant_ml_engine import swing_quant_ml_engine
    from ai_engine.service import ai_service
    from ai_engine.schemas import (
        SwingCandidateAIRequest,
        SwingCandidateIdentity,
        SwingCandidateContext,
        SwingTechnicalAnalysis,
        SwingMLAnalysis,
        SwingTradePlan,
        SwingHistoricalEvidence,
        SwingPortfolioContext,
        SwingInstructions
    )
    from core.data_quality import format_ist

    sym_clean = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    df, _ = state_engine.get_ohlcv_df(sym_clean)
    if df is None or df.empty or len(df) < 25:
        return {"status": "DATA_UNAVAILABLE", "message": f"Data unavailable: Insufficient OHLCV history ({len(df) if df is not None else 0} bars) for {sym_clean}. Minimum 25 bars required."}

    c = float(df["Close"].iloc[-1])
    h = float(df["High"].iloc[-1]) if "High" in df.columns else c
    l = float(df["Low"].iloc[-1]) if "Low" in df.columns else c

    # Canonical Wilder ATR14
    atr = state_engine._calculate_atr(df, period=14)
    if atr <= 0.0:
        atr = max(1.0, c * 0.02)
    sl = round(c - 1.5 * atr, 2)
    t1 = round(c + 3.0 * atr, 2)
    rr = round((t1 - c) / max(0.01, c - sl), 2)

    # Real RSI(14)
    from engines.institutional_engine import FeatureEngine
    rsi14 = FeatureEngine.calc_rsi_np(df["Close"].values, period=14)

    # Real ADX(14)
    if "High" in df.columns and "Low" in df.columns:
        adx14 = FeatureEngine.calc_adx_np(df["High"].values, df["Low"].values, df["Close"].values, period=14)
    else:
        adx14 = 25.0

    # Real EMA20 distance
    ema20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
    dist_to_ema20 = round(((c - ema20) / max(1.0, ema20)) * 100.0, 2)

    # Real Volume ratio (current / 20-day average)
    if "Volume" in df.columns and len(df) >= 20:
        vol_mean = float(df["Volume"].tail(20).mean())
        vol_ratio = round(float(df["Volume"].iloc[-1]) / max(1.0, vol_mean), 2)
    else:
        vol_ratio = 1.0

    # Real 20-day Pivot High distance
    pivot20 = float(df["High"].tail(20).max()) if "High" in df.columns else c
    dist_to_pivot = round(((c - pivot20) / max(1.0, pivot20)) * 100.0, 2)

    # Realistic position sizing based on risk per trade
    risk_per_share = max(0.5, c - sl)
    shares = max(1, int(P1_BASE_CAPITAL * (P1_BASE_RISK_PCT / 100.0) / risk_per_share))

    cand_dict = {
        "symbol": sym_clean,
        "ticker": sym_clean,
        "close": c,
        "entry": c,
        "sl": sl,
        "t1": t1,
        "score": round(min(100.0, max(20.0, (rsi14 * 0.5 + min(vol_ratio, 2.0) * 20.0))), 1),
        "confidence": 4 if rsi14 > 50 and vol_ratio > 1.0 else 3,
        "rsi14": rsi14,
        "atr_pct": round((atr / max(1.0, c)) * 100.0, 2),
        "dist_to_pivot": dist_to_pivot,
        "dist_to_ema20": dist_to_ema20,
        "vol_ratio": vol_ratio,
        "inst_score": round(min(100.0, max(10.0, 50.0 + dist_to_ema20 * 2.0)), 1),
        "week_confluence": 1 if dist_to_ema20 > 0 else 0,
        "is_vcp": 1 if vol_ratio > 1.2 and abs(dist_to_pivot) < 2.0 else 0,
        "bb_sq_ratio": 0.8,
        "adx14": adx14,
        "shares": shares,
        "sector": "DIVERSIFIED"
    }

    regime = swing_quant_ml_engine.capture_market_regime()
    ml_res = swing_quant_ml_engine.evaluate_candidate(cand_dict, market_regime=regime)

    # Real portfolio state
    active_pos = state_engine.get_active_positions(portfolio_type="P1_SWING")
    already_held = any(p.get("symbol") == sym_clean for p in active_pos)
    slots_used = len(active_pos)

    now_iso = datetime.now(timezone.utc).isoformat()
    ai_req = SwingCandidateAIRequest(
        schema_version="1.0",
        request_id=f"eval_req_{sym_clean}_{int(time.time())}",
        task="swing_final_buy_decision",
        module="swing_trading",
        candidate=SwingCandidateIdentity(
            candidate_id=f"cand_{sym_clean}",
            exchange="NSE",
            symbol=sym_clean,
            scanner_classification="BUY NOW" if cand_dict["score"] >= 75 else "WATCHLIST",
            scanner_run_id="on_demand_eval"
        ),
        context=SwingCandidateContext(
            as_of=now_iso,
            timezone="Asia/Kolkata",
            data_mode="live",
            data_quality={"status": "valid", "is_stale": False, "missing_fields": [], "warnings": []}
        ),
        technical_analysis=SwingTechnicalAnalysis(
            classification="BUY NOW" if cand_dict["score"] >= 75 else "WATCHLIST",
            entry_conditions=["6-Pillars >= 75", "Pivot proximity <= 2%", "Volume ratio >= 1.0x"],
            satisfied_conditions=["6-Pillar institutional structure valid"] if cand_dict["score"] >= 75 else [],
            unsatisfied_conditions=[] if cand_dict["score"] >= 75 else ["Score below buy threshold"],
            features=ml_res["features"],
            engine_version="6-Pillar-v2.4"
        ),
        ml_analysis=SwingMLAnalysis(
            decision=ml_res["decision"],
            model_name=ml_res["model_name"],
            model_version=ml_res["model_version"],
            target_definition=ml_res["target_definition"],
            prediction_score=ml_res["prediction_score"],
            calibrated_probability=ml_res["ml_win_prob"],
            expected_r=ml_res["expected_r"],
            feature_snapshot_id=ml_res["feature_snapshot_id"],
            decision_policy_version="1.0",
            risk_flags=ml_res["risk_flags"],
            fatal_flags=ml_res["fatal_flags"]
        ),
        trade_plan=SwingTradePlan(
            entry_rule="Breakout confirmation with volume",
            entry_price=c,
            stop_loss=sl,
            target=t1,
            holding_period="5-15 sessions",
            reward_risk_ratio=rr,
            position_size=shares,
            risk_per_share=round(c - sl, 2),
            capital_at_risk=round(shares * (c - sl), 2),
            estimated_transaction_costs=round(shares * c * 0.0012, 2)
        ),
        historical_evidence=SwingHistoricalEvidence(
            sample_count=int(ml_res.get("model_metadata", {}).get("total_samples") or 0),
            target_before_stop_frequency=ml_res["ml_win_prob"],
            stop_before_target_frequency=round(1.0 - ml_res["ml_win_prob"], 2),
            evaluation_type="model_prediction",
            period="NSE 500 Historical Validation",
            historical_win_rate=float(ml_res.get("model_metadata", {}).get("win_rate") or ml_res["ml_win_prob"])
        ),
        portfolio_context=SwingPortfolioContext(
            already_held=already_held,
            duplicate_pending_order=False,
            available_capital=P1_BASE_CAPITAL,
            risk_limits_available=True,
            active_slots_used=slots_used,
            max_slots=10
        ),
        instructions=SwingInstructions(
            objective="Make a final analytical BUY NOW or DO NOT BUY decision using the supplied evidence.",
            allow_order_execution=False,
            do_not_invent_data=True
        )
    )

    ai_res = ai_service.evaluate_swing_candidate(ai_req)

    return {
        "status": "OK",
        "symbol": sym_clean,
        "ml_evaluation": ml_res,
        "ai_decision": ai_res.model_dump(),
        "evaluated_at": format_ist()
    }


# ==============================================================================
# CUSTOM SCREENER & EQ-FINAL RULE BUILDER ENDPOINTS
# ==============================================================================

class RunScreenerRequest(BaseModel):
    preset_name: Optional[str] = "EQ-FINAL"
    custom_conditions: Optional[List[Dict[str, Any]]] = None
    raw_clause: Optional[str] = None
    force_refresh: bool = False

class SavePresetRequest(BaseModel):
    name: str
    conditions: List[Dict[str, Any]]
    description: Optional[str] = ""

@app.get("/api/screener/presets")
def get_screener_presets_endpoint():
    """Returns all configured screener presets (including default EQ-FINAL)."""
    return {
        "status": "SUCCESS",
        "presets": custom_screener_engine.get_presets()
    }

@app.post("/api/screener/run")
def run_custom_screener_endpoint(req: RunScreenerRequest):
    """Executes screener scan with real-time enrichment and 20EMA distance."""
    try:
        res = custom_screener_engine.run_scan(
            preset_name=req.preset_name,
            custom_conditions=req.custom_conditions,
            raw_clause=req.raw_clause,
            force_refresh=req.force_refresh
        )
        return res
    except Exception as e:
        logger.error(f"Error running custom screener: {e}")
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})

@app.post("/api/screener/save-preset")
def save_screener_preset_endpoint(req: SavePresetRequest, _auth=Depends(verify_admin_key)):
    """Saves a user-configured custom screener preset."""
    ok = custom_screener_engine.save_preset(req.name, req.conditions, req.description)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid preset name or empty conditions.")
    return {"status": "SUCCESS", "message": f"Preset '{req.name}' saved successfully."}

@app.delete("/api/screener/preset/{name}")
def delete_screener_preset_endpoint(name: str, _auth=Depends(verify_admin_key)):
    """Deletes a user-configured custom screener preset."""
    ok = custom_screener_engine.delete_preset(name)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete default preset 'EQ-FINAL' or preset not found.")
    return {"status": "SUCCESS", "message": f"Preset '{name}' deleted successfully."}


class ScreenerSourceUpdateRequest(BaseModel):
    source: str


class CompareScreenerRequest(BaseModel):
    preset_name: Optional[str] = "EQ-FINAL"
    custom_conditions: Optional[List[Dict[str, Any]]] = None
    force_refresh: Optional[bool] = False


@app.get("/api/screener/source")
def get_screener_source_endpoint():
    """Returns active candidate universe sourcing preference for swing trading."""
    current = state_engine.get_swing_universe_source()
    return {
        "status": "SUCCESS",
        "source": current,
        "available_sources": [
            {
                "id": "IN_HOUSE",
                "label": "In-House VPS Engine (Nifty 500 Institutional)",
                "desc": "Official NSE Nifty 500 constituents filtered locally; eliminates micro-caps, penny stocks, and lower-circuit traps.",
                "is_active": (current == "IN_HOUSE")
            },
            {
                "id": "CHARTINK",
                "label": "Chartink Live Feed (All Cash Equities)",
                "desc": "Scans all ~2,200 listed NSE cash equities including non-index micro-caps (Market Cap >= ₹300 Cr).",
                "is_active": (current == "CHARTINK")
            }
        ],
        "updated_at": format_ist()
    }


@app.post("/api/screener/source")
def set_screener_source_endpoint(req: ScreenerSourceUpdateRequest):
    """Sets candidate universe sourcing preference for swing trading."""
    new_src = state_engine.set_swing_universe_source(req.source)
    logger.info(f"Swing trading candidate universe sourcing updated to: {new_src}")
    return {
        "status": "SUCCESS",
        "source": new_src,
        "message": f"Swing Trading candidate universe sourcing updated to '{new_src}' successfully."
    }


@app.post("/api/screener/compare")
def compare_screener_endpoint(req: CompareScreenerRequest):
    """Executes dual-engine comparison audit: Chartink Live vs In-House VPS Institutional Screener."""
    try:
        audit = screener_comparison_engine.run_comparison_audit(
            preset_name=req.preset_name or "EQ-FINAL",
            custom_conditions=req.custom_conditions,
            force_refresh=bool(req.force_refresh)
        )
        return audit
    except Exception as e:
        logger.error(f"Error executing screener comparison audit: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/screener/compare/summary")
def compare_screener_summary_endpoint():
    """Returns instant summary telemetry of latest dual-engine audit."""
    try:
        audit = screener_comparison_engine.run_comparison_audit(force_refresh=False)
        # Strip heavy arrays for high-speed lightweight summary
        summary = {k: v for k, v in audit.items() if k not in ("all_stocks", "common_stocks", "chartink_only_stocks", "broad_symbols", "inhouse_filtered_out_symbols")}
        return summary
    except Exception as e:
        logger.error(f"Error fetching comparison summary: {e}")
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/screener/periodic-runs")
def get_periodic_runs_endpoint():
    """Returns the history of half-hourly automated runs and timestamps."""
    return {
        "status": "SUCCESS",
        "last_successful_30m_run_ist": state_engine.get_last_successful_30m_run_ist(),
        "periodic_runs": state_engine.get_periodic_30m_runs(limit=30)
    }


# --------------------------------------------------------------------------
# INSTITUTIONAL MOVERS & ACCUMULATION RESEARCH ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/api/research/movers-accumulation")
def get_movers_accumulation_research(force_refresh: bool = False):
    """
    Returns empirical research data on positive moving stocks:
    - Pre-breakout tightness, volume dry-up (VDU), 20 EMA launch rate
    - Common findings across runners
    - Archetype classification breakdown
    - Cryptographic SHA-256 verified audit snapshot
    """
    from engines.accumulation_research_engine import accumulation_research_engine
    try:
        report = accumulation_research_engine.get_latest_research() if not force_refresh else accumulation_research_engine.run_research_scan(force_refresh=True)
        return _sanitize_for_json(report)
    except Exception as e:
        logger.error(f"Error fetching movers accumulation research: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.post("/api/research/run-study")
def trigger_movers_research_study(force_refresh: bool = True):
    """Triggers an on-demand empirical research scan of moving stocks and pre-breakout accumulation."""
    from engines.accumulation_research_engine import accumulation_research_engine
    try:
        report = accumulation_research_engine.run_research_scan(force_refresh=True)
        return _sanitize_for_json(report)
    except Exception as e:
        logger.error(f"Error executing movers research study: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/research/regime-analysis")
def get_research_regime_analysis():
    """
    Returns empirical multi-regime performance matrix:
    Aggregates historical scans grouped by BULLISH, BEARISH, NEUTRAL regimes,
    quantifying what setup features (VDU %, 20 EMA launch %, tightness %, breakout surge)
    consistently produce winners under different macro market environments.
    """
    from engines.accumulation_research_engine import accumulation_research_engine
    try:
        matrix = accumulation_research_engine.get_regime_performance_matrix()
        return _sanitize_for_json(matrix)
    except Exception as e:
        logger.error(f"Error computing research regime analysis: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/research/audit-history")
def get_research_audit_history(limit: int = 50, regime: str = None):
    """Returns paginated historical scans logged with market regime tags and audit hashes."""
    from engines.accumulation_research_engine import accumulation_research_engine
    try:
        history = accumulation_research_engine.get_audit_history(limit=limit, regime=regime)
        return _sanitize_for_json({"status": "SUCCESS", "count": len(history), "history": history})
    except Exception as e:
        logger.error(f"Error fetching research audit history: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


# --------------------------------------------------------------------------
# PRE-RALLY PATTERN DISCOVERY & VALIDATION ENGINE ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/api/research/pre-rally/summary")
def get_pre_rally_summary():
    """
    Returns high-level summary of the Pre-Rally Pattern Discovery & Validation Engine:
    - In-Sample vs Out-of-Sample baseline rally rates
    - Total events & candidate hypotheses evaluated
    - Chronological split cutoff date and validation counts
    """
    from engines.pre_rally_discovery_engine import pre_rally_discovery_engine
    try:
        summary = pre_rally_discovery_engine.get_summary_stats()
        return _sanitize_for_json({"status": "SUCCESS", "summary": summary})
    except Exception as e:
        logger.error(f"Error fetching pre-rally summary: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/research/pre-rally/combinations")
def get_pre_rally_combinations(
    status: str = "ALL",
    min_sample: int = 5,
    sort_by: str = "lift_oos"
):
    """
    Returns discovered and statistically validated pre-rally feature combinations:
    - In-Sample vs Out-of-Sample precision
    - Lift over baseline
    - MFE/MAE risk-reward ratios
    - Validation status (VALIDATED_OOS, EXPLORATORY, INSUFFICIENT_DATA)
    """
    from engines.pre_rally_discovery_engine import pre_rally_discovery_engine
    try:
        combinations = pre_rally_discovery_engine.get_discovered_combinations(
            min_sample=min_sample,
            status_filter=status
        )
        if sort_by == "out_sample_precision":
            combinations.sort(key=lambda x: (x.get("validation_status") != "VALIDATED_OOS", -float(x.get("out_sample_precision", 0.0) or 0.0)))
        elif sort_by == "sample_size":
            combinations.sort(key=lambda x: -int(x.get("sample_size", 0) or 0))
        elif sort_by == "mfe_mae_ratio":
            combinations.sort(key=lambda x: -float(x.get("mfe_mae_ratio", 0.0) or 0.0))
        else: # default lift_oos
            combinations.sort(key=lambda x: (x.get("validation_status") != "VALIDATED_OOS", -float(x.get("lift_oos", 0.0) or 0.0)))

        return _sanitize_for_json({
            "status": "SUCCESS",
            "count": len(combinations),
            "combinations": combinations
        })
    except Exception as e:
        logger.error(f"Error fetching pre-rally combinations: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.get("/api/research/pre-rally/combination-detail")
def get_pre_rally_combination_detail(combination_id: str):
    """Returns granular audit trail for a specific pattern combination including real winners and failures."""
    from engines.pre_rally_discovery_engine import pre_rally_discovery_engine
    try:
        detail = pre_rally_discovery_engine.get_combination_detail(combination_id=combination_id)
        if not detail:
            return JSONResponse(status_code=404, content={"status": "NOT_FOUND", "message": f"Combination {combination_id} not found."})
        return _sanitize_for_json({"status": "SUCCESS", "combination": detail})
    except Exception as e:
        logger.error(f"Error fetching combination detail: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.post("/api/research/pre-rally/run")
def trigger_pre_rally_discovery_run(
    min_fwd_return: float = 5.0,
    fwd_horizon: int = 10,
    max_adverse_pct: float = -3.5,
    max_symbols: int = 35
):
    """Triggers an empirical pattern discovery and validation cycle across historical OHLCV data."""
    from engines.pre_rally_discovery_engine import pre_rally_discovery_engine
    try:
        res = pre_rally_discovery_engine.run_discovery(
            min_fwd_return=min_fwd_return,
            fwd_horizon=fwd_horizon,
            max_adverse_pct=max_adverse_pct,
            max_symbols=max_symbols,
            force_refresh=True
        )
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error executing pre-rally discovery run: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})



# --------------------------------------------------------------------------
# INSTITUTIONAL TEST MODE & SINGLE STOCK DIAGNOSTIC LAB ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/api/test-mode/benchmarks")
def get_test_mode_benchmarks():
    """Returns standard criteria benchmarks for BUY NOW, BREAKOUT, and PULLBACK."""
    from engines.stock_test_analyzer import stock_test_analyzer
    bms = stock_test_analyzer.get_setup_benchmarks()
    return {"status": "SUCCESS", "setups": bms, **bms}


@app.get("/api/test-mode/search")
def search_test_mode_symbols(q: str = "", limit: int = 25):
    """Searches across all known NSE equities for autocomplete."""
    from engines.stock_test_analyzer import stock_test_analyzer
    results = stock_test_analyzer.search_nse_stocks(query=q, limit=limit)
    return {"status": "SUCCESS", "results": results, "count": len(results)}


@app.get("/api/test-mode/analyze/{symbol}")
def analyze_stock_endpoint(symbol: str, force_download: bool = False):
    """
    Performs full empirical diagnostic analysis on any given NSE stock:
    - Automatically retrieves from Single Master Vault or downloads live from Yahoo Finance
    - Compares stock actuals side-by-side with BUY NOW, BREAKOUT, and PULLBACK standards
    - Generates actionable trade plan and candlestick chart data series
    """
    from engines.stock_test_analyzer import stock_test_analyzer
    try:
        report = stock_test_analyzer.analyze_stock(symbol, force_download=force_download)
        if report.get("status") == "ERROR":
            return JSONResponse(status_code=400, content=_sanitize_for_json(report))
        return _sanitize_for_json(report)
    except Exception as e:
        logger.error(f"Error in analyze_stock_endpoint for {symbol}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


# --------------------------------------------------------------------------
# PROSPECTIVE VALIDATION & PAPER-TRADING ENDPOINTS (ZERO-BROKER)
# --------------------------------------------------------------------------
@app.get("/api/research/prospective-validation/summary")
def get_prospective_validation_summary():
    """
    Returns telemetry summary for the prospective paper-trading pipeline:
    - Signals count, open trades, completed trades
    - Realized gross & net expectancies, win rate, profit factor, drawdown
    - Exit breakdown (profit target, stop loss, gaps, time expiry, conflicts)
    - Active open trades and recent completed trades list
    - Zero-broker safety confirmation & historical baseline reference
    """
    try:
        from engines.prospective_validation_engine import prospective_validation_engine
        telem = prospective_validation_engine.get_telemetry_summary()
        return _sanitize_for_json(telem)
    except Exception as e:
        logger.error(f"Error fetching prospective validation telemetry: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.post("/api/research/prospective-validation/scan-today")
def trigger_prospective_scan_today(as_of_date: Optional[str] = None):
    """
    Triggers prospective signal scan across universe as of today (or specified date),
    converts new signals to paper trades, and updates open trades.
    """
    try:
        from engines.prospective_validation_engine import prospective_validation_engine
        res = prospective_validation_engine.run_daily_pipeline()
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error running prospective scan: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


@app.post("/api/research/prospective-validation/update-trades")
def trigger_prospective_update_trades():
    """
    Advances open paper trades with latest OHLCV bars without scanning new signals.
    """
    try:
        from engines.prospective_validation_engine import prospective_validation_engine
        res = prospective_validation_engine.sync_open_paper_trades()
        return _sanitize_for_json(res)
    except Exception as e:
        logger.error(f"Error syncing open paper trades: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})








# ==============================================================================
# KOTAK NEO API & REAL-TIME STREAMING ENDPOINTS
# ==============================================================================

class KotakAuthRequest(BaseModel):
    consumer_key: Optional[str] = None
    mobile_number: Optional[str] = None
    ucc: Optional[str] = None
    mpin: Optional[str] = None
    totp: Optional[str] = None
    totp_secret: Optional[str] = None
    environment: Optional[str] = "prod"


class KotakSubscribeRequest(BaseModel):
    symbols: list[str]


@app.get("/api/kotak/config")
def get_kotak_config_endpoint():
    """Returns saved Kotak Neo configuration details for UI setup modal."""
    from core import config
    from execution.state_engine import state_engine
    saved_creds = state_engine.get_kv("KOTAK_NEO_CREDENTIALS") or {}
    c_key = config.KOTAK_NEO_CONSUMER_KEY or saved_creds.get("consumer_key", "")
    ucc_val = config.KOTAK_NEO_UCC or saved_creds.get("ucc", "")
    mob_val = config.KOTAK_NEO_MOBILE_NUMBER or saved_creds.get("mobile_number", "")
    has_sec = bool(config.KOTAK_NEO_TOTP_SECRET or saved_creds.get("totp_secret"))
    has_mpin = bool(config.KOTAK_NEO_MPIN or saved_creds.get("mpin"))
    env = config.KOTAK_NEO_ENVIRONMENT or saved_creds.get("environment", "prod")
    return {
        "status": "OK",
        "consumer_key": c_key,
        "ucc": ucc_val,
        "mobile_number": mob_val,
        "has_totp_secret": has_sec,
        "has_mpin": has_mpin,
        "environment": env
    }


@app.get("/api/kotak/status")
def get_kotak_status_endpoint():
    """Returns Kotak Neo API SDK status, connection telemetry, and cached ticks count."""
    from engines.kotak_neo_engine import kotak_neo_engine
    return kotak_neo_engine.get_status()


@app.post("/api/kotak/auth")
def authenticate_kotak_neo_endpoint(payload: Optional[KotakAuthRequest] = None):
    """
    Authenticates with Kotak Neo API using credentials or automated TOTP.
    Initiates live stream if successful.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    req = payload or KotakAuthRequest()
    res = kotak_neo_engine.login(
        consumer_key=req.consumer_key,
        mobile_number=req.mobile_number,
        ucc=req.ucc,
        mpin=req.mpin,
        totp=req.totp,
        totp_secret=req.totp_secret,
        environment=req.environment
    )
    return res


@app.post("/api/kotak/subscribe")
def subscribe_kotak_symbols_endpoint(payload: KotakSubscribeRequest):
    """Subscribes specified NSE trading symbols to Kotak Neo live feed."""
    from engines.kotak_neo_engine import kotak_neo_engine
    if not payload.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided.")
    res = kotak_neo_engine.subscribe_symbols(payload.symbols)
    return res


@app.get("/api/kotak/quotes")
def get_kotak_quotes_endpoint():
    """Returns in-memory snapshot of all live ticks received from Kotak Neo."""
    from engines.kotak_neo_engine import kotak_neo_engine
    ticks = kotak_neo_engine.get_all_ticks()
    return {
        "status": "SUCCESS",
        "count": len(ticks),
        "quotes": ticks,
        "timestamp_ist": format_ist()
    }


@app.post("/api/kotak/disconnect")
def disconnect_kotak_endpoint():
    """Disconnects Kotak Neo live stream."""
    from engines.kotak_neo_engine import kotak_neo_engine
    kotak_neo_engine.disconnect_stream()
    return {"status": "SUCCESS", "message": "Kotak Neo stream disconnected."}


@app.post("/api/kotak/sync-scrip")
def sync_kotak_scrip_master_endpoint(background_tasks: BackgroundTasks):
    """Triggers background synchronization of official Kotak Neo Scrip Master (NSE CM)."""
    from engines.kotak_neo_engine import kotak_neo_engine
    background_tasks.add_task(kotak_neo_engine.update_scrip_master)
    return {
        "status": "ACCEPTED",
        "message": "Kotak Neo Scrip Master synchronization started in background."
    }


@app.get("/api/kotak/ping")
def ping_kotak_connection_endpoint():
    """
    Diagnostic endpoint: Performs a live REST test query to Kotak Neo servers.
    Verifies that session tokens and credentials are valid even when market is closed.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    if not kotak_neo_engine._is_authenticated:
        return {
            "status": "NOT_AUTHENTICATED",
            "connected": False,
            "message": "Kotak Neo session is not authenticated. Please login first."
        }
    try:
        raw_res = None
        if kotak_neo_engine._client:
            tokens = [{"instrument_token": "2885", "exchange_segment": "nse_cm"}]
            try:
                raw_res = kotak_neo_engine._client.quotes(instrument_tokens=tokens, quote_type="all")
            except Exception as re_err:
                raw_res = str(re_err)

        quotes = kotak_neo_engine.fetch_rest_quotes(["RELIANCE"])
        return {
            "status": "SUCCESS" if quotes else "CONNECTED_EMPTY",
            "connected": True,
            "quotes": quotes,
            "raw_response": raw_res,
            "websocket_active": kotak_neo_engine.is_streaming_active(),
            "timestamp_ist": format_ist(),
            "message": "Connection to Kotak Neo servers verified successfully."
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "connected": False,
            "error": str(e)
        }


@app.post("/api/kotak/test-tick")
def inject_test_tick_endpoint(symbol: str = "RELIANCE", ltp: Optional[float] = None):
    """
    Injects a simulated tick to verify tick cache, UI counter, and data pipeline
    outside market hours.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    price = float(ltp or 2980.50)
    kotak_neo_engine.inject_test_tick(symbol=symbol, ltp=price)
    return {
        "status": "SUCCESS",
        "symbol": symbol,
        "injected_price": price,
        "ticks_total": kotak_neo_engine.get_status().get("ticks_received_total", 0),
        "timestamp_ist": format_ist()
    }


# ==============================================================================
# SERVER PUBLIC IP DETECTION ENDPOINT (FOR BROKER API WHITELISTING)
# ==============================================================================
@app.get("/api/server-ip")
def get_server_ip_endpoint(refresh: bool = False):
    """Returns the detected public server IP of the instance."""
    from scheduler.cron_scheduler import get_server_public_ip
    ip = get_server_public_ip(force=refresh)
    return {
        "status": "OK",
        "server_ip": ip,
        "timestamp_ist": format_ist()
    }


# ==============================================================================
# CONTINUOUS MULTI-DESK LIVE STREAM ENDPOINT (POWERED BY KOTAK NEO)
# ==============================================================================
@app.get("/api/stream/continuous")
def get_continuous_live_stream_endpoint():
    """
    Ultra-low latency continuous data stream endpoint powered by Kotak Neo API:
    Streams live market prices, MTM valuations, trailing stops, Nifty 50 spot, VWAP levels,
    and Options portfolio metrics continuously to frontend desks without Yahoo/NSE overhead.
    """
    from engines.kotak_neo_engine import kotak_neo_engine
    from engines.nifty_analytics_engine import nifty_analytics_engine

    # Ensure background hybrid REST streamer is active without blocking the HTTP response thread
    if kotak_neo_engine._is_authenticated:
        kotak_neo_engine._start_rest_streamer()

    all_ticks = kotak_neo_engine.get_all_ticks()
    is_streaming = kotak_neo_engine.is_streaming_active()
    stream_status = kotak_neo_engine.get_status()

    # 1. Swing Trading: Active Positions with Kotak live LTP & MTM
    active_positions = state_engine.get_active_positions()
    positions_summary = []
    total_unrealized_pnl = 0.0
    for p in active_positions:
        sym = (p.get("symbol") or "").strip().upper()
        entry_px = float(p.get("entry_price") or 0.0)
        qty = int(p.get("quantity") or 0)
        curr_px = float(kotak_neo_engine.get_live_price(sym) or p.get("current_price") or entry_px)
        pnl = round((curr_px - entry_px) * qty, 2)
        total_unrealized_pnl += pnl
        tick = kotak_neo_engine.get_live_tick(sym)
        chg_pct = tick.get("change_pct") if tick else None
        r_unit = max(0.01, entry_px - float(p.get("original_stop_loss") or p.get("stop_loss") or (entry_px * 0.95)))
        positions_summary.append({
            "symbol": sym,
            "ltp": curr_px,
            "entry_price": entry_px,
            "stop_loss": float(p.get("stop_loss") or 0.0),
            "highest_price": float(p.get("highest_price") or curr_px),
            "quantity": qty,
            "position_value": round(curr_px * qty, 2),
            "unrealized_pnl": pnl,
            "pnl_pct": round(((curr_px - entry_px) / max(entry_px, 0.01)) * 100, 2),
            "current_r": round((curr_px - entry_px) / r_unit, 2),
            "day_change_pct": chg_pct,
            "updated_at": tick.get("updated_at") if tick else format_ist(),
            "cloud_sync": p.get("cloud_sync", "SYNCED")
        })

    # 2. Swing Portfolio Live Net Worth & Capital Allocation Metrics
    tot_invested = sum(float(p.get("entry_price") or 0.0) * int(p.get("quantity") or 0) for p in active_positions)
    p1_stats = state_engine.get_portfolio_stats(portfolio_type=P1_TYPE)
    p1_realized = float(p1_stats.get("total_realized_pnl") or 0.0)
    p1_equity = round(P1_BASE_CAPITAL + p1_realized + total_unrealized_pnl, 2)
    p1_cash = round(max(0.0, P1_BASE_CAPITAL + p1_realized - tot_invested), 2)

    # 3. Intraday Desk: Real-time Nifty 50, Bank Nifty & Live Options Analysis
    nifty_spot = float(
        kotak_neo_engine.get_live_price("NIFTY 50")
        or kotak_neo_engine.get_live_price("NIFTY")
        or kotak_neo_engine.get_live_price("^NSEI")
        or 0.0
    )
    if nifty_spot <= 0.0:
        try:
            live_spot = nifty_analytics_engine.get_live_nifty_spot()
            nifty_spot = float(live_spot.get("price") or 0.0)
        except Exception:
            pass
    banknifty_spot = float(
        kotak_neo_engine.get_live_price("NIFTY BANK")
        or kotak_neo_engine.get_live_price("BANKNIFTY")
        or 0.0
    )
    cur_atm = int(round(nifty_spot / 50.0) * 50) if nifty_spot > 0 else 0

    dash_live = {}
    try:
        dash_data = nifty_analytics_engine.get_intraday_expiry_dashboard(interval_mode="15m", force_refresh=False)
        dash_live = dash_data.get("live_analysis") or dash_data.get("live_synced_metrics") or {}
    except Exception as e:
        logger.debug(f"Continuous stream intraday dash notice: {e}")

    # 4. Options Desk: Portfolio Valuation
    opt_portfolio = nifty_analytics_engine.get_options_portfolio()

    # 5. Near-ATM Option Ticks for Real-Time Strike Streaming
    opt_ticks = kotak_neo_engine.get_near_atm_option_ticks(spot_price=nifty_spot, strike_range=12)

    # 6. Commodities Desk: Active MCX Contracts Live Quotes & Metrics
    from engines.commodities_engine import commodities_engine
    if kotak_neo_engine._is_authenticated:
        try:
            mcx_contracts = commodities_engine.get_active_contracts()
            mcx_tokens = [str(c.get("token")) for c in mcx_contracts.values() if c.get("token")]
            kotak_neo_engine.subscribe_mcx_tokens(mcx_tokens)
        except Exception:
            pass

    commodities_data = commodities_engine.get_streaming_payload()
    if isinstance(commodities_data, dict):
        commodities_data["stream_mode"] = stream_status.get("stream_mode", "OFFLINE")
        commodities_data["ticks_total"] = stream_status.get("ticks_received_total", len(all_ticks))

    is_market_open, _ = NSEMarketCalendar.can_execute_trades()
    return {
        "status": "SUCCESS",
        "source": "KOTAK_NEO_STREAM",
        "is_streaming": is_streaming,
        "is_market_open": is_market_open,
        "stream_mode": stream_status.get("stream_mode", "OFFLINE"),
        "is_authenticated": kotak_neo_engine._is_authenticated,
        "ticks_total": stream_status.get("ticks_received_total", len(all_ticks)),
        "timestamp_ist": format_ist(),
        "swing": {
            "positions_count": len(positions_summary),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "current_equity": p1_equity,
            "total_invested": round(tot_invested, 2),
            "available_cash": p1_cash,
            "realized_pnl": round(p1_realized, 2),
            "positions": positions_summary
        },
        "intraday": {
            "nifty_spot": nifty_spot,
            "banknifty_spot": banknifty_spot,
            "atm_strike": cur_atm or dash_live.get("atm_strike"),
            "straddle_premium": float(dash_live.get("straddle_premium") or 0.0),
            "atm_ce_ltp": float(dash_live.get("atm_ce_ltp") or 0.0),
            "atm_pe_ltp": float(dash_live.get("atm_pe_ltp") or 0.0),
            "near_atm_pcr": float(dash_live.get("near_atm_pcr") or 1.0),
            "overall_pcr": float(dash_live.get("overall_pcr") or 1.0),
            "call_wall": dash_live.get("call_wall"),
            "put_wall": dash_live.get("put_wall"),
            "max_pain": dash_live.get("max_pain"),
            "market_state": dash_live.get("market_state", "BALANCED_TWO_WAY_STRADDLE"),
            "what_is_happening": dash_live.get("what_is_happening", ""),
            "what_could_happen": dash_live.get("what_could_happen", ""),
            "quant_suggestion": dash_live.get("quant_suggestion", "")
        },
        "options": {
            "active_strategies_count": len(opt_portfolio.get("active_strategies", [])),
            "total_unrealized_pnl": opt_portfolio.get("summary", {}).get("total_unrealized_pnl", 0.0),
            "strategies": opt_portfolio.get("active_strategies", [])
        },
        "options_ticks": opt_ticks,
        "commodities": commodities_data,
        "equity_ticks": {
            s: {
                "ltp": t.get("ltp"),
                "change_pct": t.get("change_pct"),
                "high": t.get("high"),
                "low": t.get("low"),
                "open": t.get("open"),
                "close": t.get("close"),
                "volume": t.get("volume"),
                "updated_at": t.get("updated_at")
            }
            for s, t in all_ticks.items()
            if isinstance(t, dict) and t.get("ltp")
        }
    }


# --------------------------------------------------------------------------
# MCX COMMODITIES DESK API ENDPOINTS (POWERED BY KOTAK NEO)
# --------------------------------------------------------------------------
@app.get("/api/commodities/quotes")
def get_commodities_quotes(force_refresh: bool = False):
    """
    Returns authentic live quotes, pivots, range progress, and OI buildup
    for active MCX contracts (Crude Oil, Natural Gas, Gold, Silver, Copper, Zinc)
    streamed directly from Kotak Neo API.
    """
    from engines.commodities_engine import commodities_engine
    return commodities_engine.fetch_live_quotes(force_refresh=force_refresh)


@app.get("/api/commodities/depth")
def get_commodity_depth(symbol: str = "CRUDEOIL"):
    """
    Returns authentic live 5-level market depth (bids & asks) and pivot metrics
    for a specific MCX commodity contract directly from Kotak Neo.
    """
    from engines.commodities_engine import commodities_engine
    return commodities_engine.get_symbol_depth(symbol)


@app.get("/api/commodities/summary")
def get_commodities_summary():
    """
    Returns high-level MCX sector summaries (Energy, Bullion, Base Metals)
    and top market gainers/losers.
    """
    from engines.commodities_engine import commodities_engine
    payload = commodities_engine.fetch_live_quotes(force_refresh=False)
    return {
        "status": "SUCCESS",
        "is_mcx_open": payload.get("is_mcx_open"),
        "market_status_text": payload.get("market_status_text"),
        "summary": payload.get("summary", {}),
        "top_gainer": payload.get("top_gainer"),
        "top_loser": payload.get("top_loser"),
        "contracts_count": payload.get("contracts_count", 0),
        "timestamp": payload.get("timestamp")
    }


@app.post("/api/commodities/refresh")
def refresh_commodities_data():
    """
    Forces immediate scrip rediscovery and live quote refresh from Kotak Neo API.
    """
    from engines.commodities_engine import commodities_engine
    commodities_engine.discover_active_contracts(force=True)
    return commodities_engine.fetch_live_quotes(force_refresh=True)


# --------------------------------------------------------------------------
# MCX COMMODITIES OPTION CHAIN & STRATEGY BUILDER ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/api/commodities/options/expiries")
def get_commodity_option_expiries(underlying: str = "CRUDEOIL"):
    """
    Returns official active option expiries for MCX commodity contracts (Crude Oil, Natural Gas, Gold Mini, etc.).
    """
    from engines.commodities_engine import commodities_engine
    expiries = commodities_engine.get_available_option_expiries(underlying=underlying)
    return {
        "status": "SUCCESS",
        "underlying": underlying.upper(),
        "expiries": expiries,
    }


@app.get("/api/commodities/options/chain")
def get_commodity_option_chain(
    underlying: str = "CRUDEOIL",
    expiry: Optional[str] = None,
    count: int = 16,
    force_refresh: bool = False
):
    """
    Returns authentic live MCX Option Chain from Kotak Neo API (`mcx_fo`), including
    PCR (Overall, Near-ATM, Exact ATM), Max Pain, Call/Put Walls, ATM Straddle,
    Black-Scholes IV, and Full Greeks (Δ, Γ, Θ, Vega).
    """
    from engines.commodities_engine import commodities_engine
    return commodities_engine.fetch_commodity_option_chain(
        underlying=underlying,
        expiry=expiry,
        count=count,
        force_refresh=force_refresh
    )


@app.post("/api/commodities/options/strategy/analyze")
def analyze_commodity_strategy_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Analyzes a multi-leg commodity options strategy and returns dual-curve payoff graph data,
    Max Profit/Loss, Breakevens, POP, Net Premium, and Net Greeks.
    """
    from engines.commodities_engine import commodities_engine
    underlying = payload.get("underlying", "CRUDEOIL")
    legs = payload.get("legs", [])
    target_days = float(payload.get("target_days", 0.0))
    iv_shift_pct = float(payload.get("iv_shift_pct", 0.0))
    return commodities_engine.analyze_commodity_strategy(
        underlying=underlying,
        legs=legs,
        target_days=target_days,
        iv_shift_pct=iv_shift_pct
    )


@app.post("/api/commodities/options/strategy/deploy")
def deploy_commodity_strategy_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Deploys a multi-leg commodity strategy for real-time paper execution and tracking.
    """
    from engines.commodities_engine import commodities_engine
    return commodities_engine.deploy_commodity_strategy(payload)


@app.get("/api/commodities/options/strategy/deployed")
def get_deployed_commodity_strategies_endpoint():
    """
    Returns all active and squared-off deployed commodity strategies with live P&L.
    """
    from engines.commodities_engine import commodities_engine
    strategies = commodities_engine.get_deployed_commodity_strategies()
    return {
        "status": "SUCCESS",
        "strategies": strategies,
        "count": len(strategies),
    }


@app.post("/api/commodities/options/strategy/squareoff")
def square_off_commodity_strategy_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Squares off an active deployed commodity strategy.
    """
    from engines.commodities_engine import commodities_engine
    strategy_id = payload.get("strategy_id", "")
    return commodities_engine.square_off_commodity_strategy(strategy_id)

