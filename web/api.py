"""
========================================================================================
FastAPI REST API & Web Application
========================================================================================
"""
import asyncio
import logging
import threading
import math
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from contextlib import asynccontextmanager

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from core.config import P1_BASE_CAPITAL, P2_BASE_CAPITAL, P2_MAX_SLOTS
from engines.institutional_engine import (
    InstitutionalEngine, get_institutional_status, refresh_live_market_prices,
    PORTFOLIO_TYPE as P1_TYPE
)
from engines.nifty500_engine import (
    run_nifty500_pipeline_async, run_nifty500_pipeline_sync,
    get_nifty500_status, get_live_compounded_equity,
    PORTFOLIO_TYPE as P2_TYPE
)
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
    keepalive_active, keepalive_reason = NSEMarketCalendar.is_render_keepalive_window()
    return {
        "status": "ok",
        "timestamp": format_ist(),
        "render_keepalive_active": keepalive_active,
        "render_keepalive_reason": keepalive_reason
    }


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
            return HTMLResponse(content=f.read())
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

    return {
        "status": "ONLINE",
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
            "chartink_fetched_count": chartink_cnt,
            "universe_size": p1_state.get("universe_size") or sig_count,
            "cache_hits_count": p1_state.get("cache_hits_count", 0),
            "gap_filled_count": p1_state.get("gap_filled_count", 0),
            "full_downloads_count": p1_state.get("full_downloads_count", 0),
            "buy_now_count": p1_state.get("buy_now_count") if p1_state.get("universe_size") else b_now,
            "pre_breakout_count": p1_state.get("pre_breakout_count") if p1_state.get("universe_size") else p_brk,
            "wait_pullback_count": p1_state.get("wait_pullback_count") if p1_state.get("universe_size") else w_pb,
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
    if portfolio_type.strip().upper() == "ALL":
        return state_engine.get_trade_history(portfolio_type=None, limit=limit)
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_trade_history(portfolio_type=p_type, limit=limit)


@app.get("/api/portfolio/{portfolio_type}/signals")
def get_portfolio_signals(portfolio_type: str):
    p_type = state_engine._normalize_portfolio_type(portfolio_type)
    return state_engine.get_latest_signals(portfolio_type=p_type)


class RefreshPricesRequest(BaseModel):
    scope: str = "ACTIVE_AND_SIGNALS"


@app.post("/api/prices/refresh")
def refresh_prices_endpoint(req: RefreshPricesRequest = RefreshPricesRequest()):
    """
    Fetches live LTP from Yahoo Finance for all Active Portfolio Positions and Screened Signals on the page,
    updating MTM PnL, trailing stops, and signal prices immediately.
    Serialized via execution_queue to prevent collision with active scans.
    """
    def _refresh_task():
        return refresh_live_market_prices(scope=req.scope)

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


@app.get("/api/execution/queue")
def get_execution_queue_status():
    """Returns live telemetry of QuantDesk central Execution Queue."""
    return execution_queue.get_status()


@app.get("/api/scan/progress")
def get_scan_progress():
    prog = state_engine.get_scan_progress()
    p1_state = get_institutional_status()
    prog["chartink_fetched_count"] = p1_state.get("chartink_fetched_count", 0)
    prog["universe_size"] = p1_state.get("universe_size", 0)
    prog["queue_status"] = execution_queue.get_status()
    return prog


@app.post("/api/scan/trigger")
def trigger_scan(req: TriggerScanRequest):
    s_type = req.scan_type.upper()
    curr_prog = state_engine.get_scan_progress()
    q_status = execution_queue.get_status()

    # If scan is already running on execution worker
    if curr_prog.get("is_running") and q_status.get("is_running") and q_status.get("current_task", {}).get("type") == "CHARTINK_SCAN":
        return {
            "status": "ALREADY_RUNNING",
            "message": f"Scan is currently in progress: {curr_prog.get('stage')}",
            "progress": curr_prog
        }

    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()

    # Determine initial stage based on queue state
    if q_status.get("is_running"):
        init_stage = f"⏳ In Queue (Position {q_status.get('queued_count', 0) + 1}): Waiting for [{q_status.get('current_task', {}).get('name')}] to complete..."
    else:
        init_stage = f"Initiating live {s_type} scan..."

    state_engine.start_scan_progress(
        scan_type=s_type,
        initial_stage=init_stage
    )

    effective_auto_exec = bool(req.auto_execute and can_trade)
    def run_scan_worker():
        logger.info(f"⚡ Live scan started: {s_type} (Auto-Execute: {effective_auto_exec}, TradingGate: {can_trade})")
        state_engine.update_scan_progress(stage=f"Step 1/6: Fetching live {s_type} universe from Chartink...", percent=10)
        try:
            if s_type in ("MORNING_1020AM", "AFTERNOON_305PM", "AFTERNOON_300PM", "INSTITUTIONAL", "ALL"):
                inst = InstitutionalEngine()
                mode_override = "morning" if "MORNING" in s_type else ("afternoon" if "AFTERNOON" in s_type else None)
                inst.run(auto_execute=effective_auto_exec, mode_override=mode_override)

            if s_type in ("AFTERNOON_305PM", "AFTERNOON_300PM", "NIFTY500", "ALL"):
                run_nifty500_pipeline_sync(mode="MANUAL_TRIGGER", auto_execute=effective_auto_exec)
        except Exception as e:
            logger.error(f"Error during live scan {s_type}: {e}", exc_info=True)
            state_engine.finish_scan_progress(f"Scan encountered error: {str(e)}")

    task_res = execution_queue.submit_task(
        run_scan_worker,
        task_name=f"Live Scan: {s_type}",
        task_type="CHARTINK_SCAN",
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


# --------------------------------------------------------------------------
# MANUAL TRADE OPERATIONS (STRICT MARKET-HOURS TRADING GATE ENFORCED)
# --------------------------------------------------------------------------
class ClosePositionRequest(BaseModel):
    symbol: str
    exit_price: Optional[float] = None
    exit_reason: str = "MANUAL_DISCRETIONARY"
    notes: str = ""


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


@app.post("/api/position/open")
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

    max_slots = 15 if "NIFTY" in p_type else 10
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


@app.post("/api/position/close")
def close_active_position(req: ClosePositionRequest):
    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    if not can_trade:
        raise HTTPException(
            status_code=403,
            detail=gate_msg
        )

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


@app.post("/api/portfolio/reset")
def reset_portfolio(req: ResetPortfolioRequest):
    return state_engine.reset_portfolio_trades(
        portfolio_type=req.portfolio_type,
        confirm_token=req.confirm_token or ""
    )


@app.get("/api/vault/status")
def get_vault_status():
    return telegram_vault.get_status()


@app.post("/api/vault/sync")
def sync_vault_to_cloud():
    """Manually triggers background backup of the Single Master File (`data/swing_trades.db`) to Telegram Cloud."""
    cnt = state_engine.get_ohlcv_stats().get("stocks_count", 0)
    telegram_vault.sync_master_vault_async(note="Manual UI Single Master File Sync", send_notification=False)
    return {
        "status": "STARTED",
        "message": f"Updating Single Master File in-place on Telegram Cloud ({cnt} Stocks OHLCV + All Trades)..."
    }


@app.post("/api/vault/restore")
def restore_vault_from_cloud():
    """Restores the Single Master File (`data/swing_trades.db`) from Telegram Cloud Vault."""
    res = telegram_vault.restore_master_file_from_cloud()
    return {
        "status": res.get("status", "SUCCESS"),
        "db_restored": res.get("status") == "SUCCESS",
        "ohlcv_restored": res.get("status") == "SUCCESS",
        "details": res
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
def get_nifty_intraday_expiry(session_date: Optional[str] = None, interval_mode: str = "30m", force_refresh: bool = False):
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
def capture_nifty_intraday_snapshot(slot_override: Optional[str] = None, interval_mode: str = "30m"):
    """Manually or automatically captures a live Weekly Option Expiry snapshot and syncs to Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.capture_intraday_50min_snapshot(slot_override=slot_override, sync_cloud=True, interval_mode=interval_mode)


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
        return _sanitize_for_json(data)
    except Exception as e:
        logger.error(f"Error computing live predictions: {e}", exc_info=True)
        cached = state_engine.get_kv("latest_nifty_predictions")
        if cached and isinstance(cached, dict) and cached.get("status") == "OK":
            return _sanitize_for_json(cached)
        return {"status": "ERROR", "message": f"Prediction service error: {str(e)}"}


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


class AutoDeployOptionsRequest(BaseModel):
    modes: Optional[list] = ["WEEKLY", "MONTHLY", "INTRADAY"]
    force_refresh: Optional[bool] = True


@app.get("/api/options/portfolio")
def get_options_portfolio_endpoint():
    """Returns Options Trading Portfolio with live MTM valuation, Greeks, and closed performance."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.get_options_portfolio()


@app.post("/api/options/reset-portfolio")
def reset_options_portfolio_endpoint():
    """Resets the Options Trading Portfolio: clears all active and closed strategies, wipes realized PnL to ₹0, and syncs cloud vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.reset_options_portfolio()



@app.post("/api/options/deploy-strategy")
def deploy_options_strategy_endpoint(req: DeployStrategyRequest):
    """Deploys a multi-leg options strategy into the Options Trading Portfolio and syncs to Telegram Cloud Vault."""
    from engines.nifty_analytics_engine import nifty_analytics_engine
    return nifty_analytics_engine.deploy_options_strategy(req.dict())


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
    lot_size: int = 75


@app.get("/api/options/intraday-rules/backtest")
def get_options_intraday_backtest_endpoint(lookback_sessions: int = 50, lot_size: int = 75):
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
    lots = req.lot_size if req else 75
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
    lot_size: int = 75


@app.get("/api/options/positional-rules/backtest")
def get_options_positional_backtest_endpoint(
    horizon: str = "WEEKLY",
    lookback_sessions: int = 50,
    lot_size: int = 75
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
    lots = req.lot_size if req else 75
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
    """Computes exact TreeSHAP feature attributions natively via XGBoost C++ engine."""
    from engines.tree_quant_ml_engine import tree_quant_ml_engine
    # Use current market regime features
    sample_features = {
        "composite_score": 14.5,
        "near_atm_pcr": 1.22,
        "vix_spot": 13.8,
        "rsi14": 58.5,
        "trend_strength_adx": 26.0,
        "horizon_code": 1.0 if horizon == "INTRADAY" else (2.0 if horizon == "WEEKLY" else 4.0),
        "regime_code": 1.0
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

