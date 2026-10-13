"""
========================================================================================
FastAPI REST API & Web Application
========================================================================================
"""
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.telegram_vault import telegram_vault
from core.config import P1_BASE_CAPITAL, P2_BASE_CAPITAL, P2_MAX_SLOTS
from engines.institutional_engine import (
    InstitutionalEngine, get_institutional_status, PORTFOLIO_TYPE as P1_TYPE
)
from engines.nifty500_engine import (
    run_nifty500_pipeline_async, run_nifty500_pipeline_sync,
    get_nifty500_status, get_live_compounded_equity,
    PORTFOLIO_TYPE as P2_TYPE
)
from execution.state_engine import state_engine
from scheduler.cron_scheduler import (
    start_scheduler, shutdown_scheduler, get_scheduler_status,
    register_external_request_origin
)

logger = logging.getLogger("WebAPI")

app = FastAPI(
    title="Quant Trading Desk",
    description="Institutional NSE Equities & Nifty 500 Momentum Swing Desk",
    version="2.1.0"
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
    """Automatically detects the public Render URL for Anti-Sleep Guard and enforces zero browser caching."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    if host:
        register_external_request_origin(proto, host)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
async def on_startup():
    logger.info("Initializing Quant Trading Desk services...")
    # Cold-boot disaster recovery from Telegram Cloud Vault (Trade DB + OHLCV Historical Cache)
    try:
        telegram_vault.auto_restore_on_cold_boot()
    except Exception as e:
        logger.warning(f"Startup Telegram restore notice: {e}")
    start_scheduler()


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down Quant Trading Desk services...")
    shutdown_scheduler()


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


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index(request: Request):
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Quant Trading Desk</h1><p>index.html not found</p>")


@app.get("/sw.js")
def service_worker_cleanup():
    js = "self.addEventListener('install', () => self.skipWaiting()); self.addEventListener('activate', () => self.registration.unregister());"
    return Response(content=js, media_type="application/javascript")


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

    return {
        "status": "ONLINE",
        "current_time_ist": curr_time,
        "market_status": scheduler_info["market_status"],
        "trading_gate_open": can_trade,
        "trading_gate_message": gate_msg,
        "next_scan": scheduler_info["next_scan"],
        "scheduler_running": scheduler_info["scheduler_running"],
        "jobs": scheduler_info["jobs"],
        "render_keepalive": scheduler_info.get("render_keepalive", {}),
        "nse_holidays_meta": scheduler_info.get("nse_holidays_meta", {}),
        "telegram_vault": vault_info,
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


# --------------------------------------------------------------------------
# SCAN TRIGGER & LIVE PROGRESS
# --------------------------------------------------------------------------
class TriggerScanRequest(BaseModel):
    scan_type: str = "ALL"  # 'MORNING_1020AM', 'AFTERNOON_305PM', 'INSTITUTIONAL', 'NIFTY500', 'ALL'
    auto_execute: bool = True


@app.get("/api/scan/progress")
def get_scan_progress():
    prog = state_engine.get_scan_progress()
    p1_state = get_institutional_status()
    prog["chartink_fetched_count"] = p1_state.get("chartink_fetched_count", 0)
    prog["universe_size"] = p1_state.get("universe_size", 0)
    return prog


@app.post("/api/scan/trigger")
def trigger_scan(req: TriggerScanRequest):
    s_type = req.scan_type.upper()
    curr_prog = state_engine.get_scan_progress()
    if curr_prog.get("is_running"):
        return {
            "status": "ALREADY_RUNNING",
            "message": f"Scan is currently in progress: {curr_prog.get('stage')}",
            "progress": curr_prog
        }

    can_trade, gate_msg = NSEMarketCalendar.can_execute_trades()
    state_engine.start_scan_progress(
        scan_type=s_type,
        initial_stage=f"Initiating live {s_type} scan..."
    )

    def run_scan_worker():
        logger.info(f"⚡ Live scan started: {s_type} (Auto-Execute: {req.auto_execute}, TradingGate: {can_trade})")
        try:
            if s_type in ("MORNING_1020AM", "INSTITUTIONAL", "ALL"):
                inst = InstitutionalEngine()
                mode_override = "morning" if "MORNING" in s_type else ("afternoon" if "AFTERNOON" in s_type else None)
                inst.run(auto_execute=req.auto_execute, mode_override=mode_override)

            if s_type in ("AFTERNOON_305PM", "NIFTY500", "ALL"):
                run_nifty500_pipeline_sync(mode="MANUAL_TRIGGER", auto_execute=req.auto_execute)
        except Exception as e:
            logger.error(f"Error during live scan {s_type}: {e}", exc_info=True)
            state_engine.finish_scan_progress(f"Scan encountered error: {str(e)}")

    t = threading.Thread(target=run_scan_worker, daemon=True)
    t.start()

    return {
        "status": "STARTED",
        "message": f"Live scan '{s_type}' started! Fetching latest stocks & updating OHLCV gaps...",
        "auto_execute": req.auto_execute,
        "trading_gate_open": can_trade,
        "trading_gate_message": gate_msg,
        "timestamp": format_ist(),
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

    ok = state_engine.open_position(
        symbol=req.symbol,
        entry_price=req.entry_price,
        quantity=max(1, int(req.quantity)),
        stop_loss=req.stop_loss,
        target=req.target,
        target_2=req.target_2 or round(req.entry_price + 2.0 * max(1.0, req.target - req.entry_price), 2),
        triggered_from="UI_MANUAL",
        notes=req.notes,
        sector=req.sector,
        portfolio_type=req.portfolio_type,
        entry_reason="UI Execution"
    )
    if not ok:
        raise HTTPException(status_code=403, detail="Order blocked by Strict Market-Hours Trading Gate.")
    return {"status": "opened", "symbol": req.symbol, "timestamp": format_ist()}


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
    if payload.get("status") == "NOT_FOUND":
        fetcher = DirectYahooFinanceFetcher(max_workers=4)
        _, df, _ = fetcher.fetch_or_gapfill_ticker(sym, persist_immediately=True)
        if df is not None and not df.empty:
            payload = state_engine.get_stock_chart_payload(sym, limit_bars=bars)
    return payload


# --------------------------------------------------------------------------
# PORTFOLIO RESET & SINGLE MASTER FILE TELEGRAM CLOUD VAULT ENDPOINTS
# --------------------------------------------------------------------------
class ResetPortfolioRequest(BaseModel):
    portfolio_type: Optional[str] = None  # None = reset both portfolios to ₹0 PnL


@app.post("/api/portfolio/reset")
def reset_portfolio(req: ResetPortfolioRequest):
    return state_engine.reset_portfolio_trades(portfolio_type=req.portfolio_type)


@app.get("/api/vault/status")
def get_vault_status():
    return telegram_vault.get_status()


@app.post("/api/vault/sync")
def sync_vault_to_cloud():
    """Manually triggers background backup of the Single Master File (`data/swing_trades.db`) to Telegram Cloud."""
    cnt = state_engine.get_ohlcv_stats().get("stocks_count", 0)
    threading.Thread(
        target=telegram_vault.backup_master_file_to_cloud,
        kwargs={"custom_note": "Manual UI Single Master File Sync"},
        daemon=True
    ).start()
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

