"""
========================================================================================
Quant Trading Desk - Master Entry Point
========================================================================================
Starts FastAPI web server with automated APScheduler on Asia/Kolkata (IST):
- 10:20 AM IST: Morning Intraday Momentum & Entry Scan
- 03:00 PM IST: Afternoon Close-of-Day Swing Scan & MOC Execution

Ready for local development and Render Cloud deployment.
========================================================================================
"""

import sys
import os
import argparse
import logging
import uvicorn

# Configure Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"

from core.config import HOST, PORT
from core.data_quality import format_ist
from scheduler.cron_scheduler import start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def run_cli_scan(scan_type: str):
    """Executes on-demand scan via CLI."""
    scan_type = scan_type.lower()
    logger.info(f"Executing CLI scan for: {scan_type}")

    if scan_type in ("morning", "1020", "p1", "institutional"):
        from engines.institutional_engine import InstitutionalEngine
        engine = InstitutionalEngine()
        signals = engine.run(auto_execute=True, mode_override="morning")
        print(f"\n[OK] Institutional Scan Complete. {len(signals)} stocks evaluated.")

    elif scan_type in ("afternoon", "305", "p2", "nifty500"):
        from engines.nifty500_engine import run_nifty500_pipeline_sync
        res = run_nifty500_pipeline_sync(mode="CLI_TRIGGER", auto_execute=True)
        print(f"\n[OK] Nifty 500 Scan Complete: {res.get('orders_executed', 0)} orders placed.")

    elif scan_type == "all":
        from engines.institutional_engine import InstitutionalEngine
        from engines.nifty500_engine import run_nifty500_pipeline_sync
        engine = InstitutionalEngine()
        engine.run(auto_execute=True, mode_override="afternoon")
        run_nifty500_pipeline_sync(mode="CLI_TRIGGER", auto_execute=True)
        print("\n[OK] All scans executed successfully.")


def run_cli_backtest():
    """Runs the 250-day Event-Driven Portfolio Backtest."""
    from engines.institutional_engine import (
        InstitutionalEngine, InstitutionalPortfolioBacktester
    )
    engine = InstitutionalEngine()
    print("Fetching tickers and universe data for backtest...")
    tickers = engine.chartink.fetch_tickers()
    ohlcv = engine.yahoo.fetch_universe_ohlcv(tickers, min_bars=105)
    bt = InstitutionalPortfolioBacktester(capital=1000000.0, risk_per_trade_pct=0.75)
    bt.run(ohlcv, lookback_days=250)


def main():
    parser = argparse.ArgumentParser(description="Quant Trading Desk - Automated Swing Engines")
    parser.add_argument("--scan", type=str, choices=["morning", "afternoon", "p1", "p2", "all"], help="Run immediate scan")
    parser.add_argument("--backtest", action="store_true", help="Run 250-day Event-Driven Institutional Backtest")
    parser.add_argument("--host", type=str, default=HOST, help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    args = parser.parse_args()

    if args.scan:
        run_cli_scan(args.scan)
        return

    if args.backtest:
        run_cli_backtest()
        return

    # Start Server with Scheduler
    print("=" * 80)
    print("QUANT TRADING DESK — PRODUCTION SERVER")
    print(f"Current Time: {format_ist()}")
    print("Dual Automated Engines:")
    print("  • Portfolio 1: Institutional 6-Pillar Percentile Engine (₹5,00,000 Base)")
    print("  • Portfolio 2: Nifty 500 Momentum Compounded Engine (₹1,00,000 Dynamic Compounding)")
    print("Automated Daily Schedule (NSE Mon-Fri):")
    print("  • 10:20 AM IST: Morning Intraday Momentum & Entry Scan")
    print("  • 03:00 PM IST: Afternoon MOC Swing Scan & 4-Tier Strategy Exits")
    print(f"Listening on http://{args.host}:{args.port}")
    print("=" * 80)

    uvicorn.run("web.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
