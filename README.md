# Quant Trading Desk — Dual Institutional Swing Engines

Institutional-grade automated swing trading platform and portfolio management engine engineered for the **National Stock Exchange of India (NSE)**.

Features two distinct, mathematically sound quantitative swing portfolios:
1. **Portfolio 1 — Institutional 6-Pillar Percentile Engine**: Liquid universe from Chartink with 30 raw technical metrics, 6 pillars, real sector relative strength, ML meta-classifier re-ranking, and score-weighted position sizing.
2. **Portfolio 2 — Nifty 500 Momentum Compounded Engine**: Daily Market-On-Close (MOC) swing engine for Nifty 500 constituents with dynamic equity reinvestment compounding, 10 slots cap, 7-point entry criteria matrix, and 4-tier strategy exit hierarchy.

Integrated with an automated scheduler running every trading day at **10:20 AM IST** and **3:00 PM IST**, deployed on a Render backend with a responsive PC and mobile web application.

---

## Architecture & System Design

```
                                  NSE Indian Equities Market
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         Chartink Screener (Liquid)                        Nifty 500 Constituents
                      │                                               │
                      ▼                                               ▼
     Direct Yahoo Finance Chart API                 Async Batch Candle Downloader
         (2-Year OHLCV + Local Cache)                 (6-Month Candles + ^CRSLDX)
                      │                                               │
                      ▼                                               ▼
     6-Pillar Percentile Engine                     7-Point Entry Criteria Matrix
     ├── Trend (25%)                                ├── Macro Regime Gate (^CRSLDX > 50EMA)
     ├── Momentum (20%)                             ├── Structural Uptrend (20>50 & P>50)
     ├── Volatility (15%)                           ├── RS Leader vs Nifty 500 (> 0.0)
     ├── Volume (15%)                               ├── Pullback Pocket (0.985 - 1.020 EMA20)
     ├── Sector RS (15%)                            ├── RSI Cooldown (40 - 62)
     └── Proximity (10%)                            ├── Bullish Bounce Bar
                      │                             └── Volume Expansion (>= 0.8x SMA20)
                      ▼                                               │
       Calibrated ML Meta-Classifier                                  ▼
      P(Reach Target 1 before Stop)                 Dynamic Equity Compounding Engine
                      │                             (1.5% Risk Budget • 10 Max Slots)
                      ▼                                               │
     Score-Weighted Position Sizing                                   ▼
        Tier A+ / A / B+ (SL, T1, T2)               4-Tier Strategy Exit Hierarchy
                      │                             ├── P1: Initial SL (-2.0 ATR)
                      │                             ├── P2: Trailing SL (5D Low @ +2R)
                      │                             ├── P3: Fast Cut (< 20 EMA)
                      │                             └── P4: Time Exit (8-Day Stop)
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                 SQLite State Engine (WAL Mode)
                                              │
                     ┌────────────────────────┼────────────────────────┐
                     ▼                        ▼                        ▼
           APScheduler Daemon        FastAPI REST API         Responsive Web Desk
           10:20 AM & 3:00 PM       Endpoints & Health       PC & Mobile Home Layout
```

---

## Quantitative Philosophy & Timing Rationale

### 1. Why 10:20 AM IST? (Morning Intraday Scan)
- **Market Opening Noise Elimination**: NSE opens at 09:15 AM. The initial 45–60 minutes are dominated by retail emotional trades, overnight gap reactions, and stop-hunts.
- **True Opening Range & RVOL**: By 10:20 AM, institutional liquidity blocks have settled, true opening ranges have formed, and opening volume expansion (RVOL) provides genuine statistical significance.
- **Action**: Evaluates morning momentum, checks Minervini Stage 2 breakouts, and reconciles overnight gap positions.

### 2. Why 3:00 PM IST? (Afternoon MOC Swing Scan)
- **Market-On-Close (MOC) Precision**: NSE regular trading closes at 3:30 PM. Running at 3:00 PM gives 30 minutes of execution runway.
- **92% Volume Profile Completion**: Daily volume distribution is over 90% finalized, preventing morning fakeouts.
- **Intraday Confirmation**: Applies a ±15 point confirmation score based on range position and volume delivery.
- **Action**: Audits Nifty 500 macro regime gate, reconciles 4-tier strategy exits, and enters new qualifying setups before the market close.

### 3. NSE Calendar & Holiday Guard
- Automatically identifies Saturdays, Sundays, and official NSE Exchange holidays (Republic Day, Holi, Diwali, etc.).
- On non-trading days, the scheduler safely remains dormant to prevent off-market network requests.

---

## Dual Portfolios Overview

| Parameter | Portfolio 1: Institutional 6-Pillar | Portfolio 2: Nifty 500 Momentum |
| :--- | :--- | :--- |
| **Target Universe** | Chartink fundamentally-screened liquid mid/large caps | Official Nifty 500 constituents |
| **Capital Model** | ₹5,00,000 Base capital | ₹1,00,000 Base with dynamic equity reinvestment compounding |
| **Risk per Trade** | Tiered: 1.0% (A+), 0.75% (A), 0.60% (B+) | 1.5% of live mark-to-market compounded equity |
| **Slot Limit** | Max 10 positions, max 2 per sector | 10 slots (10% allocation cap per slot) |
| **Primary Entry** | 6-Pillar percentile rank + ML P(W) re-ranking | 7-point technical & regime matrix |
| **Stop Loss** | Min(20d Low, Entry - 1.5x ATR14) / Hard 8% | 2.0x ATR Wilder volatility stop |
| **Target / Exit** | T1 = +1.5R (book 50%, BE trail), T2 = +3.0R | 4-Tier: Initial SL, Trailing 5D Low @ +2R, Fast Cut < 20 EMA, 8-Day time exit |

---

## Local Development & Setup

### Prerequisites
- Python 3.10+ installed.

### Installation
```bash
git clone https://github.com/your-username/tradingdesk.git
cd tradingdesk

# Create virtual environment (optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Server & Automated Scheduler
```bash
python main.py
```
Open your browser to: **`http://localhost:8000`**

### Running Immediate CLI Scans
```bash
# Run morning 10:20 AM scan
python main.py --scan morning

# Run afternoon 3:00 PM scan
python main.py --scan afternoon

# Run Portfolio 1 (Institutional 6-Pillar) scan only
python main.py --scan p1

# Run Portfolio 2 (Nifty 500 Momentum) scan only
python main.py --scan p2
```

### Running 250-Day Backtest Simulator
```bash
python main.py --backtest
```

---

## Render Cloud Deployment Guide

This project is 100% prepared for 1-click deployment on **Render**:

### Option 1: Render Blueprint (`render.yaml`)
1. Push this repository to GitHub or GitLab.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New** > **Blueprint**.
3. Connect your repository. Render will automatically read `render.yaml` and configure:
   - Web Service: `quant-trading-desk`
   - Region: `singapore` (closest to India / NSE)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn web.api:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/health`
4. Click **Apply**.

### Option 2: Manual Web Service on Render
1. Click **New** > **Web Service**.
2. Connect your Git repository.
3. Configure the following:
   - **Environment**: `Python 3`
   - **Region**: `Singapore (ap-southeast-1)`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn web.api:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `TZ` = `Asia/Kolkata`
   - `P1_BASE_CAPITAL` = `500000.0`
   - `P2_BASE_CAPITAL` = `100000.0`
   - `P2_RISK_PER_TRADE` = `0.015`
   - `P2_MAX_SLOTS` = `10`
5. Click **Create Web Service**.

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Server health check (Render probe) |
| `/api/status` | `GET` | Live IST clock, market status, next scheduled run, engine status |
| `/api/portfolio/{type}/summary` | `GET` | Equity, active positions, win rate, profit factor, expectancy |
| `/api/portfolio/{type}/positions`| `GET` | Open positions with real-time MTM PnL and stop distance |
| `/api/portfolio/{type}/history`  | `GET` | Closed trades audit log with R-multiples and hold durations |
| `/api/portfolio/{type}/signals`  | `GET` | Latest screened setups with 6-pillar breakdown & sizing |
| `/api/scan/trigger`             | `POST`| Trigger on-demand background scan (`MORNING_1020AM`, `AFTERNOON_305PM`, etc.) |
| `/api/position/close`           | `POST`| Manually close an active position |
| `/api/scan/logs`                | `GET` | Execution duration, candidate counts, and status logs |
| `/api/calendar`                 | `GET` | NSE trading calendar and upcoming market holidays |
