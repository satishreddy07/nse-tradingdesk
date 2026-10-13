"""
Application Configuration & Environment Settings
"""
import os
from pathlib import Path

# Base Paths (Single Master File Architecture: only `data/swing_trades.db` is created on disk)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_DATA_DIR = DATA_DIR / "historical_data"
SEEDS_DIR = DATA_DIR / "seeds"

DB_PATH = Path(os.environ.get("DB_PATH", str(DATA_DIR / "swing_trades.db")))

# Server Configuration (Render compatibility)
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

# Quant Engine Defaults
# Portfolio 1: Institutional 6-Pillar Engine
P1_BASE_CAPITAL = float(os.environ.get("P1_BASE_CAPITAL", 500000.0))  # ₹5,00,000
P1_BASE_RISK_PCT = float(os.environ.get("P1_BASE_RISK_PCT", 0.75))   # 0.75% base risk
CHARTINK_URL = os.environ.get("CHARTINK_URL", "https://chartink.com/screener/testtttt-157")

# Portfolio 2: Nifty 500 Momentum Compounded Engine
P2_BASE_CAPITAL = float(os.environ.get("P2_BASE_CAPITAL", 100000.0))  # ₹1,00,000
P2_RISK_PER_TRADE = float(os.environ.get("P2_RISK_PER_TRADE", 0.015)) # 1.5% dynamic compounded risk
P2_MAX_SLOTS = int(os.environ.get("P2_MAX_SLOTS", 10))                # 10 maximum concurrent slots
P2_STOP_ATR_MULT = float(os.environ.get("P2_STOP_ATR_MULT", 2.0))     # 2.0x ATR Wilder stop

# Scheduler Config
SCHEDULE_MORNING_HOUR = 10
SCHEDULE_MORNING_MINUTE = 20
SCHEDULE_AFTERNOON_HOUR = 15
SCHEDULE_AFTERNOON_MINUTE = 5

# Telegram Cloud Vault Config (Persists Trade DB + OHLCV Historical Cache across Render restarts)
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8887427921:AAHS4Nq3BtxPW6vZnQ7z3crpPCprVHbJsMI"
)
TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "8183911891"
)
VAULT_MANIFEST_PATH = DATA_DIR / "vault_manifest.json"
VAULT_CONFIG_PATH = DATA_DIR / "vault_config.json"

