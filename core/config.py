"""
Application Configuration & Environment Settings
"""
import os
from pathlib import Path

# Base Paths (Single Master File Architecture: only `data/swing_trades.db` is created on disk)
BASE_DIR = Path(__file__).resolve().parent.parent

# Automatically load local .env if present (without external dependencies)
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass

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
P1_MAX_SLOTS = int(os.environ.get("P1_MAX_SLOTS", 10))               # 10 maximum concurrent slots
CHARTINK_URL = os.environ.get("CHARTINK_URL", "https://chartink.com/screener/testtttt-157")

# Portfolio 2: Nifty 500 Momentum Compounded Engine
P2_BASE_CAPITAL = float(os.environ.get("P2_BASE_CAPITAL", 100000.0))  # ₹1,00,000
P2_RISK_PER_TRADE = float(os.environ.get("P2_RISK_PER_TRADE", 0.015)) # 1.5% dynamic compounded risk
P2_MAX_SLOTS = int(os.environ.get("P2_MAX_SLOTS", 10))                # 10 maximum concurrent slots
P2_STOP_ATR_MULT = float(os.environ.get("P2_STOP_ATR_MULT", 2.0))     # 2.0x ATR Wilder stop

# Portfolio 3: Pre-Breakout Coil & Intraday Volume Confirmation Engine
P3_BASE_CAPITAL = float(os.environ.get("P3_BASE_CAPITAL", 200000.0))  # ₹2,00,000
P3_RISK_PER_TRADE = float(os.environ.get("P3_RISK_PER_TRADE", 0.010)) # 1.0% fixed risk model per trade
P3_MAX_SLOTS = int(os.environ.get("P3_MAX_SLOTS", 10))               # 10 maximum concurrent slots
P3_ALLOCATION_CAP_PCT = float(os.environ.get("P3_ALLOCATION_CAP_PCT", 0.20)) # 20% max allocation per stock
P3_TARGET_R = float(os.environ.get("P3_TARGET_R", 2.0))              # +2.0R first half profit target
P3_VOLUME_PACING_THRESHOLD = float(os.environ.get("P3_VOLUME_PACING_THRESHOLD", 3.12)) # 3.12x volume pacing confirmation

# Risk Controls & Capital Allocation Gates (F-38)
MAX_POSITION_CAP_PCT = float(os.environ.get("MAX_POSITION_CAP_PCT", 0.20)) # 20% max capital per position
MAX_SECTOR_POSITIONS = int(os.environ.get("MAX_SECTOR_POSITIONS", 3))       # Max 3 positions in single sector
DAILY_LOSS_LIMIT_PCT = float(os.environ.get("DAILY_LOSS_LIMIT_PCT", 0.03)) # 3% daily loss kill switch
EQUITY_ROUND_TRIP_COST_RATE = float(os.environ.get("EQUITY_ROUND_TRIP_COST_RATE", 0.0012)) # 0.12% round trip


# Options Quant Desk Defaults (Nifty 50 Lot Size = 65)
NIFTY_LOT_SIZE = int(os.environ.get("NIFTY_LOT_SIZE", 65))

# Scheduler Config (10:20 AM Morning & 3:00 PM MOC Close)
SCHEDULE_MORNING_HOUR = int(os.environ.get("SCHEDULE_MORNING_HOUR", 10))
SCHEDULE_MORNING_MINUTE = int(os.environ.get("SCHEDULE_MORNING_MINUTE", 20))
SCHEDULE_AFTERNOON_HOUR = int(os.environ.get("SCHEDULE_AFTERNOON_HOUR", 15))
SCHEDULE_AFTERNOON_MINUTE = int(os.environ.get("SCHEDULE_AFTERNOON_MINUTE", 0))  # 3:00 PM IST sharp

# Telegram Cloud Vault Config (Persists Trade DB + OHLCV Historical Cache across Render restarts)
TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "8887427921:AAHS4Nq3BtxPW6vZnQ7z3crpPCprVHbJsMI"
)
TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "8183911891"
)
VAULT_MANIFEST_PATH = Path(os.environ.get("VAULT_MANIFEST_PATH", str(DB_PATH.parent / "vault_manifest.json")))
VAULT_CONFIG_PATH = Path(os.environ.get("VAULT_CONFIG_PATH", str(DB_PATH.parent / "vault_config.json")))

# ==============================================================================
# AI Intelligence Layer Configuration (Groq & OpenRouter Free Tier)
# ==============================================================================
AI_ENABLED = os.environ.get("AI_ENABLED", "true").lower() in ("true", "1", "yes")
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower()
AI_PRIMARY_MODEL = os.environ.get("AI_PRIMARY_MODEL", "openai/gpt-oss-120b")
AI_FALLBACK_MODEL = os.environ.get("AI_FALLBACK_MODEL", "openai/gpt-oss-20b")
AI_TIMEOUT_SECONDS = int(os.environ.get("AI_TIMEOUT_SECONDS", 20))
AI_MAX_RETRIES = int(os.environ.get("AI_MAX_RETRIES", 1))

# Groq Free Tier
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# OpenRouter Free Tier
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
# Security & Administration
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# Quantitative Options Risk & Regulatory Rates
# Current RBI Repo Rate benchmark proxy (5.50%)
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", 0.055))
DEFAULT_OPTIONS_IV = float(os.environ.get("DEFAULT_OPTIONS_IV", 13.0))

# Options Regulatory Costs (Per NSE Circular NSE/FATAX/73524 - STT on Option Sale = 0.15%)
OPTIONS_STT_RATE = float(os.environ.get("OPTIONS_STT_RATE", 0.0015))  # 0.15% on sale turnover
OPTIONS_EXCHANGE_TURNOVER_RATE = float(os.environ.get("OPTIONS_EXCHANGE_TURNOVER_RATE", 0.000505))  # 0.0505%
OPTIONS_SEBI_TURNOVER_RATE = float(os.environ.get("OPTIONS_SEBI_TURNOVER_RATE", 0.000001))
OPTIONS_BROKERAGE_PER_ORDER = float(os.environ.get("OPTIONS_BROKERAGE_PER_ORDER", 20.0))
OPTIONS_GST_RATE = float(os.environ.get("OPTIONS_GST_RATE", 0.18))
OPTIONS_STAMP_DUTY_BUY = float(os.environ.get("OPTIONS_STAMP_DUTY_BUY", 0.00003))

def calc_regulatory_options_cost(premium: float, qty: int, is_sell: bool = True) -> float:
    """Calculates canonical Indian regulatory transaction costs for an options trade."""
    turnover = max(0.0, float(premium) * int(qty))
    stt = (OPTIONS_STT_RATE * turnover) if is_sell else 0.0
    ex_fee = OPTIONS_EXCHANGE_TURNOVER_RATE * turnover
    sebi = OPTIONS_SEBI_TURNOVER_RATE * turnover
    stamp = (OPTIONS_STAMP_DUTY_BUY * turnover) if not is_sell else 0.0
    brokerage = OPTIONS_BROKERAGE_PER_ORDER
    gst = OPTIONS_GST_RATE * (brokerage + ex_fee + sebi)
    return round(stt + ex_fee + sebi + stamp + brokerage + gst, 2)



# ==============================================================================
# Kotak Neo API & Real-Time Market Streaming Configuration
# ==============================================================================
KOTAK_NEO_ENABLED = os.environ.get("KOTAK_NEO_ENABLED", "true").lower() in ("true", "1", "yes")
KOTAK_NEO_CONSUMER_KEY = os.environ.get("KOTAK_NEO_CONSUMER_KEY", "")
KOTAK_NEO_MOBILE_NUMBER = os.environ.get("KOTAK_NEO_MOBILE_NUMBER", "")
KOTAK_NEO_UCC = os.environ.get("KOTAK_NEO_UCC", "")
KOTAK_NEO_MPIN = os.environ.get("KOTAK_NEO_MPIN", "")
KOTAK_NEO_TOTP_SECRET = os.environ.get("KOTAK_NEO_TOTP_SECRET", "")
KOTAK_NEO_ENVIRONMENT = os.environ.get("KOTAK_NEO_ENVIRONMENT", "prod")
