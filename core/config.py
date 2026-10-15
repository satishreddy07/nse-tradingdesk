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
CHARTINK_URL = os.environ.get("CHARTINK_URL", "https://chartink.com/screener/testtttt-157")

# Portfolio 2: Nifty 500 Momentum Compounded Engine
P2_BASE_CAPITAL = float(os.environ.get("P2_BASE_CAPITAL", 100000.0))  # ₹1,00,000
P2_RISK_PER_TRADE = float(os.environ.get("P2_RISK_PER_TRADE", 0.015)) # 1.5% dynamic compounded risk
P2_MAX_SLOTS = int(os.environ.get("P2_MAX_SLOTS", 10))                # 10 maximum concurrent slots
P2_STOP_ATR_MULT = float(os.environ.get("P2_STOP_ATR_MULT", 2.0))     # 2.0x ATR Wilder stop

# Options Quant Desk Defaults
NIFTY_LOT_SIZE = int(os.environ.get("NIFTY_LOT_SIZE", 75))

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
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

