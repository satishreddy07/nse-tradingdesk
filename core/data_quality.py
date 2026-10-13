"""
Core Data Quality & Timezone Utilities
"""
import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

def now_ist() -> datetime.datetime:
    """Returns current datetime in Asia/Kolkata (IST)."""
    return datetime.datetime.now(IST)

def format_ist(dt: datetime.datetime = None) -> str:
    """Formats datetime in human-readable IST string."""
    if dt is None:
        dt = now_ist()
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")
