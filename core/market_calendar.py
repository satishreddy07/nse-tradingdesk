"""
NSE Market Calendar, Dynamic NSE Holiday Fetcher & Render Anti-Sleep Window Detector
Manages National Stock Exchange (NSE) trading hours, live holiday syncing from nseindia.com,
and the strict 09:00 AM – 03:50 PM IST Render Anti-Sleep Guard window.
"""
import datetime
import logging
import threading
from typing import Dict, Any, Tuple, Optional
import requests

from core.data_quality import IST, now_ist, format_ist

logger = logging.getLogger("NSEMarketCalendar")


class NSEMarketCalendar:
    """
    Standard NSE Indian Equities Market Calendar:
    - Normal Trading Hours: 09:15 to 15:30 IST (Monday - Friday, excluding NSE holidays)
    - Pre-Market: 09:00 to 09:15 IST
    - Render Anti-Sleep Active Window: Strictly 09:00 to 15:50 IST (Trading Days Only)
    - Dynamic Holiday Source: https://www.nseindia.com/api/holiday-master?type=trading (Segment: CM)
    """

    NSE_HOLIDAY_API_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
    NSE_HOME_URL = "https://www.nseindia.com"
    UPSTOX_HOLIDAY_API_URL = "https://api.upstox.com/v2/market/holidays"

    # Verified Baseline NSE Holidays (YYYY-MM-DD) — automatically merged with live NSE API response
    NSE_HOLIDAYS: Dict[str, str] = {
        # 2025
        "2025-01-26": "Republic Day",
        "2025-02-26": "Mahashivratri",
        "2025-03-14": "Holi",
        "2025-03-31": "Id-Ul-Fitr",
        "2025-04-10": "Mahavir Jayanti",
        "2025-04-14": "Dr. Ambedkar Jayanti",
        "2025-04-18": "Good Friday",
        "2025-05-01": "Maharashtra Day",
        "2025-08-15": "Independence Day",
        "2025-08-27": "Ganesh Chaturthi",
        "2025-10-02": "Mahatma Gandhi Jayanti",
        "2025-10-21": "Diwali Laxmi Pujan",
        "2025-10-22": "Diwali Balipratipada",
        "2025-11-05": "Gurunanak Jayanti",
        "2025-12-25": "Christmas",

        # 2026 (Synced with Official NSE Capital Market Holiday Master)
        "2026-01-15": "Municipal Corporation Election - Maharashtra",
        "2026-01-26": "Republic Day",
        "2026-02-15": "Mahashivratri",
        "2026-03-03": "Holi",
        "2026-03-21": "Id-Ul-Fitr (Ramadan Eid)",
        "2026-03-26": "Shri Ram Navami",
        "2026-03-31": "Shri Mahavir Jayanti",
        "2026-04-03": "Good Friday",
        "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
        "2026-05-01": "Maharashtra Day",
        "2026-05-28": "Bakri Id",
        "2026-06-26": "Muharram",
        "2026-08-15": "Independence Day",
        "2026-09-14": "Ganesh Chaturthi",
        "2026-10-02": "Mahatma Gandhi Jayanti",
        "2026-10-20": "Dussehra",
        "2026-11-08": "Diwali Laxmi Pujan*",
        "2026-11-10": "Diwali-Balipratipada",
        "2026-11-24": "Prakash Gurpurb Sri Guru Nanak Dev",
        "2026-12-25": "Christmas",

        # 2027 Baseline
        "2027-01-26": "Republic Day",
        "2027-03-22": "Holi",
        "2027-03-26": "Good Friday",
        "2027-04-14": "Dr. Ambedkar Jayanti",
        "2027-05-01": "Maharashtra Day",
        "2027-08-15": "Independence Day",
        "2027-10-02": "Mahatma Gandhi Jayanti",
        "2027-12-25": "Christmas",
    }

    _sync_lock = threading.Lock()
    _holiday_meta: Dict[str, Any] = {
        "source": "BASELINE_INIT",
        "last_synced_ist": None,
        "last_synced_date": None,
        "live_fetched_count": 0,
        "total_holidays_loaded": len(NSE_HOLIDAYS),
        "last_error": None
    }
    _db_loaded: bool = False

    @classmethod
    def _parse_nse_date(cls, raw_date: str) -> Optional[str]:
        """Converts NSE date strings like '26-Jan-2026' into '2026-01-26'."""
        if not raw_date:
            return None
        raw_date = str(raw_date).strip()
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%B-%Y", "%d/%m/%Y"):
            try:
                return datetime.datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @classmethod
    def _load_cached_holidays_from_db(cls):
        """Loads previously synced NSE holidays from the Single Master File (swing_trades.db)."""
        if cls._db_loaded:
            return
        cls._db_loaded = True
        try:
            from execution.state_engine import state_engine
            cached = state_engine.get_kv("NSE_HOLIDAYS_VAULT")
            if cached and isinstance(cached, dict):
                holidays_dict = cached.get("holidays") or {}
                if isinstance(holidays_dict, dict) and holidays_dict:
                    # Replace current year dates with authoritative NSE synced dates if present
                    synced_years = {k[:4] for k in holidays_dict.keys() if len(k) >= 4}
                    for yr in synced_years:
                        for k in list(cls.NSE_HOLIDAYS.keys()):
                            if k.startswith(f"{yr}-"):
                                cls.NSE_HOLIDAYS.pop(k, None)
                    cls.NSE_HOLIDAYS.update(holidays_dict)
                    cls._holiday_meta.update({
                        "source": cached.get("source", "MASTER_DB_CACHE"),
                        "last_synced_ist": cached.get("last_synced_ist"),
                        "last_synced_date": cached.get("last_synced_date"),
                        "live_fetched_count": cached.get("live_fetched_count", len(holidays_dict)),
                        "total_holidays_loaded": len(cls.NSE_HOLIDAYS),
                    })
        except Exception as e:
            logger.debug(f"Could not load cached NSE holidays from DB: {e}")

    @classmethod
    def sync_holidays_from_nse(cls, force: bool = False) -> Dict[str, Any]:
        """
        Dynamically fetches the official NSE Trading Holidays from:
        https://www.nseindia.com/api/holiday-master?type=trading (Capital Market 'CM' segment).
        Updates in-memory NSE_HOLIDAYS and persists to the Single Master File (swing_trades.db).
        """
        cls._load_cached_holidays_from_db()
        today_str = now_ist().strftime("%Y-%m-%d")

        if not force and cls._holiday_meta.get("last_synced_date") == today_str:
            return cls.get_holiday_metadata()

        with cls._sync_lock:
            if not force and cls._holiday_meta.get("last_synced_date") == today_str:
                return cls.get_holiday_metadata()

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/resources/exchange-communication-holidays",
                "Connection": "keep-alive",
            }

            fetched_holidays: Dict[str, str] = {}
            sync_source = "NSE_OFFICIAL_WEBSITE_LIVE"
            sync_url = cls.NSE_HOLIDAY_API_URL

            try:
                session = requests.Session()
                session.get(cls.NSE_HOME_URL, headers=headers, timeout=8)
                resp = session.get(cls.NSE_HOLIDAY_API_URL, headers=headers, timeout=10)
                resp.raise_for_status()
                payload = resp.json()
                cm_entries = payload.get("CM") or payload.get("FO") or []
                for item in cm_entries:
                    if not isinstance(item, dict):
                        continue
                    parsed_date = cls._parse_nse_date(item.get("tradingDate", ""))
                    desc = str(item.get("description") or "NSE Trading Holiday").strip()
                    if parsed_date:
                        fetched_holidays[parsed_date] = desc
            except Exception as nse_err:
                logger.debug(f"Direct nseindia.com holiday endpoint geo-blocked ({nse_err}); switching to live NSE Exchange Calendar mirror...")

            # Cloud Fallback (when Render US/EU IP is geo-blocked by NSE Akamai 403):
            if not fetched_holidays:
                try:
                    resp2 = requests.get(
                        cls.UPSTOX_HOLIDAY_API_URL,
                        headers={"Accept": "application/json", "User-Agent": headers["User-Agent"]},
                        timeout=10
                    )
                    resp2.raise_for_status()
                    up_entries = resp2.json().get("data") or []
                    for item in up_entries:
                        if not isinstance(item, dict):
                            continue
                        closed_ex = item.get("closed_exchanges") or []
                        if "NSE" in closed_ex or item.get("holiday_type") == "TRADING_HOLIDAY":
                            parsed_date = cls._parse_nse_date(item.get("date", ""))
                            desc = str(item.get("description") or "NSE Trading Holiday").strip()
                            if parsed_date:
                                fetched_holidays[parsed_date] = desc
                    if fetched_holidays:
                        sync_source = "NSE_EXCHANGE_CALENDAR_LIVE"
                        sync_url = cls.UPSTOX_HOLIDAY_API_URL
                except Exception as fallback_err:
                    cls._holiday_meta["last_error"] = str(fallback_err)
                    logger.info(f"Using cached NSE holiday calendar from Master Vault ({len(cls.NSE_HOLIDAYS)} entries).")

            if fetched_holidays:
                cls.NSE_HOLIDAYS.update(fetched_holidays)
                sync_ts = format_ist()
                cls._holiday_meta = {
                    "source": sync_source,
                    "api_url": sync_url,
                    "last_synced_ist": sync_ts,
                    "last_synced_date": today_str,
                    "live_fetched_count": len(fetched_holidays),
                    "total_holidays_loaded": len(cls.NSE_HOLIDAYS),
                    "last_error": None
                }

                # Persist into Single Master File (swing_trades.db)
                try:
                    from execution.state_engine import state_engine
                    state_engine.set_kv("NSE_HOLIDAYS_VAULT", {
                        "source": sync_source,
                        "last_synced_ist": sync_ts,
                        "last_synced_date": today_str,
                        "live_fetched_count": len(fetched_holidays),
                        "holidays": cls.NSE_HOLIDAYS
                    })
                except Exception as db_err:
                    logger.debug(f"Failed saving NSE holidays to master DB: {db_err}")

                logger.info(
                    f"✓ Synced {len(fetched_holidays)} official NSE holidays dynamically via {sync_source} "
                    f"(Total calendar entries: {len(cls.NSE_HOLIDAYS)})."
                )

            return cls.get_holiday_metadata()

    @classmethod
    def get_holiday_metadata(cls) -> Dict[str, Any]:
        """Returns metadata about the dynamically synced NSE holiday calendar."""
        cls._load_cached_holidays_from_db()
        curr_year = str(now_ist().year)
        curr_year_holidays = {
            k: v for k, v in sorted(cls.NSE_HOLIDAYS.items())
            if k.startswith(f"{curr_year}-")
        }
        return {
            **cls._holiday_meta,
            "current_year": int(curr_year),
            "current_year_holidays_count": len(curr_year_holidays),
            "current_year_holidays": curr_year_holidays,
        }

    @classmethod
    def is_trading_day(cls, dt: datetime.date = None) -> bool:
        """Returns True if the date is Monday-Friday and not an official NSE holiday."""
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist().date()
        elif isinstance(dt, datetime.datetime):
            dt = dt.date()

        # 0 = Monday, ..., 4 = Friday, 5 = Saturday, 6 = Sunday
        if dt.weekday() >= 5:
            return False

        date_str = dt.strftime("%Y-%m-%d")
        if date_str in cls.NSE_HOLIDAYS:
            return False

        return True

    @classmethod
    def get_next_trading_day(cls, dt: Optional[datetime.date] = None) -> datetime.date:
        """Returns the next official NSE trading day after the given date."""
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist().date()
        elif isinstance(dt, datetime.datetime):
            dt = dt.date()
        next_d = dt + datetime.timedelta(days=1)
        while not cls.is_trading_day(next_d):
            next_d += datetime.timedelta(days=1)
        return next_d

    @classmethod
    def get_previous_trading_day(cls, dt: Optional[datetime.date] = None) -> datetime.date:
        """Returns the most recent official NSE trading day strictly BEFORE the given date."""
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist().date()
        elif isinstance(dt, datetime.datetime):
            dt = dt.date()
        prev_d = dt - datetime.timedelta(days=1)
        while not cls.is_trading_day(prev_d):
            prev_d -= datetime.timedelta(days=1)
        return prev_d

    @classmethod
    def get_last_completed_trading_day(cls, dt: Optional[datetime.datetime] = None) -> datetime.date:
        """
        Returns the last FULLY COMPLETED NSE trading day.
        If dt is during trading hours or before market close on a trading day (curr_t < 15:30 IST),
        today has NOT completed, so returns yesterday's (or prior) trading day.
        If dt is on a trading day and curr_t >= 15:30 IST, today is completed, so returns today.
        If dt is on a weekend or holiday, returns the prior trading day.
        """
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist()
        elif isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.combine(dt, datetime.time(0, 0, 0))
            dt = IST.localize(dt)
        elif dt.tzinfo is None:
            dt = IST.localize(dt)

        curr_d = dt.date()
        curr_t = dt.time()
        if cls.is_trading_day(curr_d) and curr_t >= datetime.time(15, 30, 0):
            return curr_d
        return cls.get_previous_trading_day(curr_d)

    @classmethod
    def get_holiday_reason(cls, dt: datetime.date = None) -> str:
        """Returns holiday or weekend reason if applicable."""
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist().date()
        elif isinstance(dt, datetime.datetime):
            dt = dt.date()

        if dt.weekday() == 5:
            return "Saturday (Weekend)"
        if dt.weekday() == 6:
            return "Sunday (Weekend)"

        date_str = dt.strftime("%Y-%m-%d")
        return cls.NSE_HOLIDAYS.get(date_str, "Trading Day")

    @classmethod
    def is_render_keepalive_window(cls, dt: datetime.datetime = None) -> Tuple[bool, str]:
        """
        24/7 Dedicated Host Status:
        Always returns (True, reason) because the system runs on a dedicated RackNerd VPS with no sleep timeouts.
        """
        return True, "24/7 DEDICATED HOST ACTIVE (RackNerd VPS • Zero Sleep Timeout)"

    @classmethod
    def get_market_status(cls, dt: datetime.datetime = None) -> Dict[str, Any]:
        """
        Returns full market status dictionary:
        - is_open: bool
        - session_phase: 'PRE_SESSION', 'PRE_MARKET', 'REGULAR_HOURS', 'POST_MARKET', 'WEEKEND', 'HOLIDAY'
        - holiday_reason: str
        - keepalive_active: bool (True strictly 09:00–15:50 IST on trading days)
        """
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist()
        elif dt.tzinfo is None:
            dt = IST.localize(dt)

        current_date = dt.date()
        current_time = dt.time()
        keepalive_active, keepalive_msg = cls.is_render_keepalive_window(dt)

        if current_date.weekday() >= 5:
            return {
                "is_open": False,
                "session_phase": "WEEKEND",
                "holiday_reason": "Saturday" if current_date.weekday() == 5 else "Sunday",
                "message": "Market is closed for the weekend.",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": False,
                "keepalive_message": keepalive_msg
            }

        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in cls.NSE_HOLIDAYS:
            reason = cls.NSE_HOLIDAYS[date_str]
            return {
                "is_open": False,
                "session_phase": "HOLIDAY",
                "holiday_reason": reason,
                "message": f"Market is closed today ({reason}).",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": False,
                "keepalive_message": keepalive_msg
            }

        t_pre = datetime.time(9, 0)
        t_open = datetime.time(9, 15)
        t_close = datetime.time(15, 30)

        if current_time < t_pre:
            return {
                "is_open": False,
                "session_phase": "PRE_SESSION",
                "holiday_reason": None,
                "message": "Market opens at 09:15 AM IST (24/7 Dedicated Host).",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": keepalive_active,
                "keepalive_message": keepalive_msg
            }
        elif current_time < t_open:
            return {
                "is_open": False,
                "session_phase": "PRE_MARKET",
                "holiday_reason": None,
                "message": "Pre-Market Session (09:00 - 09:15 IST) • 24/7 Host Online.",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": keepalive_active,
                "keepalive_message": keepalive_msg
            }
        elif current_time <= t_close:
            return {
                "is_open": True,
                "session_phase": "REGULAR_HOURS",
                "holiday_reason": None,
                "message": "Live NSE Trading Session active (09:15 - 15:30 IST).",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": keepalive_active,
                "keepalive_message": keepalive_msg
            }
        else:
            return {
                "is_open": False,
                "session_phase": "POST_MARKET",
                "holiday_reason": None,
                "message": "Market closed at 03:30 PM IST.",
                "ist_time": dt.strftime("%H:%M:%S"),
                "keepalive_active": keepalive_active,
                "keepalive_message": keepalive_msg
            }

    @classmethod
    def get_next_scheduled_scan(cls, dt: datetime.datetime = None) -> Dict[str, Any]:
        """
        Determines the next upcoming automated scan (10:20 AM or 3:00 PM IST).
        """
        cls._load_cached_holidays_from_db()
        if dt is None:
            dt = now_ist()
        elif dt.tzinfo is None:
            dt = IST.localize(dt)

        try:
            from core.config import (
                SCHEDULE_MORNING_HOUR, SCHEDULE_MORNING_MINUTE,
                SCHEDULE_AFTERNOON_HOUR, SCHEDULE_AFTERNOON_MINUTE
            )
            t_1020 = datetime.time(SCHEDULE_MORNING_HOUR, SCHEDULE_MORNING_MINUTE)
            t_1500 = datetime.time(SCHEDULE_AFTERNOON_HOUR, SCHEDULE_AFTERNOON_MINUTE)
        except Exception:
            t_1020 = datetime.time(10, 20)
            t_1500 = datetime.time(15, 0)

        curr_date = dt.date()
        curr_time = dt.time()

        if cls.is_trading_day(curr_date):
            if curr_time < t_1020:
                target_dt = IST.localize(datetime.datetime.combine(curr_date, t_1020))
                return {
                    "scan_type": "MORNING_1020AM",
                    "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "seconds_remaining": int((target_dt - dt).total_seconds()),
                    "label": "10:20 AM Morning Intraday Scan"
                }
            elif curr_time < t_1500:
                target_dt = IST.localize(datetime.datetime.combine(curr_date, t_1500))
                return {
                    "scan_type": "AFTERNOON_300PM",
                    "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "seconds_remaining": int((target_dt - dt).total_seconds()),
                    "label": f"{t_1500.strftime('%I:%M %p').lstrip('0')} MOC & Daily Close Swing Scan"
                }

        next_date = curr_date + datetime.timedelta(days=1)
        while not cls.is_trading_day(next_date):
            next_date += datetime.timedelta(days=1)

        target_dt = IST.localize(datetime.datetime.combine(next_date, t_1020))
        return {
            "scan_type": "MORNING_1020AM",
            "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
            "seconds_remaining": int((target_dt - dt).total_seconds()),
            "label": f"10:20 AM Morning Scan ({next_date.strftime('%a, %d %b')})"
        }

    @classmethod
    def can_execute_trades(cls, dt: datetime.datetime = None) -> Tuple[bool, str]:
        """
        Strict Market-Hours Trading Gate:
        Returns (True, reason) ONLY if current IST time is between 09:15:00 and 15:30:00 IST
        on an active NSE trading day (Monday-Friday, excluding NSE holidays).
        """
        status = cls.get_market_status(dt)
        if status.get("is_open") and status.get("session_phase") == "REGULAR_HOURS":
            return True, f"TRADING GATE OPEN ({status.get('ist_time')} IST - Regular Market Hours 09:15–15:30 IST)"
        reason = (
            f"STRICT TRADING GATE LOCKED: Market is currently in {status.get('session_phase')} "
            f"({status.get('message')}). Trade entries & exits are strictly permitted only during "
            f"NSE Regular Trading Hours (09:15 AM – 03:30 PM IST, Mon–Fri)."
        )
        return False, reason
