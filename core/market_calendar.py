"""
NSE Market Calendar & Trading Session Detector
Manages National Stock Exchange (NSE) trading hours, holidays, and schedule checks.
"""
import datetime
from typing import Dict, Any, Tuple
from core.data_quality import IST, now_ist

class NSEMarketCalendar:
    """
    Standard NSE Indian Equities Market Calendar:
    Normal Trading Hours: 09:15 to 15:30 IST (Monday - Friday)
    Pre-Market: 09:00 to 09:08 IST
    Muhurat Trading: Scheduled separately on Diwali
    """
    
    # Official NSE Holidays (YYYY-MM-DD)
    NSE_HOLIDAYS = {
        # 2024
        "2024-01-22": "Special Holiday (Ayodhya Pran Pratishtha)",
        "2024-01-26": "Republic Day",
        "2024-03-08": "Mahashivratri",
        "2024-03-25": "Holi",
        "2024-03-29": "Good Friday",
        "2024-04-11": "Id-Ul-Fitr (Ramzan Id)",
        "2024-04-17": "Shri Ram Navami",
        "2024-05-01": "Maharashtra Day",
        "2024-05-20": "General Elections",
        "2024-06-17": "Bakri Id",
        "2024-07-17": "Muharram",
        "2024-08-15": "Independence Day",
        "2024-10-02": "Mahatma Gandhi Jayanti",
        "2024-11-01": "Diwali Laxmi Pujan (Muhurat session evening)",
        "2024-11-15": "Gurunanak Jayanti",
        "2024-11-20": "Maharashtra Assembly Election",
        "2024-12-25": "Christmas",
        
        # 2025
        "2025-01-26": "Republic Day",
        "2025-02-26": "Mahashivratri",
        "2025-03-14": "Holi",
        "2025-03-31": "Id-Ul-Fitr",
        "2025-04-10": "Mahavir Jayanti",
        "2025-04-14": "Dr. Ambedkar Jayanti",
        "2025-04-18": "Good Friday",
        "2025-05-01": "Maharashtra Day",
        "2025-06-07": "Bakri Id",
        "2025-08-15": "Independence Day",
        "2025-08-27": "Ganesh Chaturthi",
        "2025-10-02": "Mahatma Gandhi Jayanti",
        "2025-10-21": "Diwali Laxmi Pujan",
        "2025-10-22": "Diwali Balipratipada",
        "2025-11-05": "Gurunanak Jayanti",
        "2025-12-25": "Christmas",

        # 2026
        "2026-01-26": "Republic Day",
        "2026-02-15": "Mahashivratri",
        "2026-03-03": "Holi",
        "2026-03-20": "Id-Ul-Fitr",
        "2026-04-03": "Good Friday",
        "2026-04-14": "Dr. Ambedkar Jayanti",
        "2026-05-01": "Maharashtra Day",
        "2026-05-27": "Bakri Id",
        "2026-06-25": "Muharram",
        "2026-08-15": "Independence Day",
        "2026-09-14": "Ganesh Chaturthi",
        "2026-10-02": "Mahatma Gandhi Jayanti",
        "2026-10-20": "Dussehra",
        "2026-11-08": "Diwali Laxmi Pujan",
        "2026-11-24": "Gurunanak Jayanti",
        "2026-12-25": "Christmas",

        # 2027
        "2027-01-26": "Republic Day",
        "2027-03-22": "Holi",
        "2027-03-26": "Good Friday",
        "2027-04-14": "Dr. Ambedkar Jayanti",
        "2027-05-01": "Maharashtra Day",
        "2027-08-15": "Independence Day",
        "2027-10-02": "Mahatma Gandhi Jayanti",
        "2027-12-25": "Christmas",
    }

    @classmethod
    def is_trading_day(cls, dt: datetime.date = None) -> bool:
        """Returns True if the date is a Monday-Friday and not an NSE official holiday."""
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
    def get_holiday_reason(cls, dt: datetime.date = None) -> str:
        """Returns holiday reason if applicable."""
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
    def get_market_status(cls, dt: datetime.datetime = None) -> Dict[str, Any]:
        """
        Returns full market status dictionary:
        - is_open: bool
        - session_phase: 'CLOSED', 'PRE_MARKET', 'REGULAR_HOURS', 'POST_MARKET', 'WEEKEND', 'HOLIDAY'
        - next_open: str
        - holiday_reason: str
        """
        if dt is None:
            dt = now_ist()
        elif dt.tzinfo is None:
            dt = IST.localize(dt)

        current_date = dt.date()
        current_time = dt.time()

        if current_date.weekday() >= 5:
            return {
                "is_open": False,
                "session_phase": "WEEKEND",
                "holiday_reason": "Saturday" if current_date.weekday() == 5 else "Sunday",
                "message": "Market is closed for the weekend.",
                "ist_time": dt.strftime("%H:%M:%S")
            }

        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in cls.NSE_HOLIDAYS:
            reason = cls.NSE_HOLIDAYS[date_str]
            return {
                "is_open": False,
                "session_phase": "HOLIDAY",
                "holiday_reason": reason,
                "message": f"Market is closed today ({reason}).",
                "ist_time": dt.strftime("%H:%M:%S")
            }

        t_pre = datetime.time(9, 0)
        t_open = datetime.time(9, 15)
        t_close = datetime.time(15, 30)

        if current_time < t_pre:
            return {
                "is_open": False,
                "session_phase": "PRE_SESSION",
                "holiday_reason": None,
                "message": "Market opens at 09:15 AM IST.",
                "ist_time": dt.strftime("%H:%M:%S")
            }
        elif current_time < t_open:
            return {
                "is_open": False,
                "session_phase": "PRE_MARKET",
                "holiday_reason": None,
                "message": "Pre-Market Order Matching (09:00 - 09:15 IST).",
                "ist_time": dt.strftime("%H:%M:%S")
            }
        elif current_time <= t_close:
            return {
                "is_open": True,
                "session_phase": "REGULAR_HOURS",
                "holiday_reason": None,
                "message": "Live Trading Session active.",
                "ist_time": dt.strftime("%H:%M:%S")
            }
        else:
            return {
                "is_open": False,
                "session_phase": "POST_MARKET",
                "holiday_reason": None,
                "message": "Market closed at 03:30 PM IST.",
                "ist_time": dt.strftime("%H:%M:%S")
            }

    @classmethod
    def get_next_scheduled_scan(cls, dt: datetime.datetime = None) -> Dict[str, Any]:
        """
        Determines the next upcoming automated scan (10:20 AM or 3:05 PM IST).
        """
        if dt is None:
            dt = now_ist()
        elif dt.tzinfo is None:
            dt = IST.localize(dt)

        t_1020 = datetime.time(10, 20)
        t_1505 = datetime.time(15, 5)

        curr_date = dt.date()
        curr_time = dt.time()

        # If today is a trading day:
        if cls.is_trading_day(curr_date):
            if curr_time < t_1020:
                target_dt = IST.localize(datetime.datetime.combine(curr_date, t_1020))
                scan_type = "MORNING_1020AM"
                return {
                    "scan_type": scan_type,
                    "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "seconds_remaining": int((target_dt - dt).total_seconds()),
                    "label": "10:20 AM Morning Intraday Scan"
                }
            elif curr_time < t_1505:
                target_dt = IST.localize(datetime.datetime.combine(curr_date, t_1505))
                scan_type = "AFTERNOON_305PM"
                return {
                    "scan_type": scan_type,
                    "target_time": target_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "seconds_remaining": int((target_dt - dt).total_seconds()),
                    "label": "3:05 PM MOC & Daily Close Swing Scan"
                }

        # Otherwise, search for the next trading day's 10:20 AM
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
        Otherwise returns (False, strict_gate_reason).
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

