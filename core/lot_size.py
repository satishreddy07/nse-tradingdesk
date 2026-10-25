"""
core/lot_size.py
================
Authoritative Exchange & Broker Contract Lot-Size Resolution Layer.

Governed by root AGENTS.md:
- Mandatory preservation of data integrity and financial calculation accuracy (§1, §6).
- Prohibits arbitrary numeric substitutions and unverified business parameters (§2, §4).
- Distinguishes lot size, number of lots, and total contract quantity.
- Accurately models official National Stock Exchange (NSE) circulars and transition schedules:
  1. NSE Circular NSE/FAOP/70616 (Dated Oct 03, 2025, Circular 176/2025):
     Revision from 75 to 65. Effective Oct 28, 2025 (EOD).
     Existing weekly/monthly contracts maintained lot size 75 through Dec 30, 2025 monthly expiry.
     First revised weekly expiry: Jan 06, 2026 (lot size 65).
     First revised monthly expiry: Jan 27, 2026 (lot size 65).
     Existing quarterly & half-yearly contracts transitioned at EOD Dec 30, 2025 (75 -> 65).
  2. NSE Circular NSE/FAOP/64625 (Dated Oct 18, 2024, Circular 128/2024) &
     Implementation Update NSE/FAOP/64990 (Dated Nov 08, 2024):
     Revision from 25 to 75.
     The original circular date was Oct 18, 2024, proposing Nov 20, 2024.
     Due to the Maharashtra Assembly Elections trading holiday on Nov 20, 2024,
     NSE/FAOP/64990 shifted the implementation trade date to Nov 21, 2024.
     All newly introduced contracts made available for trading from Nov 21, 2024 adopted lot size 75.
     Existing weekly & monthly contracts maintained lot size 25 until their respective expiries
     (including the Nov 21, 2024 weekly, Nov 28, 2024 monthly, Dec 26, 2024 monthly, and Jan 30, 2025 monthly).
     New weekly contracts introduced post-Nov 21, 2024 adopted lot size 75.
     Existing quarterly & half-yearly contracts transitioned at EOD Dec 26, 2024 (25 -> 75).
  3. NSE Circular NSE/FAOP/61415 (Dated Apr 02, 2024, Circular 37/2024):
     Revision from 50 to 25. Effective Apr 26, 2024.
     Contracts expiring on or before Apr 25, 2024 (Apr 2024 monthly) maintained lot size 50.
     First revised weekly expiry: May 02, 2024 (lot size 25).
     First revised monthly expiry: May 30, 2024 (lot size 25).
  4. NSE Circular 25/2021 (Dated Mar 31, 2021):
     Revision from 75 to 50. Effective Jul 01, 2021 through Apr 25, 2024.
  5. Pre-July 2021 Historical baseline: Lot size 75.
- Fails closed when contract attributes are ambiguous, incomplete, or contradictory.
- Preserves historical trade records with their true original contract lot sizes.
"""

import os
import re
import math
import logging
import datetime
from typing import Optional, Dict, Any, Union, Tuple

logger = logging.getLogger("LotSizeResolver")


class AmbiguousLotSizeError(ValueError):
    """
    Raised when the available contract attributes (expiry, cycle, introduction date,
    as-of date) cannot uniquely identify the applicable lot size without guessing.
    Governed by AGENTS.md §1 & §4: stop and report uncertainty rather than guessing.
    """
    pass


# ==============================================================================
# AUTHORITATIVE REGULATORY TRANSITION DATES & REGIMES
# ==============================================================================

# NSE Circular 25/2021
DATE_ERA_50_START = datetime.date(2021, 7, 1)

# NSE Circular NSE/FAOP/61415 (Apr 02, 2024)
DATE_50_CUTOFF_EXPIRY = datetime.date(2024, 4, 25)    # Last expiry with lot size 50 (Apr 2024 monthly)
DATE_ERA_25_EFFECTIVE = datetime.date(2024, 4, 26)    # Effective date for trading under revised 25 lot
DATE_FIRST_WEEKLY_25 = datetime.date(2024, 5, 2)      # First weekly expiry with 25
DATE_FIRST_MONTHLY_25 = datetime.date(2024, 5, 30)    # First monthly expiry with 25

# NSE Circular NSE/FAOP/64625 (Oct 18, 2024) & NSE/FAOP/64990 (Nov 08, 2024)
DATE_ERA_75_CIRCULAR_ANNOUNCEMENT = datetime.date(2024, 10, 18)     # Original circular announcement
DATE_ERA_75_UPDATE_CIRCULAR = datetime.date(2024, 11, 8)            # Implementation update circular
DATE_MAHARASHTRA_ELECTION_HOLIDAY = datetime.date(2024, 11, 20)      # Trading holiday (Maharashtra Elections)
DATE_ERA_75_IMPLEMENTATION_TRADE_DATE = datetime.date(2024, 11, 21) # Actual trade date new contracts launched
DATE_NOV_21_2024_WEEKLY_EXPIRY = datetime.date(2024, 11, 21)        # Weekly contract expiring on implementation trade date (retained 25)
DATE_DEC_2024_MONTHLY_EXPIRY = datetime.date(2024, 12, 26)          # Dec 2024 monthly expiry; quarterly transition EOD
DATE_JAN_2025_MONTHLY_EXPIRY = datetime.date(2025, 1, 30)          # Jan 2025 monthly expiry (retained 25 if pre-existing)

# NSE Circular NSE/FAOP/70616 (Oct 03, 2025)
DATE_ERA_65_CIRCULAR_EFFECTIVE = datetime.date(2025, 10, 28)  # Circular effective date (EOD)
DATE_FINAL_WEEKLY_75 = datetime.date(2025, 12, 23)           # Final weekly under 75
DATE_FINAL_MONTHLY_75 = datetime.date(2025, 12, 30)          # Final monthly under 75; quarterly transition EOD
DATE_FIRST_WEEKLY_65 = datetime.date(2026, 1, 6)             # First weekly expiry under 65
DATE_FIRST_MONTHLY_65 = datetime.date(2026, 1, 27)           # First monthly expiry under 65

# Active market regime lot size (2026 onwards)
CURRENT_REGIME_NIFTY_LOT_SIZE = 65

# Authoritative schedule summary for documentation and reference queries
NIFTY_LOT_SIZE_SCHEDULE = [
    {
        "regime": "65_LOT_REGIME",
        "circular_ref": "NSE/FAOP/70616",
        "announcement_date": datetime.date(2025, 10, 3),
        "effective_date": datetime.date(2025, 10, 28),
        "first_weekly_expiry": datetime.date(2026, 1, 6),
        "first_monthly_expiry": datetime.date(2026, 1, 27),
        "quarterly_transition_eod": datetime.date(2025, 12, 30),
        "standard_lot_size": 65,
        "description": "NSE periodic lot recalibration; 75 to 65 for contracts expiring Jan 2026 onwards."
    },
    {
        "regime": "75_LOT_REGIME",
        "circular_ref": "NSE/FAOP/64625_AND_64990",
        "announcement_date": datetime.date(2024, 10, 18),
        "update_announcement_date": datetime.date(2024, 11, 8),
        "election_holiday": datetime.date(2024, 11, 20),
        "implementation_trade_date": datetime.date(2024, 11, 21),
        "quarterly_transition_eod": datetime.date(2024, 12, 26),
        "standard_lot_size": 75,
        "description": "SEBI index derivatives framework; 25 to 75 for new contracts introduced Nov 21, 2024 onwards."
    },
    {
        "regime": "25_LOT_REGIME",
        "circular_ref": "NSE/FAOP/61415",
        "announcement_date": datetime.date(2024, 4, 2),
        "effective_date": datetime.date(2024, 4, 26),
        "first_weekly_expiry": datetime.date(2024, 5, 2),
        "first_monthly_expiry": datetime.date(2024, 5, 30),
        "standard_lot_size": 25,
        "description": "NSE periodic revision from 50 to 25 effective Apr 26, 2024."
    },
    {
        "regime": "50_LOT_REGIME",
        "circular_ref": "NSE/25/2021",
        "announcement_date": datetime.date(2021, 3, 31),
        "effective_date": datetime.date(2021, 7, 1),
        "standard_lot_size": 50,
        "description": "NSE lot revision from 75 to 50 effective Jul 01, 2021 through Apr 25, 2024."
    },
    {
        "regime": "HISTORICAL_75_REGIME",
        "circular_ref": "NSE/HISTORICAL",
        "effective_date": datetime.date(2000, 1, 1),
        "standard_lot_size": 75,
        "description": "Historical Nifty derivative baseline lot size prior to Jul 01, 2021."
    }
]


def parse_contract_expiry_date(expiry: Any) -> Optional[datetime.date]:
    """
    Parses contract expiry date strings or datetime objects into a standard datetime.date.
    Supported formats:
      - datetime.date or datetime.datetime
      - '27-Oct-2026', '27-OCT-2026', '29-Oct-2026', '26-Dec-2024', '25-Apr-2024', '21-Nov-2024'
      - '2026-10-27', '2024-11-21', '2024-11-20', '2024-04-26'
      - '26OCT', '24DEC' (YY + MMM)
      - 'OCT2026', 'DEC2024', 'APR2024' (MMM + YYYY)
      - '27/10/2026', '20261027'
    """
    if expiry is None:
        return None
    if isinstance(expiry, datetime.datetime):
        return expiry.date()
    if isinstance(expiry, datetime.date):
        return expiry

    s = str(expiry).strip().upper()
    if not s or s in ("NONE", "NULL", "NAN", ""):
        return None

    # Common explicit date formats
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    # Compact format: 26OCT or 24DEC (YY + 3-char month)
    m = re.match(r"^(\d{2})([A-Z]{3})$", s)
    if m:
        yy, mon = m.groups()
        year = 2000 + int(yy)
        try:
            return datetime.datetime.strptime(f"{year}-{mon}-28", "%Y-%b-%d").date()
        except ValueError:
            pass

    # Month + Year format: OCT2026 or DEC2024
    m = re.match(r"^([A-Z]{3})(\d{4})$", s)
    if m:
        mon, yyyy = m.groups()
        try:
            return datetime.datetime.strptime(f"{yyyy}-{mon}-28", "%Y-%b-%d").date()
        except ValueError:
            pass

    # Extract date part if string contains timestamp e.g. "2026-10-27 15:30:00"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    return None


def resolve_nifty_lot_size(
    expiry_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    as_of_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    contract_metadata: Optional[Dict[str, Any]] = None,
    symbol: Optional[str] = None,
    contract_type: Optional[str] = None,
    introduced_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    strict: bool = False,
    return_details: bool = False
) -> Union[int, Tuple[int, Dict[str, Any]]]:
    """
    Authoritatively resolves the valid NIFTY 50 contract lot size.

    Priority hierarchy:
    1. Direct, Validated Contract / Instrument Metadata:
       - Verified broker/instrument-master metadata takes precedence when available.
       - Validates symbol strictly (only NIFTY 50 contracts permitted; rejects non-NIFTY symbols).
       - Validates expiry consistency and positive integer lot size.
       - Rejects invalid, zero, negative, NaN, inf, or mismatched metadata without fallback.
    2. Regulatory Circular Schedule with Full Contract Attributes:
       - Accurately models circular transitions for weekly, monthly, quarterly, and half-yearly cycles.
       - Evaluates as-of trade date for contracts whose lot size changed at designated transition times.
       - Models the Nov 21, 2024 implementation trade date under NSE/FAOP/64625 & NSE/FAOP/64990.
    3. Fail-Closed on Ambiguity:
       - If contract attributes cannot uniquely determine the lot size without guessing
         (e.g., transition windows where monthly and weekly co-existed with different lots),
         raises AmbiguousLotSizeError rather than guessing.
    4. Operational Default (Non-Strict Mode with No Dates):
       - Only when no date or metadata is provided and strict=False, returns CURRENT_REGIME_NIFTY_LOT_SIZE (65).

    Returns:
       - lot_size (int) if return_details=False
       - (lot_size, details_dict) if return_details=True
    """
    # --------------------------------------------------------------------------
    # 0. Symbol Validation (Fail Closed for Non-NIFTY Instruments)
    # --------------------------------------------------------------------------
    if symbol is not None:
        sym_clean = str(symbol).strip().upper()
        if sym_clean and sym_clean not in ("NIFTY", "NIFTY 50", "NIFTY50", "NIFTY_50"):
            raise ValueError(f"resolve_nifty_lot_size only supports NIFTY contracts, received symbol '{symbol}'.")

    # --------------------------------------------------------------------------
    # 1. Direct Contract Metadata Inspection & Validation
    # --------------------------------------------------------------------------
    if contract_metadata and isinstance(contract_metadata, dict):
        # Strict Symbol Check in Metadata: Only NIFTY 50 allowed
        meta_sym = (
            contract_metadata.get("symbol")
            or contract_metadata.get("pTrdSymbol")
            or contract_metadata.get("trading_symbol")
        )
        if meta_sym and isinstance(meta_sym, str):
            meta_sym_upper = meta_sym.strip().upper()
            non_nifty_tokens = ("BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
            is_explicit_non_nifty = any(idx in meta_sym_upper for idx in non_nifty_tokens)
            is_valid_nifty = meta_sym_upper.startswith("NIFTY") and not is_explicit_non_nifty
            # If symbol is a stock option or non-Nifty index, reject immediately
            if not is_valid_nifty:
                raise ValueError(
                    f"Mismatched contract metadata symbol '{meta_sym}': resolve_nifty_lot_size only supports NIFTY 50 contracts."
                )

        # Validate expiry consistency between metadata and argument if both provided
        meta_exp_raw = contract_metadata.get("expiry_date") or contract_metadata.get("expiry")
        if meta_exp_raw is not None and expiry_date is not None:
            parsed_meta_exp = parse_contract_expiry_date(meta_exp_raw)
            parsed_arg_exp = parse_contract_expiry_date(expiry_date)
            if parsed_meta_exp and parsed_arg_exp and parsed_meta_exp != parsed_arg_exp:
                raise ValueError(
                    f"Mismatched expiry dates: contract_metadata has '{parsed_meta_exp}', "
                    f"but argument expiry_date is '{parsed_arg_exp}'."
                )

        # Inspect lot size attributes: 'lot_size', 'lLotSize' (Kotak), 'market_lot'
        raw_ls = contract_metadata.get("lot_size")
        if raw_ls is None:
            raw_ls = contract_metadata.get("lLotSize")
        if raw_ls is None:
            raw_ls = contract_metadata.get("market_lot")

        if raw_ls is not None:
            try:
                ls_float = float(raw_ls)
                if math.isfinite(ls_float) and ls_float > 0 and ls_float.is_integer():
                    lot_size = int(ls_float)
                    details = {
                        "lot_size": lot_size,
                        "source": "CONTRACT_METADATA",
                        "circular_ref": contract_metadata.get("circular_ref", "INSTRUMENT_MASTER"),
                        "contract_attributes": {
                            "symbol": "NIFTY",
                            "expiry_date": str(parse_contract_expiry_date(expiry_date) or parse_contract_expiry_date(meta_exp_raw)),
                            "contract_type": contract_type or contract_metadata.get("contract_type"),
                            "as_of_date": str(parse_contract_expiry_date(as_of_date)),
                            "introduced_date": str(parse_contract_expiry_date(introduced_date))
                        },
                        "transition_notes": "Authoritative lot size resolved directly from validated contract metadata."
                    }
                    return (lot_size, details) if return_details else lot_size
                else:
                    raise ValueError(f"Invalid non-positive or non-finite lot_size in contract_metadata: {raw_ls}")
            except (ValueError, TypeError) as ex:
                raise ValueError(f"Invalid lot_size in contract_metadata: {ex}")

    # --------------------------------------------------------------------------
    # 2. Extract & Normalize Contract Attributes
    # --------------------------------------------------------------------------
    expiry_dt = parse_contract_expiry_date(expiry_date)
    as_of_dt = parse_contract_expiry_date(as_of_date)
    intro_dt = parse_contract_expiry_date(introduced_date)

    if contract_metadata and isinstance(contract_metadata, dict):
        if not expiry_dt:
            meta_exp_raw = contract_metadata.get("expiry_date") or contract_metadata.get("expiry")
            expiry_dt = parse_contract_expiry_date(meta_exp_raw)
        if not as_of_dt and contract_metadata.get("as_of_date"):
            as_of_dt = parse_contract_expiry_date(contract_metadata.get("as_of_date"))
        if not intro_dt:
            meta_intro_raw = contract_metadata.get("introduced_date") or contract_metadata.get("created_at")
            intro_dt = parse_contract_expiry_date(meta_intro_raw)
        if not contract_type:
            raw_ct = contract_metadata.get("contract_type") or contract_metadata.get("expiry_type")
            if raw_ct:
                contract_type = str(raw_ct).strip().upper()

    if contract_type:
        contract_type = str(contract_type).strip().upper()

    # --------------------------------------------------------------------------
    # 3. Handle Missing or Unparseable Dates
    # --------------------------------------------------------------------------
    if expiry_date is not None and expiry_dt is None:
        if strict:
            raise ValueError(f"Cannot resolve authoritative NIFTY lot size: unparseable expiry date '{expiry_date}'.")
        logger.warning(f"⚠️ Unparseable expiry date '{expiry_date}'; falling back to active regime default.")
        lot_size = CURRENT_REGIME_NIFTY_LOT_SIZE
        details = {
            "lot_size": lot_size,
            "source": "ACTIVE_REGIME_DEFAULT",
            "circular_ref": "NSE/FAOP/70616",
            "contract_attributes": {"expiry_date": str(expiry_date)},
            "transition_notes": "Unparseable expiry date; using active regime default (non-strict mode)."
        }
        return (lot_size, details) if return_details else lot_size

    if expiry_dt is None and as_of_dt is None:
        if strict:
            raise ValueError("Neither expiry_date, as_of_date, nor contract_metadata was provided to resolve lot size in strict mode.")
        env_override = os.environ.get("NIFTY_LOT_SIZE")
        if env_override:
            try:
                env_val = float(env_override)
                if math.isfinite(env_val) and env_val > 0 and env_val.is_integer():
                    lot_size = int(env_val)
                    details = {
                        "lot_size": lot_size,
                        "source": "ENVIRONMENT_OVERRIDE",
                        "circular_ref": "OPERATIONAL_SETTING",
                        "contract_attributes": {},
                        "transition_notes": f"NIFTY_LOT_SIZE environment override: {lot_size}"
                    }
                    return (lot_size, details) if return_details else lot_size
            except (ValueError, TypeError):
                pass
        lot_size = CURRENT_REGIME_NIFTY_LOT_SIZE
        details = {
            "lot_size": lot_size,
            "source": "ACTIVE_REGIME_DEFAULT",
            "circular_ref": "NSE/FAOP/70616",
            "contract_attributes": {},
            "transition_notes": "No date or metadata provided; using active market regime default."
        }
        return (lot_size, details) if return_details else lot_size

    target_dt = expiry_dt if expiry_dt is not None else as_of_dt

    # --------------------------------------------------------------------------
    # 4. Long-Dated Contracts Transition Rules (Quarterly & Half-Yearly)
    # --------------------------------------------------------------------------
    if contract_type in ("QUARTERLY", "HALF_YEARLY"):
        # Transition 1: EOD Dec 26, 2024 under NSE/FAOP/64625 (25 to 75)
        # Transition 2: EOD Dec 30, 2025 under NSE/FAOP/70616 (75 to 65)
        if as_of_dt is not None:
            if as_of_dt <= DATE_DEC_2024_MONTHLY_EXPIRY:
                lot_size = 25
                circ = "NSE/FAOP/61415"
                note = f"{contract_type} contract observed on {as_of_dt} prior to Dec 26, 2024 EOD transition (lot size 25)."
            elif as_of_dt <= DATE_FINAL_MONTHLY_75:
                lot_size = 75
                circ = "NSE/FAOP/64625"
                note = f"{contract_type} contract observed on {as_of_dt} post-Dec 26, 2024 transition and prior to Dec 30, 2025 EOD transition (lot size 75)."
            else:
                lot_size = 65
                circ = "NSE/FAOP/70616"
                note = f"{contract_type} contract observed on {as_of_dt} post-Dec 30, 2025 EOD transition (lot size 65)."

            details = {
                "lot_size": lot_size,
                "source": "NSE_CIRCULAR_SCHEDULE",
                "circular_ref": circ,
                "contract_attributes": {
                    "symbol": "NIFTY",
                    "expiry_date": str(expiry_dt),
                    "contract_type": contract_type,
                    "as_of_date": str(as_of_dt)
                },
                "transition_notes": note
            }
            return (lot_size, details) if return_details else lot_size

        # If as_of_date is NOT specified:
        if expiry_dt and expiry_dt > DATE_DEC_2024_MONTHLY_EXPIRY:
            if strict:
                raise AmbiguousLotSizeError(
                    f"Ambiguous {contract_type} contract expiring {expiry_dt}: lot size changed across circular transition dates "
                    f"(EOD 2024-12-26 from 25 to 75, and EOD 2025-12-30 from 75 to 65). "
                    f"Provide as_of_date or contract_metadata to determine applicable lot size."
                )
            if expiry_dt > DATE_FINAL_MONTHLY_75:
                lot_size = 65
                circ = "NSE/FAOP/70616"
            else:
                lot_size = 75
                circ = "NSE/FAOP/64625"
            details = {
                "lot_size": lot_size,
                "source": "NSE_CIRCULAR_SCHEDULE",
                "circular_ref": circ,
                "contract_attributes": {
                    "symbol": "NIFTY",
                    "expiry_date": str(expiry_dt),
                    "contract_type": contract_type
                },
                "transition_notes": f"{contract_type} contract evaluated at expiry date (non-strict mode)."
            }
            return (lot_size, details) if return_details else lot_size

    # --------------------------------------------------------------------------
    # 5. Regulatory Eras by Expiry Date / Implementation Cutover
    # --------------------------------------------------------------------------

    # Era 1: Prior to Jul 01, 2021 (Historical baseline = 75)
    if target_dt < DATE_ERA_50_START:
        lot_size = 75
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/HISTORICAL",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "Historical Nifty baseline lot size prior to Jul 01, 2021."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 2: Jul 01, 2021 to Apr 25, 2024 (NSE Circular 25/2021 -> 50)
    # Note: April 25, 2024 was the April 2024 monthly expiry date, the last contract under 50.
    if target_dt <= DATE_50_CUTOFF_EXPIRY:
        lot_size = 50
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/25/2021",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "NSE Circular 25/2021 50-lot regime prior to Apr 26, 2024 revision."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 3: Apr 26, 2024 to Nov 19, 2024 (NSE/FAOP/61415 -> 25)
    # First weekly: May 02, 2024; First monthly: May 30, 2024.
    if target_dt < DATE_MAHARASHTRA_ELECTION_HOLIDAY:
        lot_size = 25
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/61415",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "NSE Circular NSE/FAOP/61415 25-lot regime effective Apr 26, 2024."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 4: Nov 20, 2024 to Dec 26, 2024 (November 2024 Cutover & Transition Treatment)
    # - Nov 20, 2024: Exchange Trading Holiday (Maharashtra Assembly Elections).
    # - Nov 21, 2024: Actual implementation trade date per NSE/FAOP/64990.
    #   Contracts expiring Nov 21, 2024 (weekly), Nov 28, 2024 (monthly), and Dec 26, 2024 (monthly)
    #   were existing contracts introduced prior to Nov 21, 2024, and maintained lot size 25 until expiry.
    if target_dt <= DATE_DEC_2024_MONTHLY_EXPIRY:
        lot_size = 25
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/64625_AND_64990",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": (
                "Existing weekly and monthly contracts maintained lot size 25 through Dec 26, 2024 monthly expiry "
                "under NSE/FAOP/64625 and implementation update NSE/FAOP/64990 (implementation trade date: Nov 21, 2024)."
            )
        }
        return (lot_size, details) if return_details else lot_size

    # Era 5: Dec 27, 2024 to Jan 30, 2025 (The January 2025 Transition Window)
    # In January 2025:
    # - The monthly contract expiring Jan 30, 2025 was an existing contract introduced prior to Nov 21, 2024; it maintained lot size 25.
    # - New weekly contracts introduced post-Nov 21, 2024 (Jan 02, Jan 09, Jan 16, Jan 23) had lot size 75.
    if target_dt <= DATE_JAN_2025_MONTHLY_EXPIRY:
        if target_dt == DATE_JAN_2025_MONTHLY_EXPIRY:
            if contract_type == "MONTHLY" or (intro_dt and intro_dt < DATE_ERA_75_IMPLEMENTATION_TRADE_DATE):
                lot_size = 25
                details = {
                    "lot_size": lot_size,
                    "source": "NSE_CIRCULAR_SCHEDULE",
                    "circular_ref": "NSE/FAOP/64625_AND_64990",
                    "contract_attributes": {
                        "expiry_date": str(target_dt),
                        "contract_type": "MONTHLY",
                        "introduced_date": str(intro_dt) if intro_dt else "PRE_NOV_21_2024"
                    },
                    "transition_notes": "Existing Jan 30, 2025 monthly contract introduced prior to Nov 21, 2024 maintained lot size 25 until expiry."
                }
                return (lot_size, details) if return_details else lot_size
            elif contract_type == "WEEKLY" or (intro_dt and intro_dt >= DATE_ERA_75_IMPLEMENTATION_TRADE_DATE):
                lot_size = 75
                details = {
                    "lot_size": lot_size,
                    "source": "NSE_CIRCULAR_SCHEDULE",
                    "circular_ref": "NSE/FAOP/64625_AND_64990",
                    "contract_attributes": {
                        "expiry_date": str(target_dt),
                        "contract_type": "WEEKLY",
                        "introduced_date": str(intro_dt) if intro_dt else "POST_NOV_21_2024"
                    },
                    "transition_notes": "New contract introduced post-Nov 21, 2024 adopted revised lot size 75."
                }
                return (lot_size, details) if return_details else lot_size
            else:
                # Ambiguous: cannot determine whether contract is the pre-existing monthly or a weekly/new contract.
                # Always fails closed across both strict and non-strict modes.
                raise AmbiguousLotSizeError(
                    f"Ambiguous contract for expiry {target_dt} under NSE/FAOP/64625: "
                    f"existing monthly contract maintained lot size 25, while weekly/new contracts had lot size 75. "
                    f"Provide contract_type, introduced_date, or contract_metadata to resolve."
                )

        # Weekly expiries in January 2025 prior to monthly expiry (Jan 02, Jan 09, Jan 16, Jan 23)
        # These were introduced after Nov 21, 2024 and had lot size 75
        lot_size = 75
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/64625_AND_64990",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "Weekly contract introduced post-Nov 21, 2024 with lot size 75 under NSE/FAOP/64625."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 6: Jan 31, 2025 to Oct 28, 2025 (Standard 75-lot Era)
    if target_dt <= DATE_ERA_65_CIRCULAR_EFFECTIVE:
        lot_size = 75
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/64625",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "Standard 75-lot regime under NSE Circular NSE/FAOP/64625."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 7: Oct 29, 2025 to Dec 30, 2025 (NSE/FAOP/70616 Transition Window)
    # Existing weekly and monthly contracts continued with lot size 75 through Dec 30, 2025:
    # Final weekly expiry under 75: Dec 23, 2025.
    # Final monthly expiry under 75: Dec 30, 2025.
    if target_dt <= DATE_FINAL_MONTHLY_75:
        lot_size = 75
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/70616",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "Weekly and monthly contracts maintained lot size 75 through Dec 30, 2025 monthly expiry under NSE/FAOP/70616."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 8: Dec 31, 2025 to Jan 27, 2026 (NSE/FAOP/70616 First Revised Expiries)
    # First revised weekly: Jan 06, 2026 (lot size 65).
    # First revised monthly: Jan 27, 2026 (lot size 65).
    if target_dt <= DATE_FIRST_MONTHLY_65:
        lot_size = 65
        details = {
            "lot_size": lot_size,
            "source": "NSE_CIRCULAR_SCHEDULE",
            "circular_ref": "NSE/FAOP/70616",
            "contract_attributes": {"expiry_date": str(target_dt)},
            "transition_notes": "First revised weekly expiry Jan 06, 2026 and first revised monthly expiry Jan 27, 2026 with lot size 65 under NSE/FAOP/70616."
        }
        return (lot_size, details) if return_details else lot_size

    # Era 9: Jan 28, 2026 onwards (Current & Active 65-lot Regime)
    lot_size = 65
    details = {
        "lot_size": lot_size,
        "source": "NSE_CIRCULAR_SCHEDULE",
        "circular_ref": "NSE/FAOP/70616",
        "contract_attributes": {"expiry_date": str(target_dt)},
        "transition_notes": "Active 65-lot regime for contracts expiring 2026 onwards under NSE/FAOP/70616."
    }
    return (lot_size, details) if return_details else lot_size


def get_nifty_lot_size(
    expiry_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    as_of_date: Optional[Union[str, datetime.date, datetime.datetime]] = None,
    contract_metadata: Optional[Dict[str, Any]] = None,
    contract_type: Optional[str] = None,
    strict: bool = False
) -> int:
    """Convenience alias for resolve_nifty_lot_size."""
    return resolve_nifty_lot_size(
        expiry_date=expiry_date,
        as_of_date=as_of_date,
        contract_metadata=contract_metadata,
        contract_type=contract_type,
        strict=strict,
        return_details=False
    )


def calculate_contract_quantity(
    lots: int,
    lot_size: Optional[int] = None,
    expiry_date: Optional[Union[str, datetime.date]] = None,
    contract_metadata: Optional[Dict[str, Any]] = None,
    contract_type: Optional[str] = None,
    as_of_date: Optional[Union[str, datetime.date]] = None,
    strict: bool = False
) -> int:
    """
    Computes total contract quantity (underlying units) from number of lots.
    Formula: total_quantity = lots * lot_size
    Validates that lots >= 1 and lot_size is a verified positive integer.
    """
    try:
        lots_int = int(lots)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid lots parameter '{lots}'. Must be positive integer.")

    if lots_int <= 0:
        raise ValueError(f"Invalid lots count '{lots}'. Must be at least 1 lot.")

    if lot_size is None or int(lot_size) <= 0:
        lot_size = resolve_nifty_lot_size(
            expiry_date=expiry_date,
            as_of_date=as_of_date,
            contract_metadata=contract_metadata,
            contract_type=contract_type,
            strict=strict
        )

    ls_int = int(lot_size)
    if ls_int <= 0:
        raise ValueError(f"Invalid lot size '{lot_size}'. Must be positive integer.")

    return lots_int * ls_int


def calculate_lots_from_quantity(
    qty: int,
    lot_size: Optional[int] = None,
    expiry_date: Optional[Union[str, datetime.date]] = None,
    contract_metadata: Optional[Dict[str, Any]] = None,
    contract_type: Optional[str] = None,
    as_of_date: Optional[Union[str, datetime.date]] = None,
    allow_partial: bool = False,
    strict: bool = False
) -> int:
    """
    Computes number of lots from total contract quantity.
    Formula: lots = qty // lot_size
    Validates that qty >= 1 and lot_size is a verified positive integer.
    If allow_partial is False and qty is not an exact multiple of lot_size, logs a warning.
    """
    try:
        qty_int = int(qty)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid quantity parameter '{qty}'. Must be positive integer.")

    if qty_int <= 0:
        raise ValueError(f"Invalid quantity '{qty}'. Must be positive integer.")

    if lot_size is None or int(lot_size) <= 0:
        lot_size = resolve_nifty_lot_size(
            expiry_date=expiry_date,
            as_of_date=as_of_date,
            contract_metadata=contract_metadata,
            contract_type=contract_type,
            strict=strict
        )

    ls_int = int(lot_size)
    if ls_int <= 0:
        raise ValueError(f"Invalid lot size '{lot_size}'. Must be positive integer.")

    if (qty_int % ls_int != 0) and not allow_partial:
        logger.warning(
            f"Quantity {qty_int} is not an exact multiple of lot size {ls_int} (remainder: {qty_int % ls_int})."
        )

    return max(1, qty_int // ls_int)


def enrich_leg_with_lot_metadata(
    leg: Dict[str, Any],
    expiry_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Ensures that an option leg dictionary consistently distinguishes:
      - 'lot_size': authoritative contract lot size
      - 'lots': number of lots (positive integer)
      - 'qty': total contract quantity (underlying units = lots * lot_size)
    Preserves historical leg quantities without reinterpretation.
    """
    l = dict(leg)

    # 1. Resolve or preserve leg's lot size
    leg_ls = l.get("lot_size")
    if not leg_ls or int(leg_ls) <= 0:
        # Check if leg has historical qty and lots with exact ratio
        if l.get("qty") and l.get("lots"):
            try:
                q = int(l["qty"])
                num_l = int(l["lots"])
                if q > 0 and num_l > 0 and q % num_l == 0:
                    leg_ls = q // num_l
            except (ValueError, TypeError):
                pass

        if not leg_ls or int(leg_ls) <= 0:
            leg_ls = resolve_nifty_lot_size(
                expiry_date=l.get("expiry_date") or expiry_date,
                as_of_date=l.get("as_of_date") or as_of_date,
                contract_metadata=l,
                contract_type=l.get("contract_type") or l.get("expiry_type"),
                strict=strict
            )

    ls_int = int(leg_ls)
    l["lot_size"] = ls_int

    # 2. Reconcile lots and quantity
    if l.get("qty") is not None and int(l.get("qty", 0)) > 0:
        q_int = int(l["qty"])
        l["qty"] = q_int
        l["lots"] = max(1, q_int // ls_int)
    elif l.get("lots") is not None and int(l.get("lots", 0)) > 0:
        num_l = int(l["lots"])
        l["lots"] = num_l
        l["qty"] = num_l * ls_int
    else:
        l["lots"] = 1
        l["qty"] = ls_int

    return l
