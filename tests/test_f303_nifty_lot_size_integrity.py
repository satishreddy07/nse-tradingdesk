"""
tests/test_f303_nifty_lot_size_integrity.py
===========================================
Regression and Data Integrity Verification for Phase 3 Finding F3-03:
NIFTY Lot-Size Configuration & Authoritative Exchange Resolution.

Verification requirements under root AGENTS.md and official NSE circulars:
1. NSE Circular NSE/FAOP/61415 (Dated Apr 02, 2024, Ref 37/2024):
   - 50-to-25 revision effective for trading Apr 26, 2024.
   - Contracts expiring on or before Apr 25, 2024 retained lot size 50.
   - First revised weekly expiry: May 02, 2024 (lot size 25).
   - First revised monthly expiry: May 30, 2024 (lot size 25).
2. NSE Circular NSE/FAOP/64625 (Dated Oct 18, 2024, Ref 128/2024) &
   Implementation Update NSE/FAOP/64990 (Dated Nov 08, 2024):
   - Nov 20, 2024: Trading holiday (Maharashtra Assembly Elections).
   - Nov 21, 2024: Actual implementation trade date on which new contracts launched with lot size 75.
   - Weekly contract expiring on Nov 21, 2024 retained lot size 25.
   - Existing weekly & monthly contracts continued with lot size 25 until expiry
     (including Nov 21 weekly, Nov 28 monthly, Dec 26 monthly, and the Jan 30, 2025 monthly expiry).
   - New weekly contracts introduced post-Nov 21, 2024 adopted lot size 75.
   - Existing quarterly & half-yearly contracts transitioned at EOD Dec 26, 2024 (25 -> 75).
3. NSE Circular NSE/FAOP/70616 (Dated Oct 03, 2025, Ref 176/2025):
   - 75-to-65 revision effective Oct 28, 2025 (EOD).
   - Weekly & monthly contracts retained lot size 75 through Dec 30, 2025 monthly expiry
     (final weekly: Dec 23, 2025; final monthly: Dec 30, 2025).
   - First revised weekly expiry: Jan 06, 2026 (lot size 65).
   - First revised monthly expiry: Jan 27, 2026 (lot size 65).
   - Existing quarterly & half-yearly contracts transitioned at EOD Dec 30, 2025 (75 -> 65).
4. Ambiguous contracts where expiry alone is insufficient fail closed with AmbiguousLotSizeError.
5. Trading-sensitive callers fail closed and stop execution if lot size cannot be authoritatively resolved.
6. Authoritative contract metadata overrides generic schedule and is strictly validated (rejecting non-NIFTY and invalid lots).
"""

import math
import datetime
import pytest
from unittest.mock import patch

from core.lot_size import (
    resolve_nifty_lot_size,
    get_nifty_lot_size,
    parse_contract_expiry_date,
    calculate_contract_quantity,
    calculate_lots_from_quantity,
    enrich_leg_with_lot_metadata,
    AmbiguousLotSizeError,
    NIFTY_LOT_SIZE_SCHEDULE,
    CURRENT_REGIME_NIFTY_LOT_SIZE
)
from core.config import NIFTY_LOT_SIZE
from engines.advanced_options_pricing_engine import advanced_options_pricing_engine
from engines.nifty_analytics_engine import nifty_analytics_engine, state_engine
from engines.strategy_adjustment_engine import strategy_adjustment_engine


# ==============================================================================
# 1. 50-TO-25 TRANSITION (NSE CIRCULAR NSE/FAOP/61415)
# ==============================================================================

class Test50To25TransitionNSE61415:
    """Verifies the 50-to-25 revision effective April 26, 2024 under NSE Circular NSE/FAOP/61415."""

    def test_apr_2024_monthly_expiry_preserves_50(self):
        """April 25, 2024 was the monthly expiry for April 2024 and maintained lot size 50."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-04-25", return_details=True)
        assert lot == 50
        assert details["source"] == "NSE_CIRCULAR_SCHEDULE"
        assert details["circular_ref"] == "NSE/25/2021"

    def test_effective_date_trading_starts_25(self):
        """Contracts available for trading from April 26, 2024 onwards adopt lot size 25."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-04-26", return_details=True)
        assert lot == 25
        assert details["circular_ref"] == "NSE/FAOP/61415"

    def test_first_revised_weekly_expiry_may_02_2024(self):
        """First weekly expiry with revised lot size 25 was May 02, 2024."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-05-02", return_details=True)
        assert lot == 25
        assert details["circular_ref"] == "NSE/FAOP/61415"

    def test_first_revised_monthly_expiry_may_30_2024(self):
        """First monthly expiry with revised lot size 25 was May 30, 2024."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-05-30", return_details=True)
        assert lot == 25
        assert details["circular_ref"] == "NSE/FAOP/61415"

    @pytest.mark.parametrize("expiry", ["2024-06-27", "2024-07-25", "2024-08-29", "2024-09-26", "2024-10-31", "2024-11-14"])
    def test_subsequent_2024_contracts_have_lot_size_25(self, expiry):
        """Contracts between May 2024 and Nov 19, 2024 had lot size 25."""
        assert resolve_nifty_lot_size(expiry_date=expiry) == 25


# ==============================================================================
# 2. NOVEMBER 2024 CUTOVER (NSE/FAOP/64625 & NSE/FAOP/64990)
# ==============================================================================

class TestNovember2024CutoverNSE64625And64990:
    """Verifies the November 2024 cutover taking into account the Maharashtra election holiday."""

    def test_trade_date_nov_19_2024_before_cutover(self):
        """Contracts observed or expiring on Nov 19, 2024 (prior to holiday and cutover) had lot size 25."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-11-19", as_of_date="2024-11-19", return_details=True)
        assert lot == 25
        assert details["circular_ref"] == "NSE/FAOP/61415"

    def test_holiday_date_nov_20_2024_election_holiday(self):
        """On Nov 20, 2024 (trading holiday), unexpired active contracts were under lot size 25."""
        lot = resolve_nifty_lot_size(expiry_date="2024-11-28", as_of_date="2024-11-20")
        assert lot == 25

    def test_implementation_trade_date_nov_21_2024_weekly_expiry_retains_25(self):
        """
        On the actual implementation trade date (Nov 21, 2024), the weekly contract expiring
        on that day was an existing contract introduced prior to Nov 21 and expired with lot size 25.
        """
        lot, details = resolve_nifty_lot_size(expiry_date="2024-11-21", return_details=True)
        assert lot == 25
        assert "64625" in details["circular_ref"]

    def test_existing_nov_2024_monthly_retains_25(self):
        """Existing monthly contract expiring Nov 28, 2024 maintained lot size 25."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-11-28", return_details=True)
        assert lot == 25
        assert "64625" in details["circular_ref"]

    def test_existing_dec_2024_monthly_retains_25(self):
        """Existing monthly contract expiring Dec 26, 2024 maintained lot size 25."""
        lot, details = resolve_nifty_lot_size(expiry_date="2024-12-26", return_details=True)
        assert lot == 25
        assert "64625" in details["circular_ref"]

    def test_existing_jan_30_2025_monthly_retains_25_when_specified(self):
        """Existing Jan 30, 2025 monthly contract introduced pre-Nov 21, 2024 maintained lot size 25."""
        lot, details = resolve_nifty_lot_size(
            expiry_date="2025-01-30",
            contract_type="MONTHLY",
            return_details=True
        )
        assert lot == 25
        assert "64625" in details["circular_ref"]

        # Also via introduced_date
        lot_intro = resolve_nifty_lot_size(
            expiry_date="2025-01-30",
            introduced_date="2024-10-15"
        )
        assert lot_intro == 25

    def test_new_weekly_contracts_jan_2025_adopt_75(self):
        """New weekly contracts introduced post-Nov 21, 2024 for Jan 2025 expiries have lot size 75."""
        for exp in ("2025-01-02", "2025-01-09", "2025-01-16", "2025-01-23"):
            lot, details = resolve_nifty_lot_size(expiry_date=exp, return_details=True)
            assert lot == 75
            assert "64625" in details["circular_ref"]

    def test_jan_30_2025_weekly_or_new_contract_adopts_75(self):
        """If a contract for Jan 30, 2025 was explicitly introduced post-Nov 21, 2024 or is weekly, lot size is 75."""
        lot = resolve_nifty_lot_size(expiry_date="2025-01-30", contract_type="WEEKLY")
        assert lot == 75

        lot_intro = resolve_nifty_lot_size(expiry_date="2025-01-30", introduced_date="2024-11-25")
        assert lot_intro == 75

    def test_contracts_introduced_on_trade_date_nov_21_have_lot_size_75(self):
        """New contracts made available for trading on Nov 21, 2024 have lot size 75."""
        lot = resolve_nifty_lot_size(
            expiry_date="2025-02-27",
            introduced_date="2024-11-21"
        )
        assert lot == 75

    def test_feb_2025_and_subsequent_contracts_have_lot_size_75(self):
        """New monthly and weekly contracts from Feb 2025 to Oct 2025 have lot size 75."""
        for exp in ("2025-02-27", "2025-03-27", "2025-06-26", "2025-09-25", "2025-10-28"):
            assert resolve_nifty_lot_size(expiry_date=exp) == 75


# ==============================================================================
# 3. 75-TO-65 TRANSITION (NSE CIRCULAR NSE/FAOP/70616)
# ==============================================================================

class Test75To65TransitionNSE70616:
    """Verifies the 75-to-65 revision and transition rules under NSE Circular NSE/FAOP/70616."""

    def test_final_weekly_under_75_is_dec_23_2025(self):
        """Final weekly expiry contract under lot size 75 was Dec 23, 2025."""
        lot, details = resolve_nifty_lot_size(expiry_date="2025-12-23", return_details=True)
        assert lot == 75
        assert details["circular_ref"] == "NSE/FAOP/70616"

    def test_final_monthly_under_75_is_dec_30_2025(self):
        """Final monthly expiry contract under lot size 75 was Dec 30, 2025."""
        lot, details = resolve_nifty_lot_size(expiry_date="2025-12-30", return_details=True)
        assert lot == 75
        assert details["circular_ref"] == "NSE/FAOP/70616"

    def test_first_revised_weekly_expiry_jan_06_2026(self):
        """First weekly expiry with revised lot size 65 was Jan 06, 2026."""
        lot, details = resolve_nifty_lot_size(expiry_date="2026-01-06", return_details=True)
        assert lot == 65
        assert details["circular_ref"] == "NSE/FAOP/70616"

    def test_first_revised_monthly_expiry_jan_27_2026(self):
        """First monthly expiry with revised lot size 65 was Jan 27, 2026."""
        lot, details = resolve_nifty_lot_size(expiry_date="2026-01-27", return_details=True)
        assert lot == 65
        assert details["circular_ref"] == "NSE/FAOP/70616"

    def test_current_2026_regime_is_65(self):
        """All contracts expiring in 2026 after Jan 27, 2026 have authoritative lot size 65."""
        for exp in ("2026-02-26", "2026-06-25", "2026-10-27", "2026-12-31"):
            assert resolve_nifty_lot_size(expiry_date=exp) == 65

        assert CURRENT_REGIME_NIFTY_LOT_SIZE == 65
        assert NIFTY_LOT_SIZE == 65


# ==============================================================================
# 4. EXISTING CONTRACTS TRANSITION TIMES (QUARTERLY & HALF-YEARLY)
# ==============================================================================

class TestExistingContractsTransitionTimes:
    """Verifies existing quarterly and half-yearly contracts that change lot size at EOD transition dates."""

    def test_quarterly_contract_transition_at_eod_dec_26_2024(self):
        """
        Quarterly contract expiring 2025-06-26 had lot size 25 prior to Dec 26, 2024 EOD,
        and transitioned to 75 after Dec 26, 2024 EOD under NSE/FAOP/64625.
        """
        # Traded on 2024-12-20 (prior to transition): lot size = 25
        lot_pre, det_pre = resolve_nifty_lot_size(
            expiry_date="2025-06-26",
            contract_type="QUARTERLY",
            as_of_date="2024-12-20",
            return_details=True
        )
        assert lot_pre == 25
        assert det_pre["circular_ref"] == "NSE/FAOP/61415"

        # Traded on 2024-12-27 (post transition): lot size = 75
        lot_post, det_post = resolve_nifty_lot_size(
            expiry_date="2025-06-26",
            contract_type="QUARTERLY",
            as_of_date="2024-12-27",
            return_details=True
        )
        assert lot_post == 75
        assert "64625" in det_post["circular_ref"]

    def test_quarterly_contract_transition_at_eod_dec_30_2025(self):
        """
        Quarterly contract expiring 2026-03-26 had lot size 75 prior to Dec 30, 2025 EOD,
        and transitioned to 65 after Dec 30, 2025 EOD under NSE/FAOP/70616.
        """
        # Traded on 2025-12-15 (prior to transition): lot size = 75
        lot_pre, det_pre = resolve_nifty_lot_size(
            expiry_date="2026-03-26",
            contract_type="QUARTERLY",
            as_of_date="2025-12-15",
            return_details=True
        )
        assert lot_pre == 75
        assert "64625" in det_pre["circular_ref"]

        # Traded on 2026-01-05 (post transition): lot size = 65
        lot_post, det_post = resolve_nifty_lot_size(
            expiry_date="2026-03-26",
            contract_type="QUARTERLY",
            as_of_date="2026-01-05",
            return_details=True
        )
        assert lot_post == 65
        assert det_post["circular_ref"] == "NSE/FAOP/70616"


# ==============================================================================
# 5. AMBIGUOUS CONTRACTS FAIL CLOSED (ACROSS ALL MODES)
# ==============================================================================

class TestAmbiguousContractsFailClosed:
    """Verifies that contracts with insufficient or ambiguous attributes fail closed."""

    @pytest.mark.parametrize("strict_flag", [True, False, None])
    def test_jan_30_2025_without_cycle_fails_closed_across_all_strict_modes(self, strict_flag):
        """
        Expiry 2025-01-30 without contract_type or metadata cannot determine
        if it is the 25-lot monthly or 75-lot contract. Must raise AmbiguousLotSizeError
        regardless of whether strict is True, False, or omitted!
        """
        kwargs = {"expiry_date": "2025-01-30"}
        if strict_flag is not None:
            kwargs["strict"] = strict_flag

        with pytest.raises(AmbiguousLotSizeError, match="Ambiguous contract for expiry 2025-01-30"):
            resolve_nifty_lot_size(**kwargs)

    def test_quarterly_contract_across_transition_without_as_of_date_fails_closed_in_strict(self):
        """
        A quarterly contract expiring post-2024-12-26 without as_of_date in strict mode
        must raise AmbiguousLotSizeError rather than guessing whether it's evaluated pre- or post-transition.
        """
        with pytest.raises(AmbiguousLotSizeError, match="lot size changed across circular transition dates"):
            resolve_nifty_lot_size(
                expiry_date="2025-06-26",
                contract_type="QUARTERLY",
                strict=True
            )

    def test_unparseable_date_strict_fails_closed(self):
        """In strict mode, an unparseable expiry date raises ValueError."""
        with pytest.raises(ValueError, match="unparseable expiry date"):
            resolve_nifty_lot_size(expiry_date="NOT-A-REAL-DATE", strict=True)

    def test_strict_mode_without_any_parameters_fails_closed(self):
        """In strict mode, calling without dates or metadata fails closed."""
        with pytest.raises(ValueError, match="Neither expiry_date, as_of_date, nor contract_metadata"):
            resolve_nifty_lot_size(strict=True)


# ==============================================================================
# 6. METADATA PRECEDENCE & STRICT VALIDATION
# ==============================================================================

class TestMetadataPrecedenceAndStrictValidation:
    """Verifies that contract metadata overrides schedules and invalid/mismatched metadata is rejected."""

    def test_explicit_contract_metadata_overrides_generic_schedule(self):
        """Validated broker contract metadata overrides generic circular schedule."""
        meta = {"symbol": "NIFTY", "lot_size": 25, "circular_ref": "BROKER_OVERRIDE"}
        lot, details = resolve_nifty_lot_size(
            expiry_date="2025-01-02",
            contract_metadata=meta,
            return_details=True
        )
        assert lot == 25
        assert details["source"] == "CONTRACT_METADATA"

    def test_kotak_scrip_master_llotsize_column_support(self):
        """Supports Kotak Neo FO scrip master 'lLotSize' attribute."""
        meta = {"pSymbol": "51332", "pTrdSymbol": "NIFTY26OCT22550CE", "lLotSize": 65}
        assert resolve_nifty_lot_size(contract_metadata=meta) == 65

    @pytest.mark.parametrize("invalid_lot", [0, -65, -1, float("nan"), float("inf"), float("-inf"), "INVALID"])
    def test_rejection_of_invalid_lot_sizes_in_contract_metadata(self, invalid_lot):
        """Contract metadata with non-positive or non-finite lot_size must be rejected without schedule fallback."""
        meta = {"symbol": "NIFTY", "lot_size": invalid_lot}
        with pytest.raises(ValueError, match="Invalid"):
            resolve_nifty_lot_size(contract_metadata=meta)

    @pytest.mark.parametrize("non_nifty_sym", [
        "BANKNIFTY26OCT48000CE",
        "FINNIFTY26OCT24000PE",
        "MIDCPNIFTY26OCT12000CE",
        "NIFTYNXT5026OCT65000PE",
        "RELIANCE26OCT3000CE",
        "TCS26OCT4500CE",
        "INFY26OCT1900PE"
    ])
    def test_rejection_of_non_nifty_metadata_symbols(self, non_nifty_sym):
        """Contract metadata for non-NIFTY underlying instruments must be rejected immediately."""
        meta = {"symbol": non_nifty_sym, "lot_size": 25}
        with pytest.raises(ValueError, match="Mismatched contract metadata symbol"):
            resolve_nifty_lot_size(contract_metadata=meta)

    def test_mismatched_expiry_between_metadata_and_argument_rejected(self):
        """Conflicting expiry dates between metadata and explicit argument raise ValueError."""
        meta = {"symbol": "NIFTY", "expiry_date": "2024-12-26", "lot_size": 25}
        with pytest.raises(ValueError, match="Mismatched expiry dates"):
            resolve_nifty_lot_size(expiry_date="2026-10-27", contract_metadata=meta)

    def test_non_nifty_symbol_argument_rejected(self):
        """resolve_nifty_lot_size explicitly rejects symbols that are not NIFTY."""
        with pytest.raises(ValueError, match="only supports NIFTY"):
            resolve_nifty_lot_size(symbol="BANKNIFTY")
        with pytest.raises(ValueError, match="only supports NIFTY"):
            resolve_nifty_lot_size(symbol="FINNIFTY")
        with pytest.raises(ValueError, match="only supports NIFTY"):
            resolve_nifty_lot_size(symbol="RELIANCE")


# ==============================================================================
# 7. TRADING-SENSITIVE CALLERS FAIL-CLOSED VERIFICATION
# ==============================================================================

class TestTradingSensitiveCallersFailClosed:
    """Verifies that every trading-sensitive engine caller stops and reports when lot size cannot be resolved."""

    def test_deploy_options_strategy_rejects_ambiguous_expiry(self):
        """deploy_options_strategy fails closed if expiry is ambiguous and cycle is not specified."""
        payload = {
            "strategy_name": "Ambiguous Expiry Test",
            "symbol": "NIFTY",
            "expiry_date": "2025-01-30",  # Ambiguous without contract_type
            "spot_at_entry": 24000.0,
            "net_premium": 1000.0,
            "legs": [{"strike": 24000, "type": "CE", "action": "SELL", "lots": 1, "entry_price": 100.0}]
        }
        with pytest.raises(ValueError, match="Cannot deploy options strategy: unable to resolve authoritative lot size"):
            nifty_analytics_engine.deploy_options_strategy(payload)

    def test_deploy_options_strategy_rejects_invalid_metadata_lot_size(self):
        """deploy_options_strategy fails closed if leg contains invalid non-positive lot_size."""
        payload = {
            "strategy_name": "Corrupted Metadata Test",
            "symbol": "NIFTY",
            "expiry_date": "27-Oct-2026",
            "expiry_type": "WEEKLY",
            "spot_at_entry": 22687.0,
            "net_premium": 1000.0,
            "legs": [{"strike": 22700, "type": "CE", "action": "SELL", "lot_size": -65, "lots": 1, "entry_price": 100.0}]
        }
        with pytest.raises(ValueError, match="Cannot deploy options strategy: unable to resolve authoritative lot size"):
            nifty_analytics_engine.deploy_options_strategy(payload)

    def test_adjust_options_strategy_stops_on_unresolvable_leg(self):
        """adjust_options_strategy halts and returns error when a leg has unresolvable lot size."""
        # Deploy a valid strategy first
        deploy_payload = {
            "strategy_name": "Adjustment Safety Test",
            "symbol": "NIFTY",
            "expiry_date": "27-Oct-2026",
            "expiry_type": "WEEKLY",
            "spot_at_entry": 22687.0,
            "net_premium": 2000.0,
            "legs": [{"strike": 22700, "type": "CE", "action": "SELL", "lots": 1, "entry_price": 100.0}]
        }
        res = nifty_analytics_engine.deploy_options_strategy(deploy_payload)
        strat_id = res["strategy_id"]

        try:
            # Attempt adjustment with ambiguous unresolvable leg
            bad_adjust_payload = {
                "strategy_id": strat_id,
                "legs": [
                    {
                        "strike": 22800,
                        "type": "CE",
                        "action": "SELL",
                        "expiry_date": "2025-01-30", # Ambiguous without cycle
                        "entry_price": 50.0
                    }
                ]
            }
            adjust_res = nifty_analytics_engine.adjust_options_strategy(bad_adjust_payload)
            assert adjust_res["status"] == "ERROR"
            assert "Cannot adjust strategy: unable to resolve authoritative lot size" in adjust_res["message"]
        finally:
            nifty_analytics_engine.exit_options_strategy(strat_id, exit_reason="TEST_CLEANUP")

    def test_calculate_current_pnl_stops_on_unresolvable_leg(self):
        """strategy_adjustment_engine._calculate_current_pnl raises ValueError when leg lot size is unresolvable."""
        legs = [
            {"strike": 24000, "type": "CE", "action": "SELL", "expiry_date": "2025-01-30", "entry_price": 100.0}
        ]
        with pytest.raises(ValueError, match="Cannot calculate current P&L: unable to resolve authoritative lot size"):
            strategy_adjustment_engine._calculate_current_pnl(legs, spot_price=24000.0, horizon="INTRADAY")

    def test_formulate_strategy_stops_on_unresolvable_expiry(self):
        """advanced_options_pricing_engine.formulate_strategy raises ValueError on unparseable expiry in strict resolution."""
        with pytest.raises(ValueError, match="Cannot formulate options strategy: unable to resolve authoritative lot size"):
            advanced_options_pricing_engine.formulate_strategy(
                "INSTITUTIONAL_IRON_FLY",
                spot_price=22600.0,
                iv_pct=13.5,
                dte=5,
                horizon="WEEKLY",
                expiry_date="CORRUPTED-DATE-999"
            )


# ==============================================================================
# 8. LOT COUNT VS. CONTRACT QUANTITY DISTINCTION
# ==============================================================================

class TestQuantityAndLotCalculations:
    """Verifies rigorous distinction between number of lots and total contract units."""

    def test_calculate_contract_quantity(self):
        """Quantity is strictly lots * lot_size."""
        assert calculate_contract_quantity(lots=1, lot_size=65) == 65
        assert calculate_contract_quantity(lots=2, lot_size=65) == 130
        assert calculate_contract_quantity(lots=5, lot_size=65) == 325
        assert calculate_contract_quantity(lots=2, lot_size=75) == 150
        assert calculate_contract_quantity(lots=4, lot_size=25) == 100
        assert calculate_contract_quantity(lots=2, lot_size=50) == 100

    def test_calculate_lots_from_quantity(self):
        """Lots count is strictly qty // lot_size."""
        assert calculate_lots_from_quantity(qty=65, lot_size=65) == 1
        assert calculate_lots_from_quantity(qty=130, lot_size=65) == 2
        assert calculate_lots_from_quantity(qty=150, lot_size=75) == 2
        assert calculate_lots_from_quantity(qty=100, lot_size=25) == 4

    @pytest.mark.parametrize("invalid_qty", [0, -50, "N/A"])
    def test_invalid_quantity_rejected(self, invalid_qty):
        """Non-positive or invalid quantity raises ValueError."""
        with pytest.raises(ValueError):
            calculate_lots_from_quantity(qty=invalid_qty, lot_size=65)

    @pytest.mark.parametrize("invalid_lots", [0, -2, "INVALID"])
    def test_invalid_lots_rejected(self, invalid_lots):
        """Non-positive or invalid lots count raises ValueError."""
        with pytest.raises(ValueError):
            calculate_contract_quantity(lots=invalid_lots, lot_size=65)

    def test_enrich_leg_with_lot_metadata(self):
        """Enriches leg dictionary with both lots and lot_size without data corruption."""
        leg1 = {"strike": 22600, "type": "CE", "action": "BUY", "qty": 130}
        enriched1 = enrich_leg_with_lot_metadata(leg1, expiry_date="27-Oct-2026")
        assert enriched1["lot_size"] == 65
        assert enriched1["lots"] == 2
        assert enriched1["qty"] == 130

        leg2 = {"strike": 22600, "type": "PE", "action": "SELL", "lots": 3}
        enriched2 = enrich_leg_with_lot_metadata(leg2, expiry_date="27-Oct-2026")
        assert enriched2["lot_size"] == 65
        assert enriched2["lots"] == 3
        assert enriched2["qty"] == 195


# ==============================================================================
# 9. HISTORICAL CONTRACT PRESERVATION & DOWNSTREAM INTEGRATION
# ==============================================================================

class TestHistoricalContractPreservationAndDownstream:
    """Verifies historical trades preserve their original contract lot size forever and downstream engines integrate."""

    def test_historical_2024_contract_preserves_original_lot_size(self):
        """A historical leg from Dec 2024 with lot_size=25 retains 25."""
        hist_leg = {
            "strike": 24200,
            "type": "CE",
            "action": "SELL",
            "qty": 25,
            "lots": 1,
            "lot_size": 25,
            "expiry_date": "26-Dec-2024",
            "entry_price": 120.0
        }
        enriched = enrich_leg_with_lot_metadata(hist_leg)
        assert enriched["lot_size"] == 25
        assert enriched["lots"] == 1
        assert enriched["qty"] == 25

    def test_historical_2023_contract_preserves_50_lot_size(self):
        """A trade from 2023 with qty=100 must preserve lot_size=50 and lots=2."""
        hist_leg = {
            "strike": 19500,
            "type": "PE",
            "action": "BUY",
            "qty": 100,
            "lots": 2,
            "expiry_date": "28-Dec-2023",
            "entry_price": 85.0
        }
        enriched = enrich_leg_with_lot_metadata(hist_leg)
        assert enriched["lot_size"] == 50
        assert enriched["lots"] == 2
        assert enriched["qty"] == 100

    def test_formulate_strategy_populates_authoritative_lot_metadata(self):
        """formulate_strategy attaches lot_size, lots, and qty to all legs."""
        spot = 22687.65
        strat = advanced_options_pricing_engine.formulate_strategy(
            "INSTITUTIONAL_IRON_FLY",
            spot_price=spot,
            iv_pct=13.5,
            dte=2,
            horizon="WEEKLY",
            expiry_date="27-Oct-2026"
        )
        assert strat["lot_size"] == 65
        assert len(strat["legs"]) == 4
        for leg in strat["legs"]:
            assert leg["lot_size"] == 65
            assert leg["lots"] >= 1
            assert leg["qty"] == leg["lots"] * 65

    def test_formulate_strategy_historical_expiry_resolves_historical_lot_size(self):
        """formulate_strategy with 2024 expiry resolves historical lot_size=25."""
        spot = 24500.0
        strat = advanced_options_pricing_engine.formulate_strategy(
            "BULL_BROKEN_WING_BUTTERFLY",
            spot_price=spot,
            iv_pct=14.0,
            dte=5,
            horizon="WEEKLY",
            expiry_date="26-Dec-2024"
        )
        assert strat["lot_size"] == 25
        for leg in strat["legs"]:
            assert leg["lot_size"] == 25
            assert leg["qty"] == leg["lots"] * 25

    def test_strategy_adjustment_engine_preserves_leg_lot_size_in_pnl(self):
        """_calculate_current_pnl uses leg's authoritative lot size."""
        legs = [
            {"strike": 22700, "type": "CE", "action": "SELL", "lots": 1, "lot_size": 65, "qty": 65, "entry_price": 100.0},
            {"strike": 22700, "type": "PE", "action": "SELL", "lots": 1, "lot_size": 65, "qty": 65, "entry_price": 100.0}
        ]
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_near_atm_option_ticks", return_value={
            "22700_CE": {"ltp": 90.0},
            "22700_PE": {"ltp": 90.0}
        }):
            total_pnl, updated = strategy_adjustment_engine._calculate_current_pnl(legs, spot_price=22700.0, horizon="INTRADAY")
            assert total_pnl == 1300.0
            assert updated[0]["pnl"] == 650.0
            assert updated[1]["pnl"] == 650.0

    def test_deploy_options_strategy_persists_lot_metadata(self):
        """deploy_options_strategy validates and writes lot_size and lots to SQLite."""
        deploy_payload = {
            "strategy_name": "Test Lot Size Deployment",
            "symbol": "NIFTY",
            "expiry_date": "27-Oct-2026",
            "expiry_type": "WEEKLY",
            "spot_at_entry": 22687.0,
            "net_premium": 5000.0,
            "max_profit": 5000.0,
            "max_loss": 10000.0,
            "breakevens": [22500, 22800],
            "risk_reward": "1:2",
            "est_margin": 65000.0,
            "pop_pct": 65.0,
            "target_pnl": 2500.0,
            "stoploss_pnl": -3000.0,
            "auto_trade_mode": "MANUAL",
            "legs": [
                {"strike": 22700, "type": "CE", "action": "SELL", "lots": 1, "entry_price": 100.0},
                {"strike": 22700, "type": "PE", "action": "SELL", "lots": 1, "entry_price": 90.0}
            ]
        }
        res = nifty_analytics_engine.deploy_options_strategy(deploy_payload)
        assert res["status"] == "OK"
        strat_id = res["strategy_id"]

        with state_engine._lock:
            with state_engine._get_connection() as conn:
                row = conn.execute("SELECT legs_json FROM options_strategies WHERE strategy_id = ?", (strat_id,)).fetchone()
                import json
                saved_legs = json.loads(row["legs_json"])
                assert len(saved_legs) == 2
                assert saved_legs[0]["lot_size"] == 65
                assert saved_legs[0]["lots"] == 1
                assert saved_legs[0]["qty"] == 65

        # Cleanup
        nifty_analytics_engine.exit_options_strategy(strat_id, exit_reason="TEST_CLEANUP")
