"""
tests/test_f304_strategy_adjustment_pricing_integrity.py
========================================================
Regression and Data Integrity Verification for Phase 3 Finding F3-04:
Strategy Adjustment Pricing Integrity & Zero Theoretical Black-Scholes Fallbacks.

Verification requirements under root AGENTS.md:
1. Authentic market quotes required for all dynamic strategy adjustments.
2. Missing, zero, negative, NaN, or non-finite option quotes must FAIL CLOSED (defer adjustment).
3. Theoretical Black-Scholes (BSM) prices must NEVER be substituted as observed market quotes,
   traded entry prices, or basis for subsequent P&L tracking.
4. Partial quote availability (e.g., ATM available but wing missing, or short hedge available
   but long hedge missing) must defer the adjustment to prevent unhedged exposure.
5. Inactive / deferred adjustments must preserve existing legs unmodified and must not insert
   fabricated credit records into options_strategy_adjustments_log.
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from engines.strategy_adjustment_engine import strategy_adjustment_engine
from execution.state_engine import state_engine


# ==============================================================================
# 1. UNIT TESTS: _get_option_quote() MARKET DATA HARDENING
# ==============================================================================

class TestOptionQuoteRetrieval:
    """Verifies that _get_option_quote returns only genuine, positive, finite market quotes."""

    def test_get_option_quote_valid_kotak_tick(self):
        """Authentic LTP in Kotak ticks is returned correctly."""
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_near_atm_option_ticks") as mock_ticks:
            mock_ticks.return_value = {
                "22650_CE": {"strike": 22650, "type": "CE", "ltp": 125.40}
            }
            quote = strategy_adjustment_engine._get_option_quote(22650, "CE", 22687.0)
            assert quote == 125.40
            assert isinstance(quote, float)

    def test_get_option_quote_valid_chain_data(self):
        """Authentic LTP in option chain is returned if Kotak ticks missing."""
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_near_atm_option_ticks", return_value={}):
            with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain") as mock_chain:
                mock_chain.return_value = {
                    "weekly": {
                        "strikes": [
                            {"strike": 22750, "pe_ltp": 88.50, "ce_ltp": 42.10}
                        ]
                    }
                }
                quote_pe = strategy_adjustment_engine._get_option_quote(22750, "PE", 22687.0)
                assert quote_pe == 88.50

                quote_ce = strategy_adjustment_engine._get_option_quote(22750, "CE", 22687.0)
                assert quote_ce == 42.10

    def test_get_option_quote_missing_returns_none(self):
        """Missing strike in both Kotak ticks and chain returns None (not 0.0 or BSM price)."""
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_near_atm_option_ticks", return_value={}):
            with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", return_value={}):
                quote = strategy_adjustment_engine._get_option_quote(25000, "CE", 22687.0)
                assert quote is None

    @pytest.mark.parametrize("invalid_val", [0.0, -15.5, float("nan"), float("inf"), float("-inf"), None, "N/A"])
    def test_get_option_quote_rejects_non_positive_or_non_finite(self, invalid_val):
        """Quotes that are <= 0.0, NaN, Inf, or invalid strings must be rejected."""
        with patch("engines.kotak_neo_engine.kotak_neo_engine.get_near_atm_option_ticks") as mock_ticks:
            mock_ticks.return_value = {
                "22600_PE": {"strike": 22600, "type": "PE", "ltp": invalid_val}
            }
            with patch("engines.nifty_analytics_engine.nifty_analytics_engine.fetch_and_analyze_option_chain", return_value={}):
                quote = strategy_adjustment_engine._get_option_quote(22600, "PE", 22687.0)
                assert quote is None, f"Expected None for invalid value {invalid_val}, got {quote}"

    def test_get_option_quote_invalid_strike_or_type(self):
        """Non-numeric strike or invalid option type returns None safely."""
        assert strategy_adjustment_engine._get_option_quote(None, "CE", 22687.0) is None
        assert strategy_adjustment_engine._get_option_quote("INVALID", "CE", 22687.0) is None
        assert strategy_adjustment_engine._get_option_quote(22600, "XX", 22687.0) is None
        assert strategy_adjustment_engine._get_option_quote(22600, None, 22687.0) is None


# ==============================================================================
# 2. INTEGRATION TESTS: FAIL-CLOSED ADJUSTMENT ON MISSING MARKET DATA
# ==============================================================================

class TestIronFlyAdjustmentIntegrity:
    """Verifies Institutional Iron Fly adjustments never fabricate prices."""

    def _sample_iron_fly_legs(self):
        return [
            {"strike": 22700, "type": "CE", "action": "SELL", "qty": 75, "entry_price": 110.0, "premium": 110.0, "state": "ACTIVE"},
            {"strike": 22700, "type": "PE", "action": "SELL", "qty": 75, "entry_price": 95.0, "premium": 95.0, "state": "ACTIVE"},
            {"strike": 22900, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 30.0, "premium": 30.0, "state": "ACTIVE"},
            {"strike": 22500, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 25.0, "premium": 25.0, "state": "ACTIVE"},
        ]

    def test_iron_fly_roll_down_with_authentic_quotes_succeeds(self):
        """When authentic market quotes are available, roll down succeeds and logs LIVE_MARKET_QUOTE."""
        legs = self._sample_iron_fly_legs()
        spot = 22600.0  # Drift: -100 pts (triggers roll down)

        def mock_get_quote(strike, opt_type, spot_px):
            if strike == 22650 and opt_type == "CE":
                return 135.50
            if strike == 22850 and opt_type == "CE":
                return 42.20
            return None

        with patch.object(strategy_adjustment_engine, "_get_option_quote", side_effect=mock_get_quote):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_IF_SUCCESS", "Institutional Iron Fly", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is True
                assert "Rolled untested Call wing down to 22650 CE" in note
                atm_ce = next(l for l in new_legs if l["action"] == "SELL" and l["type"] == "CE")
                wing_ce = next(l for l in new_legs if l["action"] == "BUY" and l["type"] == "CE")

                assert atm_ce["strike"] == 22650
                assert atm_ce["entry_price"] == 135.50
                assert atm_ce["quote_source"] == "LIVE_MARKET_QUOTE"
                assert wing_ce["strike"] == 22850
                assert wing_ce["entry_price"] == 42.20
                assert wing_ce["quote_source"] == "LIVE_MARKET_QUOTE"
                mock_record.assert_called_once()

    def test_iron_fly_roll_down_missing_atm_quote_fails_closed(self):
        """When ATM quote is missing, adjustment is deferred and legs remain completely unmodified."""
        legs = self._sample_iron_fly_legs()
        spot = 22600.0

        # ATM quote missing (None), Wing quote available
        def mock_get_quote(strike, opt_type, spot_px):
            if strike == 22850 and opt_type == "CE":
                return 42.20
            return None

        with patch.object(strategy_adjustment_engine, "_get_option_quote", side_effect=mock_get_quote):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_IF_DEFER", "Institutional Iron Fly", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is False
                assert "Missing authentic market quote" in note
                assert "22650 CE (ATM roll)" in note
                atm_ce = next(l for l in new_legs if l["action"] == "SELL" and l["type"] == "CE")
                assert atm_ce["strike"] == 22700, "ATM strike must remain un-rolled"
                assert atm_ce["state"] == "ACTIVE", "ATM state must not be changed"
                assert atm_ce["entry_price"] == 110.0, "Entry price must not be corrupted"
                mock_record.assert_not_called()

    def test_iron_fly_roll_down_missing_wing_quote_fails_closed(self):
        """When wing quote is missing, partial unhedged roll must be prevented."""
        legs = self._sample_iron_fly_legs()
        spot = 22600.0

        # ATM quote available, Wing quote missing
        def mock_get_quote(strike, opt_type, spot_px):
            if strike == 22650 and opt_type == "CE":
                return 135.50
            return None

        with patch.object(strategy_adjustment_engine, "_get_option_quote", side_effect=mock_get_quote):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_IF_WING_MISSING", "Institutional Iron Fly", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is False
                assert "22850 CE (wing roll)" in note
                atm_ce = next(l for l in new_legs if l["action"] == "SELL" and l["type"] == "CE")
                wing_ce = next(l for l in new_legs if l["action"] == "BUY" and l["type"] == "CE")
                assert atm_ce["strike"] == 22700
                assert wing_ce["strike"] == 22900
                mock_record.assert_not_called()

    def test_iron_fly_roll_up_missing_quote_fails_closed(self):
        """Spot rally drift up with missing PE quotes defers adjustment."""
        legs = self._sample_iron_fly_legs()
        spot = 22800.0  # Drift: +100 pts (triggers roll up)

        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=None):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_IF_UP_DEFER", "Institutional Iron Fly", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is False
                assert "Missing authentic market quote" in note
                atm_pe = next(l for l in new_legs if l["action"] == "SELL" and l["type"] == "PE")
                assert atm_pe["strike"] == 22700
                mock_record.assert_not_called()


class TestBrokenWingButterflyAdjustmentIntegrity:
    """Verifies Broken Wing Butterfly adjustments fail closed when quotes unavailable."""

    def test_bull_bwb_missing_quote_defers_adjustment(self):
        """Bull BWB downside threat defers tightening if new wing quote is unavailable."""
        legs = [
            {"strike": 22700, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 80.0, "state": "ACTIVE"},
            {"strike": 22600, "type": "PE", "action": "SELL", "qty": 150, "entry_price": 40.0, "state": "ACTIVE"},
            {"strike": 22450, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 15.0, "state": "ACTIVE"},
        ]
        spot = 22610.0  # Within 25 pts of K2 (22600) -> threatens broken wing

        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=None):
            adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                "TEST_BWB_DEFER", "Bull Broken Wing Butterfly", legs, spot_price=spot, horizon="INTRADAY"
            )

            assert adjusted is False
            assert "Missing authentic market quote for 22500 PE" in note
            k3_leg = next(l for l in new_legs if l["action"] == "BUY" and l["strike"] == 22450)
            assert k3_leg["strike"] == 22450
            assert k3_leg["state"] == "ACTIVE"

    def test_bull_bwb_valid_quote_tightens_wing(self):
        """Bull BWB with valid quote tightens wing and logs LIVE_MARKET_QUOTE."""
        legs = [
            {"strike": 22700, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 80.0, "state": "ACTIVE"},
            {"strike": 22600, "type": "PE", "action": "SELL", "qty": 150, "entry_price": 40.0, "state": "ACTIVE"},
            {"strike": 22450, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 15.0, "state": "ACTIVE"},
        ]
        spot = 22610.0

        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=28.50):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_BWB_OK", "Bull Broken Wing Butterfly", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is True
                assert "Tightened broken put wing from 22450 to 22500 PE" in note
                k3_leg = next(l for l in new_legs if l["action"] == "BUY" and l["strike"] == 22500)
                assert k3_leg["entry_price"] == 28.50
                assert k3_leg["quote_source"] == "LIVE_MARKET_QUOTE"
                assert k3_leg["state"] == "ADJUSTED_TIGHTENED"
                mock_record.assert_called_once()

    def test_bear_bwb_missing_quote_defers_adjustment(self):
        """Bear BWB upside breakout threat defers tightening if new wing quote is unavailable."""
        legs = [
            {"strike": 22700, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 85.0, "state": "ACTIVE"},
            {"strike": 22800, "type": "CE", "action": "SELL", "qty": 150, "entry_price": 45.0, "state": "ACTIVE"},
            {"strike": 22950, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 18.0, "state": "ACTIVE"},
        ]
        spot = 22790.0  # Within 25 pts of K2 (22800)

        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=None):
            adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                "TEST_BEAR_BWB_DEFER", "Bear Broken Wing Butterfly", legs, spot_price=spot, horizon="INTRADAY"
            )

            assert adjusted is False
            assert "Missing authentic market quote for 22900 CE" in note
            k3_leg = next(l for l in new_legs if l["action"] == "BUY" and l["strike"] == 22950)
            assert k3_leg["strike"] == 22950
            assert k3_leg["state"] == "ACTIVE"


class TestRatioBackspreadAdjustmentIntegrity:
    """Verifies Ratio Backspread valley defenses fail closed when hedge quotes are unavailable."""

    def test_call_ratio_backspread_missing_either_hedge_quote_defers(self):
        """Call Ratio Backspread requiring Put Credit Spread hedge must have BOTH quotes."""
        legs = [
            {"strike": 22700, "type": "CE", "action": "SELL", "qty": 75, "entry_price": 120.0, "state": "ACTIVE"},
            {"strike": 22850, "type": "CE", "action": "BUY", "qty": 150, "entry_price": 50.0, "state": "ACTIVE"},
        ]
        spot = 22775.0  # In Valley of Death (22700 - 22850)

        # Short hedge available (22600 PE = 65.0), Long hedge missing (22550 PE = None)
        def mock_get_quote(strike, opt_type, spot_px):
            if strike == 22600 and opt_type == "PE":
                return 65.0
            return None

        with patch.object(strategy_adjustment_engine, "_get_option_quote", side_effect=mock_get_quote):
            adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                "TEST_CRB_DEFER", "Call Ratio Backspread", legs, spot_price=spot, horizon="INTRADAY"
            )

            assert adjusted is False
            assert "22550 PE (long hedge)" in note
            assert len(new_legs) == 2, "Hedge legs must NOT be added"

    def test_put_ratio_backspread_valid_quotes_deploys_hedge(self):
        """Put Ratio Backspread with valid quotes deploys Call Credit Spread hedge."""
        legs = [
            {"strike": 22700, "type": "PE", "action": "SELL", "qty": 75, "entry_price": 115.0, "state": "ACTIVE"},
            {"strike": 22550, "type": "PE", "action": "BUY", "qty": 150, "entry_price": 45.0, "state": "ACTIVE"},
        ]
        spot = 22625.0  # In Valley of Death (22550 - 22700)

        def mock_get_quote(strike, opt_type, spot_px):
            if strike == 22800 and opt_type == "CE":
                return 48.0
            if strike == 22850 and opt_type == "CE":
                return 32.0
            return None

        with patch.object(strategy_adjustment_engine, "_get_option_quote", side_effect=mock_get_quote):
            with patch.object(strategy_adjustment_engine, "_record_adjustment") as mock_record:
                adjusted, new_legs, note = strategy_adjustment_engine._check_and_apply_adjustments(
                    "TEST_PRB_OK", "Put Ratio Backspread", legs, spot_price=spot, horizon="INTRADAY"
                )

                assert adjusted is True
                assert len(new_legs) == 4, "Must add 2 hedge legs"
                short_hedge = next(l for l in new_legs if l["strike"] == 22800 and l["type"] == "CE")
                long_hedge = next(l for l in new_legs if l["strike"] == 22850 and l["type"] == "CE")

                assert short_hedge["entry_price"] == 48.0
                assert short_hedge["quote_source"] == "LIVE_MARKET_QUOTE"
                assert long_hedge["entry_price"] == 32.0
                assert long_hedge["quote_source"] == "LIVE_MARKET_QUOTE"
                mock_record.assert_called_once()


# ==============================================================================
# 3. VERIFICATION THAT BSM THEORETICAL PRICES ARE NEVER RECORDED AS ENTRY PRICES
# ==============================================================================

class TestTheoreticalPriceRejection:
    """Verifies theoretical BSM prices are strictly prohibited from entry prices."""

    def test_zero_quote_never_substitutes_bsm_price(self):
        """Simulate quote returning 0.0 or None; verify BSM price is NOT substituted."""
        legs = [
            {"strike": 22700, "type": "CE", "action": "SELL", "qty": 75, "entry_price": 100.0, "state": "ACTIVE"},
            {"strike": 22700, "type": "PE", "action": "SELL", "qty": 75, "entry_price": 100.0, "state": "ACTIVE"},
            {"strike": 22900, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 30.0, "state": "ACTIVE"},
            {"strike": 22500, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 25.0, "state": "ACTIVE"},
        ]
        spot = 22600.0

        # When quote lookup returns None
        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=None):
            adjusted, res_legs, _ = strategy_adjustment_engine._check_and_apply_adjustments(
                "TEST_NO_BSM", "Institutional Iron Fly", legs, spot_price=spot, horizon="INTRADAY"
            )
            assert adjusted is False
            for leg in res_legs:
                assert leg.get("quote_source") != "THEORETICAL_BSM"
                assert leg.get("quote_source") != "BSM_FALLBACK"

    def test_corrupted_quote_does_not_mutate_sqlite_database(self):
        """When an adjustment is deferred, no records are written to options_strategy_adjustments_log."""
        strat_id = "TEST_INTEGRITY_STRAT_1"
        legs = [
            {"strike": 22700, "type": "CE", "action": "SELL", "qty": 75, "entry_price": 100.0, "state": "ACTIVE"},
            {"strike": 22700, "type": "PE", "action": "SELL", "qty": 75, "entry_price": 100.0, "state": "ACTIVE"},
            {"strike": 22900, "type": "CE", "action": "BUY", "qty": 75, "entry_price": 30.0, "state": "ACTIVE"},
            {"strike": 22500, "type": "PE", "action": "BUY", "qty": 75, "entry_price": 25.0, "state": "ACTIVE"},
        ]

        with state_engine._lock:
            with state_engine._get_connection() as conn:
                count_before = conn.execute(
                    "SELECT COUNT(*) as c FROM options_strategy_adjustments_log WHERE strategy_id = ?",
                    (strat_id,)
                ).fetchone()["c"]

        # Attempt adjustment with missing quotes
        with patch.object(strategy_adjustment_engine, "_get_option_quote", return_value=None):
            adjusted, _, _ = strategy_adjustment_engine._check_and_apply_adjustments(
                strat_id, "Institutional Iron Fly", legs, spot_price=22600.0, horizon="INTRADAY"
            )
            assert adjusted is False

        with state_engine._lock:
            with state_engine._get_connection() as conn:
                count_after = conn.execute(
                    "SELECT COUNT(*) as c FROM options_strategy_adjustments_log WHERE strategy_id = ?",
                    (strat_id,)
                ).fetchone()["c"]

        assert count_after == count_before, "Database adjustments log must NOT have new records for deferred adjustments"
