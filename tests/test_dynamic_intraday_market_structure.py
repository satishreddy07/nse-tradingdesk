"""
Unit Tests for Pure First-Principles Quantitative Intraday Market Structure & Dynamic Regime Classification.
Validates:
1. Morning directional selloff -> BEARISH_CALL_WRITING_PRESSURE
2. Midday consolidation with lower-third base pinning -> CONSOLIDATION_WITH_BEARISH_BIAS
3. Afternoon rapid velocity surge on down day -> SHORT_COVERING_SQUEEZE (decoupled from session open)
4. Dynamic volatility scaling across different India VIX regimes (no hardcoded point numbers)
5. Robustness against missing prior slots and edge cases
"""

import pytest
import math
from engines.nifty_analytics_engine import nifty_analytics_engine


def test_volatility_scaling_formula():
    """Verify that 15m and 1-day standard deviations follow Black-Scholes square-root time scaling."""
    spot = 22000.0
    atm_iv = 13.5
    
    # 15m sigma: spot * (iv/100) * sqrt(15 / 94500)
    expected_sigma_15m = round(spot * (atm_iv / 100.0) * math.sqrt(15.0 / (375.0 * 252.0)), 2)
    # 1-day sigma: spot * (iv/100) / sqrt(252)
    expected_sigma_day = round(spot * (atm_iv / 100.0) / math.sqrt(252.0), 2)
    
    res = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=spot,
        atm_iv=atm_iv,
        prior_snaps=[],
        near_atm_pcr=1.0,
        chg_oi_pcr=1.0,
        net_ce_chg=0,
        net_pe_chg=0,
        straddle_prem=150.0
    )
    meta = res["snap_meta"]
    assert meta["sigma_15m"] == expected_sigma_15m
    assert meta["sigma_day"] == expected_sigma_day
    assert meta["sigma_15m"] > 30.0 and meta["sigma_15m"] < 45.0  # Approx ~37.4 pts


def test_scenario_a_morning_selloff():
    """
    Scenario A: 09:30 - 10:15 IST
    Spot breaks down sharply with massive Call writing additions.
    Should classify as BEARISH_CALL_WRITING_PRESSURE.
    """
    prior_snaps = [
        {
            "session_date": "2026-10-23",
            "slot_time": "09:30",
            "spot_price": 22100.0,
            "straddle_premium": 160.0,
            "near_atm_pcr": 1.05,
            "net_ce_oi_chg": 5000,
            "net_pe_oi_chg": 4000
        },
        {
            "session_date": "2026-10-23",
            "slot_time": "09:45",
            "spot_price": 22050.0,
            "straddle_premium": 155.0,
            "near_atm_pcr": 0.92,
            "net_ce_oi_chg": 25000,
            "net_pe_oi_chg": 8000
        }
    ]
    
    # At 10:00, spot drops to 22000 (-50 pts in 15m, Z_15m ~ -1.33sigma)
    res = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=22000.0,
        atm_iv=13.5,
        prior_snaps=prior_snaps,
        near_atm_pcr=0.78,
        chg_oi_pcr=0.65,
        net_ce_chg=65000,
        net_pe_chg=15000,
        straddle_prem=150.0,
        vwap=22060.0,
        call_wall=22100,
        put_wall=21900,
        max_pain=22000,
        cur_atm=22000
    )
    
    assert res["market_state"] == "BEARISH_CALL_WRITING_PRESSURE"
    assert res["structure_verdict"] == "BEARISH_TREND_EXPANSION"
    assert res["snap_meta"]["z_15m"] <= -1.0
    assert "DO NOT SHORT against" not in res["quant_suggestion"]
    assert "Sell-on-Rallies" in res["quant_suggestion"] or "Bear Call Credit Spread" in res["quant_suggestion"]


def test_scenario_b_midday_consolidation_with_bearish_bias():
    """
    Scenario B: 10:30 - 13:00 IST
    Spot stops falling, consolidates in a tight 40-pt range, but hovers near the base low (Phi <= 0.35)
    with straddle decay accelerating.
    Crucial fix: System must NOT say 'BEARISH TREND' here; it must say 'CONSOLIDATION_WITH_BEARISH_BIAS'.
    """
    prior_snaps = [
        {"slot_time": "09:30", "spot_price": 22100.0, "straddle_premium": 160.0, "near_atm_pcr": 1.05, "net_ce_oi_chg": 5000, "net_pe_oi_chg": 4000},
        {"slot_time": "10:00", "spot_price": 22000.0, "straddle_premium": 150.0, "near_atm_pcr": 0.78, "net_ce_oi_chg": 65000, "net_pe_oi_chg": 15000},
        {"slot_time": "10:30", "spot_price": 21980.0, "straddle_premium": 142.0, "near_atm_pcr": 0.76, "net_ce_oi_chg": 85000, "net_pe_oi_chg": 20000},
        {"slot_time": "11:00", "spot_price": 21990.0, "straddle_premium": 135.0, "near_atm_pcr": 0.77, "net_ce_oi_chg": 90000, "net_pe_oi_chg": 22000},
        {"slot_time": "11:30", "spot_price": 21975.0, "straddle_premium": 128.0, "near_atm_pcr": 0.75, "net_ce_oi_chg": 92000, "net_pe_oi_chg": 23000},
        {"slot_time": "12:00", "spot_price": 21985.0, "straddle_premium": 122.0, "near_atm_pcr": 0.76, "net_ce_oi_chg": 93000, "net_pe_oi_chg": 24000},
    ]
    
    # At 12:30, spot is at 21978 (range is 21975 - 22000 = 25 pts, compression ratio ~ 0.13, base skew ~ 0.12)
    res = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=21978.0,
        atm_iv=13.5,
        prior_snaps=prior_snaps,
        near_atm_pcr=0.76,
        chg_oi_pcr=0.82,
        net_ce_chg=95000,
        net_pe_chg=25000,
        straddle_prem=118.0,
        vwap=22010.0,
        call_wall=22100,
        put_wall=21900,
        max_pain=22000,
        cur_atm=22000
    )
    
    assert res["market_state"] == "CONSOLIDATION_WITH_BEARISH_BIAS"
    assert res["structure_verdict"] == "RANGE_BOUND_BEARISH_DRIFT"
    assert res["snap_meta"]["comp_ratio"] <= 0.45
    assert res["snap_meta"]["base_skew"] <= 0.35
    assert "CONSOLIDATION WITH BEARISH BIAS" in res["quant_suggestion"]
    assert "Avoid aggressive put buying near support" in res["quant_suggestion"]


def test_scenario_c_afternoon_short_covering_squeeze():
    """
    Scenario C: 13:30 - 14:15 IST
    Macro day is still down -100 pts from open, but in the last 15-30 minutes,
    Spot violently surges +80 pts and Call writers liquidate (-15,000 contracts).
    Old system failed by calling this 'BEARISH' because spot_from_open was <= -35.
    Dynamic system must decouple and classify as SHORT_COVERING_SQUEEZE!
    """
    prior_snaps = [
        {"slot_time": "09:30", "spot_price": 22100.0, "straddle_premium": 160.0, "near_atm_pcr": 1.05, "net_ce_oi_chg": 5000, "net_pe_oi_chg": 4000},
        {"slot_time": "12:30", "spot_price": 21960.0, "straddle_premium": 118.0, "near_atm_pcr": 0.75, "net_ce_oi_chg": 95000, "net_pe_oi_chg": 25000},
        {"slot_time": "13:00", "spot_price": 21970.0, "straddle_premium": 112.0, "near_atm_pcr": 0.77, "net_ce_oi_chg": 96000, "net_pe_oi_chg": 26000},
        {"slot_time": "13:30", "spot_price": 21980.0, "straddle_premium": 105.0, "near_atm_pcr": 0.80, "net_ce_oi_chg": 94000, "net_pe_oi_chg": 28000},
    ]
    
    # At 13:45, spot surges to 22045 (+65 pts in 15m, Z_15m ~ +1.73sigma) and net CE falls from 94k to 75k (-19k unwinding)
    res = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=22045.0,
        atm_iv=13.5,
        prior_snaps=prior_snaps,
        near_atm_pcr=0.88,
        chg_oi_pcr=1.35,
        net_ce_chg=75000,  # slot_ce_oi_chg = 75000 - 94000 = -19,000 (huge call short covering!)
        net_pe_chg=38000,
        straddle_prem=102.0,
        vwap=22005.0,
        call_wall=22100,
        put_wall=21900,
        max_pain=22000,
        cur_atm=22050
    )
    
    assert res["market_state"] == "SHORT_COVERING_SQUEEZE"
    assert res["structure_verdict"] == "RAPID_GAMMA_SQUEEZE_CALL_UNWINDING"
    assert res["snap_meta"]["z_15m"] >= 1.0
    assert res["snap_meta"]["slot_ce_oi_chg"] < 0
    assert "SHORT COVERING SQUEEZE" in res["quant_suggestion"]
    assert "DO NOT ENTER NAKED SHORTS" in res["quant_suggestion"]


def test_volatility_normalization_high_vs_low_vix():
    """
    Test that a 40-pt move in low VIX (11%) is a strong >1.0 sigma trend move,
    whereas in high VIX (28%), a 40-pt move is routine noise (<0.5 sigma).
    Strictly zero hardcoding!
    """
    prior = [{"slot_time": "09:30", "spot_price": 22000.0, "straddle_premium": 140.0, "near_atm_pcr": 1.0, "net_ce_oi_chg": 0, "net_pe_oi_chg": 0}]
    
    # Low VIX (11%)
    res_low_vix = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=22045.0,
        atm_iv=11.0,
        prior_snaps=prior,
        near_atm_pcr=1.1,
        chg_oi_pcr=1.2,
        net_ce_chg=1000,
        net_pe_chg=15000,
        straddle_prem=130.0
    )
    sigma_15m_low = res_low_vix["snap_meta"]["sigma_15m"]
    z_low = res_low_vix["snap_meta"]["z_15m"]
    
    # High VIX (28%)
    res_high_vix = nifty_analytics_engine.classify_market_structure_dynamic(
        spot=22045.0,
        atm_iv=28.0,
        prior_snaps=prior,
        near_atm_pcr=1.1,
        chg_oi_pcr=1.2,
        net_ce_chg=1000,
        net_pe_chg=15000,
        straddle_prem=260.0
    )
    sigma_15m_high = res_high_vix["snap_meta"]["sigma_15m"]
    z_high = res_high_vix["snap_meta"]["z_15m"]
    
    assert sigma_15m_high > sigma_15m_low * 2.0
    assert z_low > z_high * 2.0
    assert z_low >= 1.0  # Significant expansion in low VIX
    assert z_high < 0.6  # Routine noise in high VIX
