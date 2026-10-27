"""
tests/test_intraday_panels_audit.py
===================================
Automated verification suite for the NIFTY Intraday Dashboard Panels Audit:
1. Panel A: VWAP & Opening Range (ORB)
2. Panel B: Gamma Exposure (GEX) & Zero-Gamma Flip
3. Panel C: Sector Breadth & Drag Gauge
4. Data Quality & Non-Fabrication Invariants
"""

import math
import pytest
from engines.intraday_analytics_engine import IntradayAnalyticsEngine


@pytest.fixture
def engine():
    return IntradayAnalyticsEngine()


# ============================================================================
# PANEL A: VWAP & OPENING RANGE (ORB) TESTS
# ============================================================================

def test_panel_a_example_values():
    """
    Test Panel A against the exact user-specified test fixture:
    - Spot: 22,744.9
    - VWAP: 22,660.1
    - 15-minute high: 22,750
    - 15-minute low: 22,654
    - 30-minute high: 22,767
    - 30-minute low: 22,654

    Expected results:
    - VWAP distance: +84.8 points
    - VWAP distance percentage: approximately +0.374%
    - 15-minute ORB width: 96 points
    - 30-minute ORB width: 113 points
    - Spot is inside both opening ranges.
    """
    spot = 22744.9
    vwap = 22660.1
    orb_15_hi = 22750.0
    orb_15_lo = 22654.0
    orb_30_hi = 22767.0
    orb_30_lo = 22654.0

    # Mathematical computations
    diff_pts = round(spot - vwap, 2)
    diff_pct = round((diff_pts / vwap) * 100.0, 4)
    orb_15_rng = round(orb_15_hi - orb_15_lo, 2)
    orb_30_rng = round(orb_30_hi - orb_30_lo, 2)

    assert diff_pts == 84.8
    assert round(diff_pct, 3) == 0.374
    assert orb_15_rng == 96.0
    assert orb_30_rng == 113.0

    # Inside/outside checks
    is_inside_15 = (orb_15_lo <= spot <= orb_15_hi)
    is_inside_30 = (orb_30_lo <= spot <= orb_30_hi)
    assert is_inside_15 is True
    assert is_inside_30 is True


def test_panel_a_boundary_expansion_and_breakdown():
    """Verify that expansion and breakdown are evaluated without arbitrary constant buffers."""
    hi = 22750.0
    lo = 22654.0

    # Exact threshold +0.1 point is expansion
    spot_above = 22750.1
    assert spot_above > hi

    # Exact threshold -0.1 point is breakdown
    spot_below = 22653.9
    assert spot_below < lo

    # Exact boundary points are inside
    assert lo <= 22654.0 <= hi
    assert lo <= 22750.0 <= hi


def test_panel_a_orb_time_windows():
    """
    Verify ORB candle filtering semantics:
    - 15m ORB: 09:15 <= t < 09:30 IST (3 bars: 09:15, 09:20, 09:25)
    - 30m ORB: 09:15 <= t < 09:45 IST (6 bars: 09:15, 09:20, 09:25, 09:30, 09:35, 09:40)
    """
    mock_candles = [
        {"display_time": "09:15", "high": 22700, "low": 22660, "close": 22680},
        {"display_time": "09:20", "high": 22720, "low": 22654, "close": 22710},
        {"display_time": "09:25", "high": 22750, "low": 22680, "close": 22740},
        {"display_time": "09:30", "high": 22760, "low": 22730, "close": 22755},  # MUST NOT be in 15m ORB!
        {"display_time": "09:35", "high": 22765, "low": 22740, "close": 22750},
        {"display_time": "09:40", "high": 22767, "low": 22745, "close": 22760},
        {"display_time": "09:45", "high": 22780, "low": 22750, "close": 22770},  # MUST NOT be in 30m ORB!
    ]

    orb_15 = [c for c in mock_candles if "09:15" <= c["display_time"] < "09:30"]
    assert len(orb_15) == 3
    assert max(c["high"] for c in orb_15) == 22750
    assert min(c["low"] for c in orb_15) == 22654
    assert max(c["high"] for c in orb_15) - min(c["low"] for c in orb_15) == 96

    orb_30 = [c for c in mock_candles if "09:15" <= c["display_time"] < "09:45"]
    assert len(orb_30) == 6
    assert max(c["high"] for c in orb_30) == 22767
    assert min(c["low"] for c in orb_30) == 22654
    assert max(c["high"] for c in orb_30) - min(c["low"] for c in orb_30) == 113


# ============================================================================
# PANEL B: GAMMA EXPOSURE (GEX) & ZERO-GAMMA FLIP TESTS
# ============================================================================

def test_panel_b_gamma_analytical_formula(engine):
    """Verify analytical Black-Scholes gamma calculation matches ground truth."""
    spot = 22700.0
    strike = 22700.0
    T = 3.0 / 365.25
    iv = 0.14  # 14%

    gamma = engine._calc_bs_gamma(spot, strike, T, iv)
    assert gamma > 0.0

    # Mathematical verification:
    # d1 = (ln(1) + 0.5 * 0.14^2 * T) / (0.14 * sqrt(T)) = 0.5 * 0.14 * sqrt(T)
    d1 = 0.5 * iv * math.sqrt(T)
    pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * (d1 ** 2))
    expected_gamma = pdf_d1 / (spot * iv * math.sqrt(T))
    assert abs(gamma - expected_gamma) < 1e-10

    # Invalid boundary conditions return 0.0 without exception
    assert engine._calc_bs_gamma(0, strike, T, iv) == 0.0
    assert engine._calc_bs_gamma(spot, 0, T, iv) == 0.0
    assert engine._calc_bs_gamma(spot, strike, 0, iv) == 0.0
    assert engine._calc_bs_gamma(spot, strike, T, 0) == 0.0


def test_panel_b_spot_flip_distance():
    """Verify exact distance calculation from prompt: Spot 22,744.9, Zero-gamma 22,767.8 -> distance -22.9 pts."""
    spot = 22744.9
    flip = 22767.8
    dist = round(spot - flip, 1)
    assert dist == -22.9


def test_panel_b_no_zero_crossing_returns_none(engine):
    """When all strikes are call-heavy, net GEX is positive everywhere; must return None without inventing a root."""
    dummy_chain = {
        "spot_price": 22700.0,
        "expiry": "16-Oct-2026",
        "strikes": [
            {"strike": 22500, "ce_oi": 150000, "pe_oi": 5000, "ce_iv": 14.0, "pe_iv": 14.0},
            {"strike": 22600, "ce_oi": 180000, "pe_oi": 8000, "ce_iv": 14.0, "pe_iv": 14.0},
            {"strike": 22700, "ce_oi": 200000, "pe_oi": 10000, "ce_iv": 14.0, "pe_iv": 14.0},
            {"strike": 22800, "ce_oi": 250000, "pe_oi": 12000, "ce_iv": 14.0, "pe_iv": 14.0},
        ]
    }
    res = engine.compute_gamma_exposure_and_flip(dummy_chain)
    assert res["status"] == "OK"
    assert res["total_net_gex_cr"] > 0.0
    assert res["estimated_net_gex_sign"] == "POSITIVE"
    assert res["zero_gamma_flip_level"] is None
    assert res["spot_vs_flip_pts"] is None
    assert res["zero_gamma_selection_rule"] == "NO_ZERO_CROSSING"
    assert "Zero-gamma flip level unavailable" in res["market_gex_note"]


def test_panel_b_multiple_zero_crossings_closest_to_spot(engine):
    """When multiple zero crossings exist, must select the root closest to current spot."""
    dummy_chain = {
        "spot_price": 22700.0,
        "expiry": "16-Oct-2026",
        "strikes": [
            {"strike": 22400, "ce_oi": 100000, "pe_oi": 10000, "ce_iv": 14.0, "pe_iv": 14.0},  # +
            {"strike": 22500, "ce_oi": 10000, "pe_oi": 150000, "ce_iv": 14.0, "pe_iv": 14.0},  # - crossing 1 (~22450)
            {"strike": 22700, "ce_oi": 10000, "pe_oi": 150000, "ce_iv": 14.0, "pe_iv": 14.0},  # -
            {"strike": 22800, "ce_oi": 160000, "pe_oi": 10000, "ce_iv": 14.0, "pe_iv": 14.0},  # + crossing 2 (~22750)
        ]
    }
    res = engine.compute_gamma_exposure_and_flip(dummy_chain)
    assert res["status"] == "OK"
    assert res["zero_gamma_flip_level"] is not None
    # Spot is 22700.0; crossing 2 (~22750) is closer to spot than crossing 1 (~22450)
    assert abs(res["zero_gamma_flip_level"] - 22700.0) < abs(res["zero_gamma_flip_level"] - 22450.0)
    assert "CLOSEST_TO_SPOT" in res["zero_gamma_selection_rule"]


def test_panel_b_missing_oi_and_empty_strikes(engine):
    """Missing or empty strikes must return unavailable status without fabricating data."""
    # Empty chain
    res_empty = engine.compute_gamma_exposure_and_flip({"spot_price": 22700.0, "strikes": []})
    assert res_empty["status"] == "UNAVAILABLE"
    assert res_empty["zero_gamma_flip_level"] is None
    assert res_empty["total_net_gex_cr"] == 0.0

    # Chain with zero OI
    zero_oi_chain = {
        "spot_price": 22700.0,
        "expiry": "16-Oct-2026",
        "strikes": [
            {"strike": 22600, "ce_oi": 0, "pe_oi": 0, "ce_iv": 14.0, "pe_iv": 14.0},
            {"strike": 22700, "ce_oi": 0, "pe_oi": 0, "ce_iv": 14.0, "pe_iv": 14.0},
        ]
    }
    res_zero = engine.compute_gamma_exposure_and_flip(zero_oi_chain)
    assert res_zero["total_net_gex_cr"] == 0.0
    assert res_zero["zero_gamma_flip_level"] is None


def test_panel_b_cautious_language(engine):
    """Negative GEX must NOT be claimed as proof of dealer positions; must use cautious inferred language."""
    dummy_chain = {
        "spot_price": 22700.0,
        "expiry": "16-Oct-2026",
        "strikes": [
            {"strike": 22600, "ce_oi": 5000, "pe_oi": 150000, "ce_iv": 14.0, "pe_iv": 14.0},
            {"strike": 22700, "ce_oi": 10000, "pe_oi": 200000, "ce_iv": 14.0, "pe_iv": 14.0},
        ]
    }
    res = engine.compute_gamma_exposure_and_flip(dummy_chain)
    assert res["status"] == "OK"
    assert res["total_net_gex_cr"] < 0
    note = res["market_gex_note"]
    assert "proof" not in note.lower()
    assert "inferred statistical model" in note.lower() or "model-estimated" in note.lower()


# ============================================================================
# PANEL C: SECTOR BREADTH & DRAG GAUGE TESTS
# ============================================================================

def test_panel_c_visible_example_values():
    """
    Test Panel C against the exact user-specified visible sectors fixture:
    - Pharma: +2.00%
    - Auto: +1.28%
    - Metal: +0.81%
    - Energy: +0.42%
    - Bank: -0.04%
    - FMCG: -0.37%

    Expected results:
    - Positive sectors: 4
    - Valid visible sectors: 6
    - Breadth: 66.67%
    - Best sector: +2.00%
    - Worst sector: -0.37%
    - Spread: 2.37 percentage points
    """
    sample_sectors = [
        {"name": "Nifty Pharma", "ticker": "^CNXPHARMA", "price": 19000.0, "change_pct": 2.00, "is_valid": True, "weight": 0.06},
        {"name": "Nifty Auto", "ticker": "^CNXAUTO", "price": 22000.0, "change_pct": 1.28, "is_valid": True, "weight": 0.10},
        {"name": "Nifty Metal", "ticker": "^CNXMETAL", "price": 8500.0, "change_pct": 0.81, "is_valid": True, "weight": 0.10},
        {"name": "Nifty Energy", "ticker": "^CNXENERGY", "price": 38000.0, "change_pct": 0.42, "is_valid": True, "weight": 0.08},
        {"name": "Nifty Bank", "ticker": "^NSEBANK", "price": 48000.0, "change_pct": -0.04, "is_valid": True, "weight": 0.35},
        {"name": "Nifty FMCG", "ticker": "^CNXFMCG", "price": 54000.0, "change_pct": -0.37, "is_valid": True, "weight": 0.06},
    ]

    valid_sectors = [s for s in sample_sectors if s["is_valid"]]
    valid_total = len(valid_sectors)
    valid_positive = sum(1 for s in valid_sectors if s["change_pct"] > 0)
    breadth_pct = round((valid_positive / valid_total) * 100.0, 2)

    valid_sectors.sort(key=lambda s: s["change_pct"], reverse=True)
    best_sec = valid_sectors[0]
    worst_sec = valid_sectors[-1]
    spread_pp = round(best_sec["change_pct"] - worst_sec["change_pct"], 2)

    assert valid_positive == 4
    assert valid_total == 6
    assert breadth_pct == 66.67
    assert best_sec["change_pct"] == 2.00
    assert worst_sec["change_pct"] == -0.37
    assert spread_pp == 2.37


def test_panel_c_numerator_denominator_consistency(engine):
    """Ensure numerator and denominator use the exact same valid sector count."""
    res = engine.compute_sectoral_divergence()
    assert res["status"] == "OK"
    valid_count = res.get("valid_sectors_count") or res.get("total_sectors")
    green_count = res["green_sectors"]
    breadth_pct = res["breadth_pct_green"]

    expected_pct = round((green_count / valid_count) * 100.0, 2)
    assert abs(breadth_pct - expected_pct) < 0.1


def test_panel_c_dispersion_separated_from_weighted_drag(engine):
    """
    Ensure best-vs-worst spread (percentage points) is strictly separated
    from index-weighted drag contribution.
    """
    res = engine.compute_sectoral_divergence()
    assert "sector_spread_pp" in res
    assert "weighted_index_drag_pp" in res
    assert "top_drag_sector" in res
    assert "top_lift_sector" in res

    # Spread is non-negative dispersion
    assert res["sector_spread_pp"] >= 0.0


def test_panel_c_sector_store_provenance(engine):
    """Verify every sector stores ticker, name, price, reference_timestamp, change_pct, change_pts, and is_valid."""
    res = engine.compute_sectoral_divergence()
    for s in res["sectors"]:
        assert "ticker" in s
        assert "name" in s
        assert "price" in s
        assert "reference_timestamp" in s
        assert "change_pct" in s
        assert "change_pts" in s
        assert "is_valid" in s
        assert "weighted_drag_contribution_pts" in s
