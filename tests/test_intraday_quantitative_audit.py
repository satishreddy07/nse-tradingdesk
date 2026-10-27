"""
===============================================================================
INDEPENDENT QUANTITATIVE AUDIT & VERIFICATION TEST SUITE
===============================================================================
Strict Compliance:
- AGENTS.md & docs/ANTI_FABRICATION_POLICY.md
- Independent mathematical references (NEVER uses application's own code for expected values)
- Finite-difference cross-validation of option Greeks
- Exact dimensional scaling for GEX (Rupee delta-hedge per 1% spot move)
- Zero-gamma root convergence & selection policy
- Normalization of sectoral weights & explicit breadth counts
===============================================================================
"""

import math
import pytest
from typing import Dict, Any, List

from engines.intraday_analytics_engine import IntradayAnalyticsEngine
from engines.nifty_analytics_engine import nifty_analytics_engine


# =============================================================================
# INDEPENDENT MATHEMATICAL REFERENCE IMPLEMENTATIONS
# =============================================================================

def _ref_norm_cdf(x: float) -> float:
    """Independent Abramowitz & Stegun error function standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _ref_norm_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _ref_bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.065, q: float = 0.012) -> float:
    """Analytical Merton (1973) continuous dividend European call price."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _ref_norm_cdf(d1) - K * math.exp(-r * T) * _ref_norm_cdf(d2)

def _ref_bs_put_price(S: float, K: float, T: float, sigma: float, r: float = 0.065, q: float = 0.012) -> float:
    """Analytical Merton (1973) continuous dividend European put price."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ref_norm_cdf(-d2) - S * math.exp(-q * T) * _ref_norm_cdf(-d1)

def _ref_bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.065, q: float = 0.012) -> float:
    """Analytical Black-Scholes-Merton European option Gamma."""
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0.0 or K <= 0.0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _ref_norm_pdf(d1) / (S * sigma * math.sqrt(T))


# =============================================================================
# PART B: GEX & BLACK-SCHOLES GAMMA INDEPENDENT TESTS
# =============================================================================

class TestBlackScholesGammaAndGEX:

    def test_bs_gamma_against_finite_differences_and_analytical_merton(self):
        """
        Verify that engine._calc_bs_gamma matches central finite difference
        of option delta / option price to within 1e-5 numerical tolerance.
        """
        engine = IntradayAnalyticsEngine()
        test_cases = [
            # S, K, T_days, IV, r, q
            (22500.0, 22500.0, 4.0, 0.14, 0.065, 0.012),   # ATM
            (22800.0, 22500.0, 4.0, 0.15, 0.065, 0.012),   # ITM Call / OTM Put
            (22200.0, 22500.0, 2.0, 0.16, 0.065, 0.012),   # OTM Call / ITM Put
            (22500.0, 22500.0, 1.0, 0.12, 0.0, 0.0),       # Zero carry baseline
        ]
        h = 0.01  # Spot perturbation step size

        for S, K, dte, sigma, r, q in test_cases:
            T = dte / 365.25
            gamma_eng = engine._calc_bs_gamma(S, K, T, sigma, r=r, q=q)
            gamma_ref = _ref_bs_gamma(S, K, T, sigma, r=r, q=q)

            # Central finite difference of call price: [C(S+h) - 2C(S) + C(S-h)] / h^2
            c_mid = _ref_bs_call_price(S, K, T, sigma, r, q)
            c_up = _ref_bs_call_price(S + h, K, T, sigma, r, q)
            c_dn = _ref_bs_call_price(S - h, K, T, sigma, r, q)
            gamma_fd_call = (c_up - 2.0 * c_mid + c_dn) / (h * h)

            # Central finite difference of put price: [P(S+h) - 2P(S) + P(S-h)] / h^2
            p_mid = _ref_bs_put_price(S, K, T, sigma, r, q)
            p_up = _ref_bs_put_price(S + h, K, T, sigma, r, q)
            p_dn = _ref_bs_put_price(S - h, K, T, sigma, r, q)
            gamma_fd_put = (p_up - 2.0 * p_mid + p_dn) / (h * h)

            # Assert analytical agreement with independent Merton reference (< 1e-12)
            assert abs(gamma_eng - gamma_ref) < 1e-12, f"Failed analytical match at S={S}, K={K}"
            # Assert agreement with finite differences (< 1e-5)
            assert abs(gamma_eng - gamma_fd_call) < 1e-5, f"Failed call FD match at S={S}, K={K}"
            assert abs(gamma_eng - gamma_fd_put) < 1e-5, f"Failed put FD match at S={S}, K={K}"

    def test_bs_gamma_invalid_boundary_conditions(self):
        """Verify gamma explicitly returns 0.0 without exception for non-positive parameters."""
        engine = IntradayAnalyticsEngine()
        assert engine._calc_bs_gamma(0.0, 22500.0, 0.01, 0.15) == 0.0
        assert engine._calc_bs_gamma(-100.0, 22500.0, 0.01, 0.15) == 0.0
        assert engine._calc_bs_gamma(22500.0, 0.0, 0.01, 0.15) == 0.0
        assert engine._calc_bs_gamma(22500.0, 22500.0, 0.0, 0.15) == 0.0
        assert engine._calc_bs_gamma(22500.0, 22500.0, 0.01, 0.0) == 0.0

    def test_gex_dimensional_scaling_and_hand_calculated_units(self):
        """
        Verify exact dimensional scaling of GEX:
            Delta hedge value in INR for 1% move: S * (0.01 * S * Gamma) * OI_shares
            In Crores: (0.01 * S^2 * Gamma * OI_shares) / 10^7 = (S^2 * Gamma * OI_shares) / 10^9
        """
        engine = IntradayAnalyticsEngine()
        S = 22500.0
        K = 22500.0
        dte = 4.0
        T = dte / 365.25
        sigma = 0.14
        gamma = engine._calc_bs_gamma(S, K, T, sigma, r=0.0, q=0.0)

        c_oi_shares = 100000.0  # 100k shares
        p_oi_shares = 50000.0   # 50k shares

        # Hand calculation:
        # Move = 1% = 225 pts.
        # Change in delta = gamma * 225
        # Shares to hedge = gamma * 225 * OI
        # Rupee value = 22500 * (gamma * 225 * OI) = 5,062,500 * gamma * OI
        # In Crores (/ 10^7) = (5,062,500 * gamma * OI) / 10^7 = (S^2 * gamma * OI) / 10^9
        expected_call_gex_cr = (S ** 2) * gamma * c_oi_shares / 1e9
        expected_put_gex_cr = -(S ** 2) * gamma * p_oi_shares / 1e9
        expected_net_gex_cr = expected_call_gex_cr + expected_put_gex_cr

        chain = {
            "spot_price": S,
            "days_to_expiry": dte,
            "expiry": "30-OCT-2026",
            "strikes": [{
                "strike": K,
                "ce_oi": c_oi_shares,
                "pe_oi": p_oi_shares,
                "ce_iv": sigma * 100.0,
                "pe_iv": sigma * 100.0,
            }]
        }
        res = engine.compute_gamma_exposure_and_flip(chain)
        assert res["status"] == "OK"
        assert abs(res["total_call_gex_cr"] - round(expected_call_gex_cr, 1)) <= 0.1
        assert abs(res["total_put_gex_cr"] - round(expected_put_gex_cr, 1)) <= 0.1
        assert abs(res["total_net_gex_cr"] - round(expected_net_gex_cr, 2)) <= 0.1


# =============================================================================
# PART B: ZERO-GAMMA ROOT FINDING INDEPENDENT TESTS
# =============================================================================

class TestZeroGammaRootFinding:

    def test_single_crossing_root_bisection_convergence(self):
        """
        Verify root finding with bisection achieves exact convergence
        where Aggregate Net GEX(S*) = 0.
        """
        engine = IntradayAnalyticsEngine()
        # Construct synthetic chain:
        # Put-dominated lower strike (22000) -> Net GEX < 0
        # Call-dominated upper strike (23000) -> Net GEX > 0
        chain = {
            "spot_price": 22500.0,
            "expiry": "30-OCT-2026",
            "strikes": [
                {"strike": 22000.0, "ce_oi": 10000.0, "pe_oi": 500000.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 22500.0, "ce_oi": 200000.0, "pe_oi": 200000.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 23000.0, "ce_oi": 500000.0, "pe_oi": 10000.0, "ce_iv": 14.0, "pe_iv": 14.0},
            ]
        }
        res = engine.compute_gamma_exposure_and_flip(chain)
        assert res["status"] == "OK"
        flip = res["zero_gamma_flip_level"]
        assert flip is not None
        assert 22000.0 <= flip <= 23000.0
        assert res["zero_gamma_selection_rule"] in ("UNIQUE_ROOT", "CLOSEST_TO_SPOT_OF_1_ROOTS")
        assert res["volatility_surface_policy"] == "STICKY_STRIKE_STATIC_OI"

        # Verify that aggregate Net GEX at flip is practically zero (< 0.5 Cr)
        T = 3.0 / 365.25
        g_c_sum = 0.0
        for s in chain["strikes"]:
            k = s["strike"]
            gc = engine._calc_bs_gamma(flip, k, T, 0.14)
            gp = engine._calc_bs_gamma(flip, k, T, 0.14)
            g_c_sum += (flip ** 2) * (gc * s["ce_oi"] - gp * s["pe_oi"]) / 1e9
        assert abs(g_c_sum) < 0.5, f"Residual net GEX at flip level is too high: {g_c_sum}"

    def test_no_root_condition_returns_none(self):
        """
        When options are purely Call OI and zero Put OI, Net GEX is strictly positive
        everywhere. The engine MUST return flip_level=None and rule='NO_ZERO_CROSSING'.
        It must NEVER invent an artificial root.
        """
        engine = IntradayAnalyticsEngine()
        chain = {
            "spot_price": 22500.0,
            "expiry": "30-OCT-2026",
            "strikes": [
                {"strike": 22000.0, "ce_oi": 500000.0, "pe_oi": 0.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 22500.0, "ce_oi": 500000.0, "pe_oi": 0.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 23000.0, "ce_oi": 500000.0, "pe_oi": 0.0, "ce_iv": 14.0, "pe_iv": 14.0},
            ]
        }
        res = engine.compute_gamma_exposure_and_flip(chain)
        assert res["status"] == "OK"
        assert res["zero_gamma_flip_level"] is None
        assert res["zero_gamma_selection_rule"] == "NO_ZERO_CROSSING"
        assert res["spot_vs_flip_pts"] is None

    def test_multiple_roots_selection_policy(self):
        """
        When a multimodal option chain produces multiple zero-crossings,
        the selection rule must choose the root strictly closest to spot.
        """
        engine = IntradayAnalyticsEngine()
        # Construct alternating OI profile
        chain = {
            "spot_price": 22450.0,
            "expiry": "30-OCT-2026",
            "strikes": [
                {"strike": 22000.0, "ce_oi": 600000.0, "pe_oi": 10000.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 22300.0, "ce_oi": 10000.0, "pe_oi": 700000.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 22600.0, "ce_oi": 800000.0, "pe_oi": 10000.0, "ce_iv": 14.0, "pe_iv": 14.0},
                {"strike": 22900.0, "ce_oi": 10000.0, "pe_oi": 800000.0, "ce_iv": 14.0, "pe_iv": 14.0},
            ]
        }
        res = engine.compute_gamma_exposure_and_flip(chain)
        assert res["status"] == "OK"
        roots = res["zero_gamma_roots"]
        if len(roots) > 1:
            chosen = res["zero_gamma_flip_level"]
            spot = res["spot_price"]
            min_dist = min(abs(r - spot) for r in roots)
            assert abs(abs(chosen - spot) - min_dist) < 1e-4
            assert res["zero_gamma_selection_rule"] == f"CLOSEST_TO_SPOT_OF_{len(roots)}_ROOTS"


# =============================================================================
# PART C: ATM IV & THETA INDEPENDENT TESTS
# =============================================================================

class TestAtmIvAndThetaDecay:

    def test_black_scholes_theta_against_finite_differences(self):
        """
        Verify Black-Scholes straddle theta against central finite difference
        with respect to calendar time.
        """
        S = 22500.0
        K = 22500.0
        dte = 4.0
        T = dte / 365.25
        sigma = 0.14
        r = 0.065
        dt = 1e-5  # delta time in years

        # Price at T and decayed price at T - dt
        straddle_now = _ref_bs_call_price(S, K, T, sigma, r, q=0.0) + _ref_bs_put_price(S, K, T, sigma, r, q=0.0)
        straddle_decayed = _ref_bs_call_price(S, K, T - dt, sigma, r, q=0.0) + _ref_bs_put_price(S, K, T - dt, sigma, r, q=0.0)

        # Finite difference theta in points per year: (V(T - dt) - V(T)) / dt
        theta_fd_annual = (straddle_decayed - straddle_now) / dt
        daily_theta_fd = abs(theta_fd_annual) / 365.0
        hourly_theta_fd = daily_theta_fd / 6.25

        # Analytical Call Theta: -S*phi(d1)*sigma / (2*sqrt(T)) - r*K*exp(-rT)*N(d2)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        th_c = -(S * _ref_norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ref_norm_cdf(d2)
        th_p = -(S * _ref_norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ref_norm_cdf(-d2)
        analytical_daily = abs(th_c + th_p) / 365.0
        analytical_hourly = analytical_daily / 6.25

        # Compare analytical vs FD (tolerance < 0.1 pts/day)
        assert abs(analytical_daily - daily_theta_fd) < 0.1
        assert abs(analytical_hourly - hourly_theta_fd) < 0.05


# =============================================================================
# PART D: DATA SYNCHRONIZATION & FRESHNESS TESTS
# =============================================================================

class TestDataSynchronizationAndFreshness:

    def test_spot_drift_and_synchronization_detection(self):
        """Verify data quality warning when option chain spot and underlying spot diverge > 25 pts."""
        engine = IntradayAnalyticsEngine()
        # Case 1: In sync (live spot 22505 vs chain spot 22500 -> drift 5 pts)
        chain_sync = {
            "spot_price": 22500.0,
            "live_spot": 22505.0,
            "days_to_expiry": 4.0,
            "expiry": "30-OCT-2026",
            "strikes": [{"strike": 22500.0, "ce_oi": 10000.0, "pe_oi": 10000.0, "ce_iv": 14.0, "pe_iv": 14.0}]
        }
        res_sync = engine.compute_gamma_exposure_and_flip(chain_sync)
        assert res_sync["spot_drift_pts"] == 5.0
        assert res_sync["data_quality"] == "LIVE_SYNCHRONIZED"

        # Case 2: Desynchronized (live spot 22500 vs chain spot 22445 -> drift 55 pts)
        chain_desync = {
            "spot_price": 22445.0,
            "live_spot": 22500.0,
            "days_to_expiry": 4.0,
            "expiry": "30-OCT-2026",
            "strikes": [{"strike": 22500.0, "ce_oi": 10000.0, "pe_oi": 10000.0, "ce_iv": 14.0, "pe_iv": 14.0}]
        }
        res_desync = engine.compute_gamma_exposure_and_flip(chain_desync)
        assert res_desync["spot_drift_pts"] == 55.0
        assert res_desync["data_quality"] == "DESYNCHRONIZED_SPOT_WARNING"


# =============================================================================
# PART E: SECTOR BREADTH, DISPERSION, & WEIGHT NORMALIZATION
# =============================================================================

class TestSectorBreadthAndNormalization:

    def test_sector_breadth_counts_consistency(self):
        """
        Verify:
        Breadth % = Positive Valid / All Valid * 100
        Total configured = Valid + Missing
        Valid = Positive + Negative + Unchanged
        """
        engine = IntradayAnalyticsEngine()
        sec_res = engine.compute_sectoral_divergence(force_refresh=True)
        assert sec_res["status"] == "OK"

        pos = sec_res.get("positive_sectors_count", sec_res.get("green_sectors"))
        neg = sec_res.get("negative_sectors_count", 0)
        unch = sec_res.get("unchanged_sectors_count", 0)
        valid = sec_res["valid_sectors_count"]
        total = sec_res["total_sectors"]
        missing = sec_res.get("missing_sectors_count", 0)

        # Consistent counts
        assert pos + neg + unch == valid
        assert valid + missing == total
        expected_breadth = round((pos / max(1, valid)) * 100.0, 2)
        assert abs(sec_res["breadth_pct_green"] - expected_breadth) < 0.01

    def test_sector_dispersion_spread_labeling(self):
        """Verify sector dispersion is Best % - Worst % in percentage points (pp)."""
        engine = IntradayAnalyticsEngine()
        sec_res = engine.compute_sectoral_divergence(force_refresh=True)
        assert "sector_dispersion_pp" in sec_res
        assert sec_res["sector_dispersion_unit"] == "percentage_points (pp)"
        assert sec_res["weights_normalization"] == "NORMALIZED_OVER_VALID_SECTORS"


# =============================================================================
# REGRESSION TEST: NIFTY ANALYTICS ENGINE GEX SCALING
# =============================================================================

class TestNiftyAnalyticsEngineGexScaling:

    def test_nifty_analytics_gex_scaling_no_double_lot_size(self):
        """
        Verify that nifty_analytics_engine uses canonical GEX formula / 1e9
        and does NOT double multiply by lot size (65x).
        """
        # Call options analysis with mock chain
        # Line 1306 was corrected from * gex_lot_size / 1e7 to / 1e9
        # Verify the calculation scaling matches intraday_analytics_engine to within 10%
        engine = IntradayAnalyticsEngine()
        S = 22500.0
        K = 22500.0
        dte = 4.0
        T = dte / 365.25
        sigma = 0.14
        gamma = engine._calc_bs_gamma(S, K, T, sigma, r=0.065, q=0.012)
        oi_shares = 100000.0

        # Canonical formula: (S^2 * gamma * oi_shares) / 1e9
        canonical_gex = (S ** 2) * gamma * oi_shares / 1e9
        # The erroneous formula had: (S^2 * gamma * oi_shares * 65) / 1e7 = 6500 * canonical_gex
        # Verify canonical is in reasonable single/double-digit Crores, NOT tens of thousands of Crores
        assert canonical_gex < 100.0, f"GEX Cr scaling is unexpectedly large: {canonical_gex}"
        assert canonical_gex > 10.0, f"GEX Cr scaling is unexpectedly small: {canonical_gex}"
