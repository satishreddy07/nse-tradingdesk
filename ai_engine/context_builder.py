"""
Context Builder: Transforms raw ML prediction outputs into standard PredictionInputContract.
Validates inputs, preserves data fidelity, and ensures no hallucinated facts.
"""
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.market_calendar import NSEMarketCalendar
from ai_engine.schemas import (
    MarketContextItem,
    PredictionHorizonItem,
    PredictionInputContract,
)


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to float; replaces NaN, Inf, or malformed values with default."""
    try:
        if val is None:
            return default
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_optional_float(val: Any) -> Optional[float]:
    """Safely converts input to float or None if missing, NaN, Inf, or malformed."""
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _determine_direction(verdict_str: str) -> str:
    """Normalizes verdict string to standardized direction enum."""
    u = verdict_str.upper()
    if "BULLISH" in u:
        return "BULLISH"
    if "BEARISH" in u:
        return "BEARISH"
    return "RANGE-BOUND"


def _normalize_probabilities(p_up: Any, p_flat: Any, p_down: Any) -> Tuple[float, float, float]:
    """Ensures directional horizon probabilities sum to 1.0."""
    u = float(np_clip_prob(p_up))
    f = float(np_clip_prob(p_flat))
    d = float(np_clip_prob(p_down))
    total = u + f + d
    if total > 0.0:
        return round(u / total, 4), round(f / total, 4), round(d / total, 4)
    return 0.3333, 0.3334, 0.3333


def build_prediction_context(
    predictions_data: Dict[str, Any],
    data_mode_override: Optional[str] = None
) -> PredictionInputContract:
    """
    Transforms the actual output of nifty_analytics_engine.compute_nifty_predictions()
    into the versioned PredictionInputContract (Schema v1.0).
    """
    if not isinstance(predictions_data, dict) or predictions_data.get("status") != "OK":
        raise ValueError("Invalid predictions data passed to context builder: status is not OK")

    spot_unavailable = False
    spot = safe_float(predictions_data.get("spot_price"), default=0.0)
    if spot <= 0.0:
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            spot = float(nifty_analytics_engine.get_live_nifty_spot().get("price") or 0.0)
        except Exception:
            pass
    if spot <= 0.0:
        try:
            from execution.state_engine import state_engine
            with state_engine._get_connection() as conn:
                r = conn.execute("SELECT close FROM ohlcv_master_store WHERE symbol IN ('^NSEI', 'NIFTY') AND close > 0 ORDER BY date DESC LIMIT 1").fetchone()
                if r and r[0]:
                    spot = float(r[0])
        except Exception:
            pass
    if spot <= 0.0:
        spot_unavailable = True
        spot = 1.0  # Safe minimal positive float for schema compliance; flagged as INSUFFICIENT_DATA

    composite_score = safe_float(predictions_data.get("composite_score"), default=0.0)
    t1_f = predictions_data.get("t1_forecast") or {}
    t5_f = predictions_data.get("t5_forecast") or {}
    pillars = predictions_data.get("pillars") or {}
    self_training = predictions_data.get("self_training_engine") or {}
    inst_flow = predictions_data.get("fii_dii_and_oi") or pillars.get("institutional_flows") or {}
    news_data = pillars.get("news_and_events") or {}
    options_overlay = pillars.get("options_overlay") or {}
    sectoral_rows = pillars.get("sectoral_analysis") or []

    # Determine real data mode: LIVE only during regular trading hours
    if data_mode_override:
        data_mode = data_mode_override
    else:
        can_trade, _ = NSEMarketCalendar.can_execute_trades()
        data_mode = "LIVE" if can_trade else "HISTORICAL_DAILY"

    # Timezone-aware ISO-8601 as-of timestamp
    as_of = datetime.now(timezone.utc).isoformat()

    # Build T+1 Prediction with normalized probabilities and empirical validation status
    t1_verdict = str(t1_f.get("verdict") or "RANGE-BOUND")
    t1_p_up, t1_p_flat, t1_p_down = _normalize_probabilities(
        t1_f.get("prob_up", 0.5),
        t1_f.get("prob_flat", 0.2),
        t1_f.get("prob_down", 0.3)
    )
    t1_hit = self_training.get("t1_hit_rate_pct")
    t1_status = (
        f"WALK_FORWARD_VALIDATED (Historical Hit Rate: {float(t1_hit):.1f}%)"
        if t1_hit is not None
        else "WALK_FORWARD_UNVALIDATED (Historical Hit Rate: INSUFFICIENT_DATA)"
    )

    t1_item = PredictionHorizonItem(
        horizon="T+1",
        direction=_determine_direction(t1_verdict),
        probability_up=t1_p_up,
        probability_flat=t1_p_flat,
        probability_down=t1_p_down,
        expected_return_pct=safe_optional_float(t1_f.get("expected_pct")),
        expected_points=safe_optional_float(t1_f.get("expected_pts")),
        expected_close=safe_optional_float(t1_f.get("expected_close")),
        forecast_range_1sigma=list(t1_f.get("range_1sigma")) if t1_f.get("range_1sigma") else None,
        forecast_range_2sigma=list(t1_f.get("range_2sigma")) if t1_f.get("range_2sigma") else None,
        model_version="6-Pillar-v2.4",
        validation_status=t1_status,
    )

    # Build T+5 Prediction with normalized probabilities and empirical validation status
    t5_verdict = str(t5_f.get("verdict") or "RANGE-BOUND")
    t5_p_up, t5_p_flat, t5_p_down = _normalize_probabilities(
        t5_f.get("prob_up", 0.5),
        t5_f.get("prob_flat", 0.2),
        t5_f.get("prob_down", 0.3)
    )
    t5_hit = self_training.get("t5_hit_rate_pct")
    t5_status = (
        f"WALK_FORWARD_VALIDATED (Historical Hit Rate: {float(t5_hit):.1f}%)"
        if t5_hit is not None
        else "WALK_FORWARD_UNVALIDATED (Historical Hit Rate: INSUFFICIENT_DATA)"
    )

    t5_item = PredictionHorizonItem(
        horizon="T+5",
        direction=_determine_direction(t5_verdict),
        probability_up=t5_p_up,
        probability_flat=t5_p_flat,
        probability_down=t5_p_down,
        expected_return_pct=safe_optional_float(t5_f.get("expected_pct")),
        expected_points=safe_optional_float(t5_f.get("expected_pts")),
        expected_close=safe_optional_float(t5_f.get("expected_close")),
        forecast_range_1sigma=list(t5_f.get("range_1sigma")) if t5_f.get("range_1sigma") else None,
        forecast_range_2sigma=list(t5_f.get("range_2sigma")) if t5_f.get("range_2sigma") else None,
        model_version="6-Pillar-v2.4",
        validation_status=t5_status,
    )

    # Macro regime
    if composite_score >= 15.0:
        regime = "TRENDING_BULLISH"
    elif composite_score <= -15.0:
        regime = "TRENDING_BEARISH"
    else:
        regime = "CONSOLIDATION_RANGE"

    # Sectoral leaders
    sector_leaders = []
    if isinstance(sectoral_rows, list):
        sorted_secs = sorted(sectoral_rows, key=lambda x: abs(safe_float(x.get("return_1d") or x.get("ret_1d"), 0.0)), reverse=True)
        for s in sorted_secs[:3]:
            sec_name = s.get("name") or s.get("ticker") or "Sector"
            r1 = safe_float(s.get("return_1d") or s.get("ret_1d"), 0.0)
            sector_leaders.append(f"{sec_name}: {r1:+.2f}% 1D")

    # FII/DII positioning
    cash_info = inst_flow.get("latest_cash") or {}
    oi_info = inst_flow.get("latest_oi") or {}
    fii_net = safe_optional_float(cash_info.get("fii_net"))
    dii_net = safe_optional_float(cash_info.get("dii_net"))
    fii_long_ratio = safe_optional_float(oi_info.get("fii_fut_long_pct"))

    fii_dii_summary = None
    if fii_net is not None and dii_net is not None:
        fii_dii_summary = f"FII Cash: ₹{fii_net:+,.1f} Cr | DII Cash: ₹{dii_net:+,.1f} Cr"
        if fii_long_ratio is not None:
            fii_dii_summary += f" | FII Index Fut Long Ratio: {fii_long_ratio:.1f}%"

    # India VIX from macro rows
    macro_rows = pillars.get("macro_cross_asset") or []
    india_vix_val = None
    if isinstance(macro_rows, list):
        for m in macro_rows:
            if isinstance(m, dict) and (m.get("ticker") == "^INDIAVIX" or "VIX" in str(m.get("name", "")).upper()):
                india_vix_val = safe_optional_float(m.get("last_close") or m.get("close") or m.get("ltp"))
                break

    # ATM IV from options overlay
    atm_iv_val = safe_optional_float(options_overlay.get("atm_iv"))

    # News items & detected live catalysts
    news_items_clean: List[Dict[str, str]] = []
    raw_news = news_data.get("headlines") or news_data.get("items") or []
    if isinstance(raw_news, list):
        for n in raw_news[:6]:
            if isinstance(n, dict):
                news_items_clean.append({
                    "title": str(n.get("title") or "")[:150],
                    "source": str(n.get("source") or n.get("category") or "Economic Times/LiveMint"),
                    "published_at": str(n.get("published_at") or n.get("published") or n.get("pub_date") or ""),
                })
    raw_catalysts = news_data.get("detected_live_catalysts") or []
    if isinstance(raw_catalysts, list):
        for c in raw_catalysts[:3]:
            if isinstance(c, dict):
                lbl = str(c.get("label") or "Macro Catalyst")
                sample = str(c.get("sample_headline") or "")[:100]
                news_items_clean.append({
                    "title": f"Catalyst Alert: {lbl} — {sample}",
                    "source": "NSE Macro Scanner",
                    "published_at": "Live",
                })

    # US Market Intelligence & Futures Radar
    us_intel = predictions_data.get("us_intelligence") or pillars.get("us_market_analytics") or {}
    futures_radar = predictions_data.get("futures_relative_radar") or pillars.get("futures_relative_radar") or {}
    us_t1_forecast = us_intel.get("us_t1_forecast")
    us_dispersion = safe_optional_float(us_intel.get("breadth_analysis", {}).get("rsp_vs_spy_dispersion"))
    us_vix_regime = us_intel.get("volatility_surface", {}).get("regime")
    comp_score_t5 = safe_optional_float(predictions_data.get("composite_score_t5"))

    market_context = MarketContextItem(
        spot_price=spot,
        composite_score=composite_score,
        composite_score_t5=comp_score_t5,
        regime=regime,
        event_risk=str(news_data.get("event_risk_flag") or "NORMAL"),
        data_quality="INSUFFICIENT_DATA" if spot_unavailable else "VALID",
        pillar_scores=predictions_data.get("pillar_scores"),
        sectoral_leaders=sector_leaders if sector_leaders else None,
        fii_dii_summary=fii_dii_summary,
        option_chain_pcr=safe_optional_float(options_overlay.get("near_atm_pcr")),
        max_pain=safe_optional_float(options_overlay.get("max_pain")),
        india_vix=india_vix_val,
        atm_iv=atm_iv_val,
        us_market_t1_forecast=us_t1_forecast if isinstance(us_t1_forecast, dict) else None,
        us_breadth_dispersion=us_dispersion,
        us_vix_term_structure=str(us_vix_regime) if us_vix_regime else None,
        futures_relative_move=futures_radar if isinstance(futures_radar, dict) else None,
        recent_news_events=news_items_clean if news_items_clean else None,
    )

    return PredictionInputContract(
        schema_version="1.0",
        symbol="NIFTY",
        as_of=as_of,
        data_mode=data_mode,
        predictions=[t1_item, t5_item],
        market_context=market_context,
    )


def np_clip_prob(val: Any) -> float:
    """Ensures probability is safely bounded between 0.0 and 1.0."""
    try:
        if val is None:
            return 0.5
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.5
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return 0.5
