"""
engines/strategy_adjustment_engine.py
=====================================
Active Position Watchdog, Profit Locking, High-Water Mark Trailing & Auto-Adjustment Engine:
1. Multi-Layer Position Defense:
   - Layer 1: Fixed Capital Preservation Stop-Loss (Hard Exit).
   - Layer 2: Two-Stage Milestone Profit Locking:
     * Milestone 1 (40% of Target P&L reached): Move Stop-Loss to ₹0.0 (Cost-to-Cost / Break-Even).
     * Milestone 2 (65% of Target P&L reached): Lock in 50% Profit Floor + activate 15% High-Water Mark Trailing Stop.
   - Layer 3: High-Water Mark Trailing Exit:
     * If P&L drops below trailing stop floor -> Trigger TRAILING_STOP_EXITED to bank locked profit.
   - Layer 4: Dynamic Auto-Adjustment for Breached Wings:
     * Iron Fly Adjustment: When spot breaches breakeven, roll the untested decayed wing closer to spot.
     * BWB Adjustment: When spot approaches broken wing, tighten wing spacing to eliminate tail risk.
   - Layer 5: MOC (Market-on-Close) Auto-Squareoff:
     * Enforces strict 15:15 IST closure for all INTRADAY strategies so zero intraday risk is held overnight.

2. Closed-Loop ML Telemetry:
   - Records every exit, P&L, slippage, and duration to `unified_trade_learning_dataset`
     to power continuous reinforcement learning and threshold calibration.
"""

import json
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

from core.data_quality import now_ist, format_ist
from core.market_calendar import NSEMarketCalendar
from core.config import NIFTY_LOT_SIZE, resolve_nifty_lot_size
from execution.state_engine import state_engine

logger = logging.getLogger("StrategyAdjustmentEngine")


class StrategyAdjustmentEngine:
    """
    Continuous watchdog daemon monitoring all active options strategies.
    Ensures rigorous risk control, profit locking, and algorithmic defense.
    """

    def __init__(self):
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensures dedicated adjustment log table exists in master database."""
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS options_strategy_adjustments_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    adjustment_type TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    spot_at_adjustment REAL,
                    old_legs_json TEXT,
                    new_legs_json TEXT,
                    credit_collected REAL DEFAULT 0.0,
                    timestamp_ist TEXT NOT NULL
                )
                """)
                conn.commit()

    def tick_watchdog(self, force_execution: bool = False) -> Dict[str, Any]:
        """
        Main watchdog loop: Inspects all ACTIVE strategies across Intraday, Weekly, and Monthly horizons.
        Runs live quote reconciliations, updates trailing stops, triggers exits, and applies adjustments.
        """
        now_dt = now_ist()
        now_str = format_ist(now_dt)
        status_info = NSEMarketCalendar.get_market_status(now_dt)
        is_market_open = bool(status_info.get("is_open", False))

        # Allow execution if market is open OR force_execution is True (for testing/MOC)
        if not is_market_open and not force_execution:
            return {"status": "STANDBY", "reason": "Market is closed", "timestamp_ist": now_str}

        with state_engine._lock:
            with state_engine._get_connection() as conn:
                active_strats = conn.execute("""
                    SELECT * FROM options_strategies
                    WHERE status = 'ACTIVE'
                    ORDER BY id ASC
                """).fetchall()

        if not active_strats:
            return {"status": "IDLE", "active_count": 0, "timestamp_ist": now_str}

        # Fetch live spot & option ticks
        spot_price = 0.0
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            spot_price = float(nifty_analytics_engine.get_live_nifty_spot().get("price") or 0.0)
        except Exception:
            pass

        updated_count = 0
        exited_count = 0
        adjusted_count = 0

        # Check time for Intraday MOC (15:15 IST)
        time_hm = now_dt.strftime("%H:%M")
        is_moc_time = (time_hm >= "15:15" and time_hm <= "15:35")

        for row in active_strats:
            strat_id = row["strategy_id"]
            horizon = str(row["expiry_type"] or "WEEKLY").upper()
            strat_name = row["strategy_name"]
            legs_str = row["legs_json"] or "[]"
            target_pnl = abs(float(row["target_pnl"] or 10000.0))
            raw_sl = float(row["stoploss_pnl"] or -5000.0)
            # Invariant: Stop Loss PnL threshold MUST ALWAYS be strictly negative
            stoploss_pnl = -abs(raw_sl) if raw_sl != 0.0 else -5000.0
            raw_trailing = row["trailing_stop_pnl"]
            trailing_status = str(row["trailing_status"] or "INACTIVE") if "trailing_status" in row.keys() else "INACTIVE"
            if raw_trailing is None or trailing_status == "INACTIVE":
                trailing_stop_pnl = stoploss_pnl
            else:
                # Sanitize trailing stop: if positive before Milestone 1 (40% target), clamp to stoploss_pnl
                trailing_stop_pnl = float(raw_trailing)
                if trailing_stop_pnl > 0.0 and float(row["highest_pnl_reached"] or 0.0) < (target_pnl * 0.40):
                    trailing_stop_pnl = stoploss_pnl
            highest_pnl = float(row["highest_pnl_reached"] or 0.0)
            created_at = row["created_at"]

            try:
                legs = json.loads(legs_str)
            except Exception:
                legs = []

            # Compute current mark-to-market P&L across all legs
            current_pnl, updated_legs = self._calculate_current_pnl(legs, spot_price, horizon)

            # -----------------------------------------------------------------
            # MOC AUTO-SQUAREOFF FOR INTRADAY AT 15:15 IST
            # -----------------------------------------------------------------
            if "INTRADAY" in horizon and is_moc_time:
                self._execute_exit(
                    strategy_id=strat_id,
                    exit_reason="MOC_AUTO_CLOSED",
                    exit_note="15:15 IST Market-On-Close Rule: Intraday options positions automatically closed to eliminate overnight gap risk.",
                    spot_exit=spot_price,
                    realized_pnl=current_pnl
                )
                exited_count += 1
                continue

            if current_pnl > highest_pnl:
                highest_pnl = current_pnl

            # -----------------------------------------------------------------
            # LAYER 1: FIXED STOP-LOSS HIT
            # -----------------------------------------------------------------
            # Enforce that stoploss_pnl is strictly negative and PnL breached it
            if stoploss_pnl < 0.0 and current_pnl <= stoploss_pnl:
                self._execute_exit(
                    strategy_id=strat_id,
                    exit_reason="STOPLOSS_EXITED",
                    exit_note=f"Fixed Stop-Loss triggered: Current P&L (₹{current_pnl:.2f}) reached stop threshold (₹{stoploss_pnl:.2f}). Capital preserved.",
                    spot_exit=spot_price,
                    realized_pnl=current_pnl
                )
                exited_count += 1
                continue

            # -----------------------------------------------------------------
            # LAYER 2: TWO-STAGE MILESTONE PROFIT LOCKING
            # -----------------------------------------------------------------
            note_suffix = ""
            # Milestone 1: 40% Target Reached -> Move Stop to Cost-to-Cost (₹0 Break-Even)
            if current_pnl >= (target_pnl * 0.40) and trailing_stop_pnl < 0.0:
                trailing_stop_pnl = 0.0
                note_suffix = " • Milestone 1: Cost-to-Cost Break-Even Locked"
                logger.info(f"🔒 [{strat_id}] Milestone 1 Hit: Trailing Stop moved to Break-Even (₹0.0).")

            # Milestone 2: 65% Target Reached -> Lock 50% Profit Floor + 15% High-Water Mark Trailing
            if current_pnl >= (target_pnl * 0.65):
                profit_floor = round(current_pnl * 0.50, 2)
                hwm_trailing = round(highest_pnl - (highest_pnl * 0.15), 2)
                new_trail = max(trailing_stop_pnl, profit_floor, hwm_trailing)
                if new_trail > trailing_stop_pnl:
                    trailing_stop_pnl = new_trail
                    note_suffix = f" • Milestone 2: 50% Profit Floor Locked at ₹{trailing_stop_pnl:.0f}"
                    logger.info(f"💰 [{strat_id}] Milestone 2 Hit: Trailing Stop updated to ₹{trailing_stop_pnl:.2f}.")

            # -----------------------------------------------------------------
            # LAYER 3: TRAILING STOP HIT
            # -----------------------------------------------------------------
            if trailing_stop_pnl > stoploss_pnl and current_pnl < trailing_stop_pnl:
                self._execute_exit(
                    strategy_id=strat_id,
                    exit_reason="TRAILING_STOP_EXITED",
                    exit_note=f"Trailing Stop hit: Locked profit protected at ₹{trailing_stop_pnl:.2f} (Current P&L: ₹{current_pnl:.2f}, Peak: ₹{highest_pnl:.2f}).",
                    spot_exit=spot_price,
                    realized_pnl=current_pnl
                )
                exited_count += 1
                continue

            # -----------------------------------------------------------------
            # FULL TARGET PROFIT HIT
            # -----------------------------------------------------------------
            if current_pnl >= target_pnl:
                self._execute_exit(
                    strategy_id=strat_id,
                    exit_reason="TARGET_EXITED",
                    exit_note=f"Full Target Profit reached: Strategy banked ₹{current_pnl:.2f} (Target: ₹{target_pnl:.2f}).",
                    spot_exit=spot_price,
                    realized_pnl=current_pnl
                )
                exited_count += 1
                continue

            # -----------------------------------------------------------------
            # LAYER 4: DYNAMIC AUTO-ADJUSTMENTS FOR TESTED WINGS
            # -----------------------------------------------------------------
            adjusted, adj_legs, adj_note = self._check_and_apply_adjustments(strat_id, strat_name, updated_legs, spot_price, horizon)
            if adjusted:
                updated_legs = adj_legs
                adjusted_count += 1
                note_suffix += f" • {adj_note}"

            # Update database state
            with state_engine._lock:
                with state_engine._get_connection() as conn:
                    conn.execute("""
                        UPDATE options_strategies
                        SET trailing_stop_pnl = ?,
                            highest_pnl_reached = ?,
                            legs_json = ?,
                            ai_rationale = ai_rationale || ?
                        WHERE strategy_id = ?
                    """, (trailing_stop_pnl, highest_pnl, json.dumps(updated_legs), note_suffix, strat_id))
                    conn.commit()

            updated_count += 1

        return {
            "status": "OK",
            "active_evaluated": len(active_strats),
            "updated_count": updated_count,
            "exited_count": exited_count,
            "adjusted_count": adjusted_count,
            "timestamp_ist": now_str
        }

    def _calculate_current_pnl(self, legs: List[Dict[str, Any]], spot_price: float, horizon: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculates live P&L across all option legs using live quotes or BSM pricing."""
        total_pnl = 0.0
        updated = []

        from engines.kotak_neo_engine import kotak_neo_engine
        live_ticks = kotak_neo_engine.get_near_atm_option_ticks(strike_range=20)

        for leg in legs:
            l = dict(leg)
            action = l.get("action", "SELL").upper()
            opt_type = l.get("type", "CE").upper()
            strike = int(l.get("strike", 0))
            entry_p = float(l.get("entry_price") or l.get("premium") or 0.0)
            leg_ls = l.get("lot_size")
            if not leg_ls or int(leg_ls) <= 0:
                try:
                    exp = l.get("expiry_date")
                    as_of = l.get("as_of_date")
                    if not exp and not as_of:
                        cal_exp = NSEMarketCalendar.get_active_options_expiries()
                        exp = cal_exp.get("weekly_expiry")
                        as_of = now_ist().date()
                    leg_ls = resolve_nifty_lot_size(
                        expiry_date=exp,
                        as_of_date=as_of,
                        contract_metadata=l,
                        contract_type=l.get("contract_type") or l.get("expiry_type"),
                        symbol="NIFTY",
                        strict=True
                    )
                except Exception as ex:
                    raise ValueError(f"Cannot calculate current P&L: unable to resolve authoritative lot size for leg {strike} {opt_type}: {ex}")
            leg_ls = int(leg_ls)
            qty = int(l.get("qty") or (int(l.get("lots", 1)) * leg_ls))

            # Try live tick quote
            tick_key = f"{strike}_{opt_type}"
            live_quote = live_ticks.get(tick_key, {})
            current_p = float(live_quote.get("ltp") or 0.0)

            # Fallback to analytical pricing if live quote unavailable outside market hours
            if current_p <= 0.0:
                current_p = entry_p # Preserve entry if zero quote

            l["current_price"] = current_p
            if action == "SELL":
                leg_pnl = (entry_p - current_p) * qty
            else:
                leg_pnl = (current_p - entry_p) * qty

            l["pnl"] = round(leg_pnl, 2)
            total_pnl += leg_pnl
            updated.append(l)

        return round(total_pnl, 2), updated

    def _get_option_quote(self, strike: int, opt_type: str, spot_price: float) -> Optional[float]:
        """
        Attempts to get authentic market LTP from Kotak ticks or active chain.
        Returns positive finite float if valid market quote found, else None.
        Never falls back to theoretical BSM calculations.
        """
        if strike is None or opt_type is None:
            return None

        try:
            strike_int = int(strike)
            opt_type_str = str(opt_type).strip().upper()
        except (ValueError, TypeError):
            return None

        if opt_type_str not in ("CE", "PE"):
            return None

        # 1. Try Kotak Neo live ticks
        try:
            from engines.kotak_neo_engine import kotak_neo_engine
            live_ticks = kotak_neo_engine.get_near_atm_option_ticks(strike_range=30)
            tick_key = f"{strike_int}_{opt_type_str}"
            if tick_key in live_ticks:
                raw_ltp = live_ticks[tick_key].get("ltp")
                if raw_ltp is not None:
                    ltp = float(raw_ltp)
                    if math.isfinite(ltp) and ltp > 0.0:
                        return round(ltp, 2)
        except Exception:
            pass

        # 2. Try active option chain from nifty_analytics_engine
        try:
            from engines.nifty_analytics_engine import nifty_analytics_engine
            chain_data = nifty_analytics_engine.fetch_and_analyze_option_chain(force_refresh=False)
            strikes = (
                chain_data.get("weekly", {}).get("strikes")
                or chain_data.get("weekly", {}).get("all_strikes", [])
            )
            for s in strikes:
                if int(s.get("strike", 0)) == strike_int:
                    ltp_key = "ce_ltp" if opt_type_str == "CE" else "pe_ltp"
                    raw_ltp = s.get(ltp_key)
                    if raw_ltp is not None:
                        ltp = float(raw_ltp)
                        if math.isfinite(ltp) and ltp > 0.0:
                            return round(ltp, 2)
        except Exception:
            pass

        return None

    def _check_and_apply_adjustments(
        self,
        strategy_id: str,
        strat_name: str,
        legs: List[Dict[str, Any]],
        spot_price: float,
        horizon: str = "WEEKLY"
    ) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Evaluates dynamic adjustment rules for Intraday, Weekly, and Monthly horizons:
        1. Institutional Iron Fly:
           - Intraday Trigger: ±70 pts drift from ATM.
           - Weekly/Monthly Trigger: ±120 pts drift from ATM.
           - Action: Roll decayed untested short & long wing 50 pts closer to spot.
        2. Bull Broken Wing Butterfly (PE BWB):
           - Trigger: Spot drops within 25 pts of short strike K2 (threatens broken wing).
           - Action: Roll long K3 PE wing up 50 pts to tighten wing into symmetric fly and eliminate tail risk.
        3. Bear Broken Wing Butterfly (CE BWB):
           - Trigger: Spot rallies within 25 pts of short strike K2 (threatens broken wing).
           - Action: Roll long K3 CE wing down 50 pts to tighten wing into symmetric fly and eliminate tail risk.
        4. Call Ratio Backspread (1x2 Call):
           - Trigger: Spot stalls in Valley of Death between K1 and K2.
           - Action: Deploy OTM Put Credit Spread hedge to inject credit and finance valley drag.
        5. Put Ratio Backspread (1x2 Put):
           - Trigger: Spot stalls in Valley of Death between K2 and K1.
           - Action: Deploy OTM Call Credit Spread hedge to inject credit and finance valley drag.
        """
        if not legs or spot_price <= 0:
            return False, legs, ""

        import copy
        old_legs = copy.deepcopy(legs)
        is_intraday = ("INTRADAY" in horizon.upper())
        drift_threshold = 70.0 if is_intraday else 120.0

        from engines.advanced_options_pricing_engine import calculate_bsm_greeks

        # ---------------------------------------------------------------------
        # 1. INSTITUTIONAL IRON FLY ADJUSTMENT
        # ---------------------------------------------------------------------
        if "Iron Fly" in strat_name and len(legs) >= 4:
            atm_ce = next((l for l in legs if l["action"] == "SELL" and l["type"] == "CE"), None)
            atm_pe = next((l for l in legs if l["action"] == "SELL" and l["type"] == "PE"), None)
            wing_ce = next((l for l in legs if l["action"] == "BUY" and l["type"] == "CE"), None)
            wing_pe = next((l for l in legs if l["action"] == "BUY" and l["type"] == "PE"), None)

            if atm_ce and atm_pe:
                atm_strike = atm_ce["strike"]
                drift = spot_price - atm_strike

                # Drift Down: Call side decayed -> Roll Call wing down 50 pts
                if drift <= -drift_threshold and atm_ce.get("state") != "ADJUSTED_ROLLED":
                    new_ce_strike = atm_strike - 50
                    new_w_k = (wing_ce["strike"] - 50) if wing_ce else None

                    # Market quote verification: Fail-closed before modifying position legs
                    px_atm = self._get_option_quote(new_ce_strike, "CE", spot_price)
                    px_w = self._get_option_quote(new_w_k, "CE", spot_price) if wing_ce else None

                    missing = []
                    if px_atm is None or px_atm <= 0.0:
                        missing.append(f"{new_ce_strike} CE (ATM roll)")
                    if wing_ce and (px_w is None or px_w <= 0.0):
                        missing.append(f"{new_w_k} CE (wing roll)")

                    if missing:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {', '.join(missing)}. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    # Analytical Greeks update & quote assignment
                    g = calculate_bsm_greeks(spot_price, new_ce_strike, 1.0 / 365.0, 0.07, 0.135, "CE")
                    atm_ce["strike"] = new_ce_strike
                    atm_ce["state"] = "ADJUSTED_ROLLED"
                    atm_ce["entry_price"] = round(px_atm, 2)
                    atm_ce["price"] = round(px_atm, 2)
                    atm_ce["premium"] = round(px_atm, 2)
                    atm_ce["quote_source"] = "LIVE_MARKET_QUOTE"
                    atm_ce["greeks"] = g

                    if wing_ce and px_w is not None:
                        gw = calculate_bsm_greeks(spot_price, new_w_k, 1.0 / 365.0, 0.07, 0.135, "CE")
                        wing_ce["strike"] = new_w_k
                        wing_ce["entry_price"] = round(px_w, 2)
                        wing_ce["price"] = round(px_w, 2)
                        wing_ce["premium"] = round(px_w, 2)
                        wing_ce["quote_source"] = "LIVE_MARKET_QUOTE"
                        wing_ce["greeks"] = gw

                    adj_note = f"Auto-Adjustment [{horizon}]: Rolled untested Call wing down to {new_ce_strike} CE (Drift: {drift:.1f} pts) to expand lower breakeven."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "IRON_FLY_WING_ROLL_DOWN", adj_note, spot_price, old_legs, legs, credit_collected=px_atm)
                    return True, legs, adj_note

                # Drift Up: Put side decayed -> Roll Put wing up 50 pts
                elif drift >= drift_threshold and atm_pe.get("state") != "ADJUSTED_ROLLED":
                    new_pe_strike = atm_strike + 50
                    new_w_k = (wing_pe["strike"] + 50) if wing_pe else None

                    # Market quote verification: Fail-closed before modifying position legs
                    px_atm = self._get_option_quote(new_pe_strike, "PE", spot_price)
                    px_w = self._get_option_quote(new_w_k, "PE", spot_price) if wing_pe else None

                    missing = []
                    if px_atm is None or px_atm <= 0.0:
                        missing.append(f"{new_pe_strike} PE (ATM roll)")
                    if wing_pe and (px_w is None or px_w <= 0.0):
                        missing.append(f"{new_w_k} PE (wing roll)")

                    if missing:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {', '.join(missing)}. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    g = calculate_bsm_greeks(spot_price, new_pe_strike, 1.0 / 365.0, 0.07, 0.135, "PE")
                    atm_pe["strike"] = new_pe_strike
                    atm_pe["state"] = "ADJUSTED_ROLLED"
                    atm_pe["entry_price"] = round(px_atm, 2)
                    atm_pe["price"] = round(px_atm, 2)
                    atm_pe["premium"] = round(px_atm, 2)
                    atm_pe["quote_source"] = "LIVE_MARKET_QUOTE"
                    atm_pe["greeks"] = g

                    if wing_pe and px_w is not None:
                        gw = calculate_bsm_greeks(spot_price, new_w_k, 1.0 / 365.0, 0.07, 0.135, "PE")
                        wing_pe["strike"] = new_w_k
                        wing_pe["entry_price"] = round(px_w, 2)
                        wing_pe["price"] = round(px_w, 2)
                        wing_pe["premium"] = round(px_w, 2)
                        wing_pe["quote_source"] = "LIVE_MARKET_QUOTE"
                        wing_pe["greeks"] = gw

                    adj_note = f"Auto-Adjustment [{horizon}]: Rolled untested Put wing up to {new_pe_strike} PE (Drift: {drift:+.1f} pts) to expand upper breakeven."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "IRON_FLY_WING_ROLL_UP", adj_note, spot_price, old_legs, legs, credit_collected=px_atm)
                    return True, legs, adj_note

        # ---------------------------------------------------------------------
        # 2. BULL BROKEN WING BUTTERFLY (PE BWB) ADJUSTMENT
        # ---------------------------------------------------------------------
        elif "Bull Broken Wing" in strat_name or ("Bull" in strat_name and any(l["type"] == "PE" for l in legs)):
            short_pe = next((l for l in legs if l["action"] == "SELL" and l["type"] == "PE"), None)
            if short_pe:
                k2 = short_pe["strike"]
                long_pe_k3 = next((l for l in legs if l["action"] == "BUY" and l["type"] == "PE" and l["strike"] < k2), None)
                # If spot drops and threatens short strike K2 (approaches broken wing risk zone)
                threat_threshold = k2 + (25.0 if is_intraday else 50.0)
                if spot_price <= threat_threshold and long_pe_k3 and long_pe_k3.get("state") != "ADJUSTED_TIGHTENED":
                    old_k3 = long_pe_k3["strike"]
                    new_k3 = old_k3 + 50
                    px = self._get_option_quote(new_k3, "PE", spot_price)
                    if px is None or px <= 0.0:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {new_k3} PE. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    long_pe_k3["strike"] = new_k3
                    long_pe_k3["state"] = "ADJUSTED_TIGHTENED"
                    g = calculate_bsm_greeks(spot_price, new_k3, 1.0 / 365.0, 0.07, 0.135, "PE")
                    long_pe_k3["entry_price"] = round(px, 2)
                    long_pe_k3["price"] = round(px, 2)
                    long_pe_k3["premium"] = round(px, 2)
                    long_pe_k3["quote_source"] = "LIVE_MARKET_QUOTE"
                    long_pe_k3["greeks"] = g

                    adj_note = f"Auto-Adjustment [{horizon}]: Tightened broken put wing from {old_k3} to {new_k3} PE. Neutralized downside gap risk."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "BULL_BWB_WING_TIGHTEN_UP", adj_note, spot_price, old_legs, legs)
                    return True, legs, adj_note

        # ---------------------------------------------------------------------
        # 3. BEAR BROKEN WING BUTTERFLY (CE BWB) ADJUSTMENT
        # ---------------------------------------------------------------------
        elif "Bear Broken Wing" in strat_name or ("Bear" in strat_name and any(l["type"] == "CE" for l in legs)):
            short_ce = next((l for l in legs if l["action"] == "SELL" and l["type"] == "CE"), None)
            if short_ce:
                k2 = short_ce["strike"]
                long_ce_k3 = next((l for l in legs if l["action"] == "BUY" and l["type"] == "CE" and l["strike"] > k2), None)
                # If spot rallies and threatens short strike K2 (approaches broken wing risk zone)
                threat_threshold = k2 - (25.0 if is_intraday else 50.0)
                if spot_price >= threat_threshold and long_ce_k3 and long_ce_k3.get("state") != "ADJUSTED_TIGHTENED":
                    old_k3 = long_ce_k3["strike"]
                    new_k3 = old_k3 - 50
                    px = self._get_option_quote(new_k3, "CE", spot_price)
                    if px is None or px <= 0.0:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {new_k3} CE. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    long_ce_k3["strike"] = new_k3
                    long_ce_k3["state"] = "ADJUSTED_TIGHTENED"
                    g = calculate_bsm_greeks(spot_price, new_k3, 1.0 / 365.0, 0.07, 0.135, "CE")
                    long_ce_k3["entry_price"] = round(px, 2)
                    long_ce_k3["price"] = round(px, 2)
                    long_ce_k3["premium"] = round(px, 2)
                    long_ce_k3["quote_source"] = "LIVE_MARKET_QUOTE"
                    long_ce_k3["greeks"] = g

                    adj_note = f"Auto-Adjustment [{horizon}]: Tightened broken call wing from {old_k3} to {new_k3} CE. Neutralized upside breakout risk."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "BEAR_BWB_WING_TIGHTEN_DOWN", adj_note, spot_price, old_legs, legs)
                    return True, legs, adj_note

        # ---------------------------------------------------------------------
        # 4. CALL RATIO BACKSPREAD (1x2 Call Backspread) ADJUSTMENT
        # ---------------------------------------------------------------------
        elif "Call Ratio Backspread" in strat_name or ("Call" in strat_name and "Backspread" in strat_name):
            short_ce = next((l for l in legs if l["action"] == "SELL" and l["type"] == "CE"), None)
            long_ce = next((l for l in legs if l["action"] == "BUY" and l["type"] == "CE"), None)
            if short_ce and long_ce:
                k1 = short_ce["strike"]
                k2 = long_ce["strike"]
                has_hedge = any(l["type"] == "PE" for l in legs)
                # Valley of Death: Spot stalls between K1 and K2 without explosive breakout
                if k1 <= spot_price <= k2 and not has_hedge and short_ce.get("state") != "ADJUSTED_VALLEY_FINANCED":
                    # Sell OTM Put Credit Spread (K1 - 100 / K1 - 150) to finance valley drag
                    hedge_short_k = k1 - 100
                    hedge_long_k = k1 - 150
                    px_short = self._get_option_quote(hedge_short_k, "PE", spot_price)
                    px_long = self._get_option_quote(hedge_long_k, "PE", spot_price)

                    missing = []
                    if px_short is None or px_short <= 0.0:
                        missing.append(f"{hedge_short_k} PE (short hedge)")
                    if px_long is None or px_long <= 0.0:
                        missing.append(f"{hedge_long_k} PE (long hedge)")

                    if missing:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {', '.join(missing)}. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    g_hs = calculate_bsm_greeks(spot_price, hedge_short_k, 1.0 / 365.0, 0.07, 0.135, "PE")
                    g_hl = calculate_bsm_greeks(spot_price, hedge_long_k, 1.0 / 365.0, 0.07, 0.135, "PE")

                    strat_ls = next((int(l["lot_size"]) for l in legs if l.get("lot_size") and int(l["lot_size"]) > 0), None)
                    if not strat_ls and legs:
                        for l in legs:
                            if l.get("qty") and int(l["qty"]) > 0:
                                q = int(l["qty"])
                                num_l = int(l.get("lots") or 1)
                                if num_l > 0 and q % num_l == 0:
                                    cand_ls = q // num_l
                                    if cand_ls in (25, 50, 65, 75):
                                        strat_ls = cand_ls
                                        break
                    if not strat_ls:
                        exp = next((l.get("expiry_date") for l in legs if l.get("expiry_date")), None)
                        if exp:
                            try:
                                strat_ls = resolve_nifty_lot_size(
                                    expiry_date=exp,
                                    contract_type=horizon,
                                    symbol="NIFTY",
                                    strict=True
                                )
                            except Exception as ex:
                                defer_note = f"Adjustment deferred [{horizon}]: Cannot resolve authoritative lot size for hedge legs: {ex}"
                                logger.error(f"🛑 [{strategy_id}] {defer_note}")
                                return False, legs, defer_note
                        else:
                            strat_ls = resolve_nifty_lot_size(strict=False)

                    credit_got = round((px_short - px_long) * strat_ls, 2)

                    legs.append({
                        "action": "SELL",
                        "type": "PE",
                        "strike": hedge_short_k,
                        "qty": strat_ls,
                        "lots": 1,
                        "lot_size": strat_ls,
                        "ratio": 1,
                        "entry_price": round(px_short, 2),
                        "price": round(px_short, 2),
                        "premium": round(px_short, 2),
                        "current_price": round(px_short, 2),
                        "quote_source": "LIVE_MARKET_QUOTE",
                        "greeks": g_hs,
                        "state": "HEDGE_LEG"
                    })
                    legs.append({
                        "action": "BUY",
                        "type": "PE",
                        "strike": hedge_long_k,
                        "qty": strat_ls,
                        "lots": 1,
                        "lot_size": strat_ls,
                        "ratio": 1,
                        "entry_price": round(px_long, 2),
                        "price": round(px_long, 2),
                        "premium": round(px_long, 2),
                        "current_price": round(px_long, 2),
                        "quote_source": "LIVE_MARKET_QUOTE",
                        "greeks": g_hl,
                        "state": "HEDGE_LEG"
                    })
                    short_ce["state"] = "ADJUSTED_VALLEY_FINANCED"

                    adj_note = f"Auto-Adjustment [{horizon}]: Deployed Put Credit Spread ({hedge_short_k}/{hedge_long_k} PE, Credit: ₹{credit_got:.0f}) to finance Valley of Death stall at {k1}-{k2}."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "CALL_RATIO_VALLEY_FINANCED", adj_note, spot_price, old_legs, legs, credit_collected=credit_got)
                    return True, legs, adj_note

        # ---------------------------------------------------------------------
        # 5. PUT RATIO BACKSPREAD (1x2 Put Backspread) ADJUSTMENT
        # ---------------------------------------------------------------------
        elif "Put Ratio Backspread" in strat_name or ("Put" in strat_name and "Backspread" in strat_name):
            short_pe = next((l for l in legs if l["action"] == "SELL" and l["type"] == "PE"), None)
            long_pe = next((l for l in legs if l["action"] == "BUY" and l["type"] == "PE"), None)
            if short_pe and long_pe:
                k1 = short_pe["strike"]
                k2 = long_pe["strike"]
                has_hedge = any(l["type"] == "CE" for l in legs)
                # Valley of Death: Spot stalls between K2 and K1 without downward acceleration
                if k2 <= spot_price <= k1 and not has_hedge and short_pe.get("state") != "ADJUSTED_VALLEY_FINANCED":
                    # Sell OTM Call Credit Spread (K1 + 100 / K1 + 150) to finance valley drag
                    hedge_short_k = k1 + 100
                    hedge_long_k = k1 + 150
                    px_short = self._get_option_quote(hedge_short_k, "CE", spot_price)
                    px_long = self._get_option_quote(hedge_long_k, "CE", spot_price)

                    missing = []
                    if px_short is None or px_short <= 0.0:
                        missing.append(f"{hedge_short_k} CE (short hedge)")
                    if px_long is None or px_long <= 0.0:
                        missing.append(f"{hedge_long_k} CE (long hedge)")

                    if missing:
                        defer_note = f"Adjustment deferred [{horizon}]: Missing authentic market quote for {', '.join(missing)}. Cannot execute on theoretical BSM prices."
                        logger.warning(f"⚠️ [{strategy_id}] {defer_note}")
                        return False, legs, defer_note

                    g_hs = calculate_bsm_greeks(spot_price, hedge_short_k, 1.0 / 365.0, 0.07, 0.135, "CE")
                    g_hl = calculate_bsm_greeks(spot_price, hedge_long_k, 1.0 / 365.0, 0.07, 0.135, "CE")

                    strat_ls = next((int(l["lot_size"]) for l in legs if l.get("lot_size") and int(l["lot_size"]) > 0), None)
                    if not strat_ls and legs:
                        for l in legs:
                            if l.get("qty") and int(l["qty"]) > 0:
                                q = int(l["qty"])
                                num_l = int(l.get("lots") or 1)
                                if num_l > 0 and q % num_l == 0:
                                    cand_ls = q // num_l
                                    if cand_ls in (25, 50, 65, 75):
                                        strat_ls = cand_ls
                                        break
                    if not strat_ls:
                        exp = next((l.get("expiry_date") for l in legs if l.get("expiry_date")), None)
                        if exp:
                            try:
                                strat_ls = resolve_nifty_lot_size(
                                    expiry_date=exp,
                                    contract_type=horizon,
                                    symbol="NIFTY",
                                    strict=True
                                )
                            except Exception as ex:
                                defer_note = f"Adjustment deferred [{horizon}]: Cannot resolve authoritative lot size for hedge legs: {ex}"
                                logger.error(f"🛑 [{strategy_id}] {defer_note}")
                                return False, legs, defer_note
                        else:
                            strat_ls = resolve_nifty_lot_size(strict=False)

                    credit_got = round((px_short - px_long) * strat_ls, 2)

                    legs.append({
                        "action": "SELL",
                        "type": "CE",
                        "strike": hedge_short_k,
                        "qty": strat_ls,
                        "lots": 1,
                        "lot_size": strat_ls,
                        "ratio": 1,
                        "entry_price": round(px_short, 2),
                        "price": round(px_short, 2),
                        "premium": round(px_short, 2),
                        "current_price": round(px_short, 2),
                        "quote_source": "LIVE_MARKET_QUOTE",
                        "greeks": g_hs,
                        "state": "HEDGE_LEG"
                    })
                    legs.append({
                        "action": "BUY",
                        "type": "CE",
                        "strike": hedge_long_k,
                        "qty": strat_ls,
                        "lots": 1,
                        "lot_size": strat_ls,
                        "ratio": 1,
                        "entry_price": round(px_long, 2),
                        "price": round(px_long, 2),
                        "premium": round(px_long, 2),
                        "current_price": round(px_long, 2),
                        "quote_source": "LIVE_MARKET_QUOTE",
                        "greeks": g_hl,
                        "state": "HEDGE_LEG"
                    })
                    short_pe["state"] = "ADJUSTED_VALLEY_FINANCED"

                    adj_note = f"Auto-Adjustment [{horizon}]: Deployed Call Credit Spread ({hedge_short_k}/{hedge_long_k} CE, Credit: ₹{credit_got:.0f}) to finance Valley of Death stall at {k2}-{k1}."
                    logger.info(f"⚙️ [{strategy_id}] {adj_note}")
                    self._record_adjustment(strategy_id, "PUT_RATIO_VALLEY_FINANCED", adj_note, spot_price, old_legs, legs, credit_collected=credit_got)
                    return True, legs, adj_note

        return False, legs, ""

    def _record_adjustment(
        self,
        strat_id: str,
        adj_type: str,
        reason: str,
        spot: float,
        old_legs: List[Dict[str, Any]],
        new_legs: Optional[List[Dict[str, Any]]] = None,
        credit_collected: float = 0.0
    ):
        """Records an adjustment event in SQLite audit log."""
        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("""
                    INSERT INTO options_strategy_adjustments_log (
                        strategy_id, adjustment_type, trigger_reason, spot_at_adjustment,
                        old_legs_json, new_legs_json, credit_collected, timestamp_ist
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strat_id,
                    adj_type,
                    reason,
                    spot,
                    json.dumps(old_legs),
                    json.dumps(new_legs if new_legs is not None else old_legs),
                    credit_collected,
                    format_ist()
                ))
                conn.commit()

    def _execute_exit(self, strategy_id: str, exit_reason: str, exit_note: str, spot_exit: float, realized_pnl: float):
        """Formally closes a strategy in database and dispatches outcome to ML training dataset."""
        now_str = format_ist()
        logger.info(f"🛑 [WATCHDOG EXIT: {strategy_id}] Reason: {exit_reason} | Realized P&L: ₹{realized_pnl:+.2f}")

        with state_engine._lock:
            with state_engine._get_connection() as conn:
                conn.execute("""
                    UPDATE options_strategies
                    SET status = ?,
                        closed_at = ?,
                        spot_at_exit = ?,
                        realized_pnl = ?,
                        exit_reason = ?,
                        exit_note = ?
                    WHERE strategy_id = ?
                """, (exit_reason, now_str, spot_exit, realized_pnl, exit_reason, exit_note, strategy_id))
                conn.commit()

        # Closed-Loop ML Learning Dispatch
        try:
            from core.unified_ml_dataset import unified_ml_dataset
            unified_ml_dataset.record_trade_outcome(
                trade_id=strategy_id,
                gross_pnl=realized_pnl,
                net_pnl=realized_pnl - 85.0, # Net of regulatory transaction costs
                exit_price=spot_exit,
                exit_reason=exit_reason,
                mae_inr=min(0.0, realized_pnl),
                mfe_inr=max(0.0, realized_pnl)
            )
        except Exception as e:
            logger.debug(f"ML outcome recording notice: {e}")


# Global Singleton Instance
strategy_adjustment_engine = StrategyAdjustmentEngine()
