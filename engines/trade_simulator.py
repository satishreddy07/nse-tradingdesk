"""
========================================================================================
Independent Trade Execution Simulator Engine (`engines/trade_simulator.py`)
========================================================================================
Deterministic, Zero-Mock Execution Simulator for Pre-Rally & Swing Strategies:
- Strict separation of Statistical Outcome Labels from Executable Strategy Returns.
- Realistic Bar-by-Bar Path Execution:
    * Signal evaluated at session T-1 or T cutoff; realistic entry execution.
    * Realized fills on gaps: gap-down below stop fills at Open (loss expansion);
      gap-up above target fills at Open (gain expansion).
    * Conservative same-bar conflict resolution (stop triggered first if ambiguous).
    * Time-based expiry exit at session H Close.
    * Diagnostic MFE/MAE measurement strictly isolated from realized P&L.
- Configurable Indian Equities Statutory Cost Model:
    * Brokerage, STT (0.1% buy + 0.1% sell), Exchange Turnover Fee, GST (18%),
      Stamp Duty (0.015% buy), SEBI charges, and configurable slippage.
    * Also supports baseline flat percentage models (e.g. 0.35% roundtrip).
- Complete Trade-Level Accounting with immutable configuration hashing.
========================================================================================
"""

import math
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("TradeSimulator")

SIMULATOR_VERSION = "2.1.0-AUDITED"

# ==============================================================================
# CONFIGURABLE TRANSACTION COST MODELS
# ==============================================================================

class TransactionCostModel:
    """
    Computes exact execution frictions and statutory taxes for NSE cash equities.
    Separates Buy-side charges from Sell-side charges.
    """
    def __init__(
        self,
        cost_type: str = "STATUTORY_NSE_CASH",
        flat_roundtrip_pct: float = 0.35,
        brokerage_pct: float = 0.0,       # 0.0% for discount delivery; or 0.02%
        brokerage_flat_cap: float = 20.0,  # Max ₹20 per executed order
        stt_pct_buy: float = 0.10,         # 0.10% delivery STT on buy
        stt_pct_sell: float = 0.10,        # 0.10% delivery STT on sell
        exchange_turnover_pct: float = 0.00297,  # NSE cash turnover charges
        gst_pct_on_charges: float = 18.0,  # 18% GST on (Brokerage + Exchange)
        sebi_charges_pct: float = 0.0001,  # ₹10 per crore = 0.0001%
        stamp_duty_pct_buy: float = 0.015, # 0.015% on buy turnover
        slippage_pct_per_leg: float = 0.05 # 0.05% slippage on entry & exit
    ):
        self.cost_type = cost_type
        self.flat_roundtrip_pct = flat_roundtrip_pct
        self.brokerage_pct = brokerage_pct
        self.brokerage_flat_cap = brokerage_flat_cap
        self.stt_pct_buy = stt_pct_buy
        self.stt_pct_sell = stt_pct_sell
        self.exchange_turnover_pct = exchange_turnover_pct
        self.gst_pct_on_charges = gst_pct_on_charges
        self.sebi_charges_pct = sebi_charges_pct
        self.stamp_duty_pct_buy = stamp_duty_pct_buy
        self.slippage_pct_per_leg = slippage_pct_per_leg

    def calculate_costs(
        self,
        entry_price: float,
        exit_price: float,
        shares: int = 100
    ) -> Dict[str, Any]:
        """Calculates itemized and percentage costs for an executed trade."""
        if entry_price <= 0 or exit_price <= 0 or shares <= 0:
            return {
                "total_charges_pct": 0.0,
                "total_charges_inr": 0.0,
                "breakdown": {}
            }

        buy_turnover = entry_price * shares
        sell_turnover = exit_price * shares
        total_turnover = buy_turnover + sell_turnover

        if self.cost_type == "FLAT_PERCENT":
            charges_pct = self.flat_roundtrip_pct
            charges_inr = buy_turnover * (charges_pct / 100.0)
            return {
                "total_charges_pct": round(charges_pct, 4),
                "total_charges_inr": round(charges_inr, 2),
                "breakdown": {"flat_roundtrip_pct": charges_pct}
            }

        # Statutory Itemized Cost Schedule (NSE Cash Delivery)
        # 1. Brokerage
        brok_buy = min(self.brokerage_flat_cap, buy_turnover * (self.brokerage_pct / 100.0))
        brok_sell = min(self.brokerage_flat_cap, sell_turnover * (self.brokerage_pct / 100.0))
        total_brokerage = brok_buy + brok_sell

        # 2. STT (Securities Transaction Tax) - 0.1% on buy and 0.1% on sell for equity delivery
        stt_buy = buy_turnover * (self.stt_pct_buy / 100.0)
        stt_sell = sell_turnover * (self.stt_pct_sell / 100.0)
        total_stt = stt_buy + stt_sell

        # 3. Exchange Turnover Charges (NSE 0.00297%)
        exch_charges = total_turnover * (self.exchange_turnover_pct / 100.0)

        # 4. GST (18% on Brokerage + Exchange charges)
        gst = (total_brokerage + exch_charges) * (self.gst_pct_on_charges / 100.0)

        # 5. SEBI Turnover Charges (0.0001%)
        sebi = total_turnover * (self.sebi_charges_pct / 100.0)

        # 6. Stamp Duty (0.015% on Buy side only)
        stamp_duty = buy_turnover * (self.stamp_duty_pct_buy / 100.0)

        # 7. Slippage & Spread Friction
        slippage_inr = (buy_turnover * (self.slippage_pct_per_leg / 100.0)) + (sell_turnover * (self.slippage_pct_per_leg / 100.0))

        total_charges_inr = total_brokerage + total_stt + exch_charges + gst + sebi + stamp_duty + slippage_inr
        total_charges_pct = (total_charges_inr / buy_turnover) * 100.0

        return {
            "total_charges_pct": round(total_charges_pct, 4),
            "total_charges_inr": round(total_charges_inr, 2),
            "breakdown": {
                "brokerage_inr": round(total_brokerage, 2),
                "stt_inr": round(total_stt, 2),
                "exchange_charges_inr": round(exch_charges, 2),
                "gst_inr": round(gst, 2),
                "stamp_duty_inr": round(stamp_duty, 2),
                "sebi_inr": round(sebi, 2),
                "slippage_inr": round(slippage_inr, 2)
            }
        }


# ==============================================================================
# TRADE SIMULATION ENGINE
# ==============================================================================

class TradeSimulator:
    """
    Independent, deterministic trade simulator enforcing:
    - Explicit execution models: NEXT_OPEN vs SAME_CLOSE.
    - True gap execution past stops or targets.
    - Conservative same-candle conflict resolution.
    - Day H close time stop.
    - Rigorous itemized friction accounting.
    - Zero future candle snooping.
    """

    def __init__(
        self,
        target_pct: float = 5.0,
        stop_pct: float = -3.5,
        horizon_days: int = 10,
        entry_model: str = "SAME_CLOSE", # 'SAME_CLOSE' or 'NEXT_OPEN'
        cost_model: Optional[TransactionCostModel] = None
    ):
        self.target_pct = float(target_pct)
        self.stop_pct = float(stop_pct)
        self.horizon_days = int(horizon_days)
        self.entry_model = entry_model.upper()
        self.cost_model = cost_model or TransactionCostModel(cost_type="STATUTORY_NSE_CASH")

    def get_config_hash(self) -> str:
        """Returns deterministic MD5 hash of simulator parameters."""
        cfg = {
            "version": SIMULATOR_VERSION,
            "target_pct": self.target_pct,
            "stop_pct": self.stop_pct,
            "horizon_days": self.horizon_days,
            "entry_model": self.entry_model,
            "cost_type": self.cost_model.cost_type,
            "flat_cost": self.cost_model.flat_roundtrip_pct,
            "stt_buy": self.cost_model.stt_pct_buy,
            "stt_sell": self.cost_model.stt_pct_sell
        }
        return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

    def simulate_trade(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        symbol: str = "UNKNOWN",
        run_id: str = "AUDIT_RUN"
    ) -> Dict[str, Any]:
        """
        Executes deterministic simulation on a single candidate signal.
        - signal_idx: row index in df where the pre-rally condition was detected (e.g. at session T-1 or T).
        """
        if df is None or len(df) <= signal_idx:
            return {"status": "ERROR", "error_reason": "INSUFFICIENT_DATA"}

        n_bars = len(df)

        # 1. Determine Entry Bar and Entry Price
        if self.entry_model == "NEXT_OPEN":
            entry_idx = signal_idx + 1
            if entry_idx >= n_bars:
                return {"status": "PENDING", "error_reason": "AWAITING_NEXT_OPEN_CANDLE"}
            entry_date = str(df.index[entry_idx])[:10]
            entry_price = float(df["Open"].iloc[entry_idx])
            start_eval_idx = signal_idx
        else: # "SAME_CLOSE"
            entry_idx = signal_idx
            entry_date = str(df.index[entry_idx])[:10]
            entry_price = float(df["Close"].iloc[entry_idx])
            start_eval_idx = signal_idx

        if entry_price <= 0.0 or np.isnan(entry_price):
            return {"status": "ERROR", "error_reason": "INVALID_ENTRY_PRICE"}

        # 2. Check for Corporate Action / Abnormal Price Discontinuity (>40% unadjusted jump)
        if entry_idx > 0:
            prev_close = float(df["Close"].iloc[entry_idx - 1])
            if prev_close > 0:
                day_jump = abs(entry_price - prev_close) / prev_close
                if day_jump >= 0.40:
                    return {
                        "status": "EXCLUDED",
                        "error_reason": "CORP_ACTION_DISCONTINUITY",
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "jump_pct": round(day_jump * 100.0, 1)
                    }

        signal_date = str(df.index[signal_idx])[:10]
        target_price = round(entry_price * (1.0 + self.target_pct / 100.0), 2)
        stop_price = round(entry_price * (1.0 + self.stop_pct / 100.0), 2)

        # 3. Simulate Forward Window Bar-by-Bar
        exit_price = None
        exit_date = None
        exit_reason = None
        hold_days = 0

        # Diagnostics: track maximum favorable and adverse excursions
        peak_high = entry_price
        trough_low = entry_price

        # Evaluation runs from start_eval_idx + 1 up to start_eval_idx + horizon_days
        for k in range(1, self.horizon_days + 1):
            curr_idx = start_eval_idx + k
            if curr_idx >= n_bars:
                # Trade is still active/open if forward horizon has not yet elapsed in stored data
                break

            hold_days = k
            bar_date = str(df.index[curr_idx])[:10]
            bar_open = float(df["Open"].iloc[curr_idx])
            bar_high = float(df["High"].iloc[curr_idx])
            bar_low = float(df["Low"].iloc[curr_idx])
            bar_close = float(df["Close"].iloc[curr_idx])

            # Integrity check on candle
            if (
                math.isnan(bar_open) or math.isnan(bar_high) or math.isnan(bar_low) or math.isnan(bar_close)
                or bar_open <= 0 or bar_high <= 0 or bar_low <= 0 or bar_close <= 0 or bar_low > bar_high
            ):
                return {"status": "ERROR", "error_reason": "CORRUPT_CANDLE_DATA", "date": bar_date}

            # Check for Corporate Action / Abnormal Price Discontinuity during holding (>40% unadjusted jump)
            prev_bar_close = float(df["Close"].iloc[curr_idx - 1])
            if prev_bar_close > 0:
                jump = abs(bar_open - prev_bar_close) / prev_bar_close
                if jump >= 0.40:
                    return {
                        "status": "EXCLUDED",
                        "error_reason": "CORP_ACTION_DISCONTINUITY",
                        "symbol": symbol,
                        "date": bar_date,
                        "jump_pct": round(jump * 100.0, 1)
                    }

            if bar_high > peak_high:
                peak_high = bar_high
            if bar_low < trough_low:
                trough_low = bar_low

            # A. Gap-Down Check at Open (Opens below stop loss on subsequent session)
            if curr_idx > entry_idx and bar_open <= stop_price:
                exit_price = bar_open  # Fills at gap-down open price (realistic slippage!)
                exit_date = bar_date
                exit_reason = "STOP_GAP"
                break

            # B. Gap-Up Check at Open (Opens above target on subsequent session)
            if curr_idx > entry_idx and bar_open >= target_price:
                exit_price = bar_open  # Fills at gap-up open price (realistic gain expansion!)
                exit_date = bar_date
                exit_reason = "TARGET_GAP"
                break

            # C. Intraday Threshold Boundary Checks
            hit_target = (bar_high >= target_price)
            hit_stop = (bar_low <= stop_price)

            # D. Ambiguous Same-Bar Conflict Resolution
            if hit_target and hit_stop:
                # Both breached on same day: conservative institutional risk protocol assumes STOP hit first
                exit_price = stop_price
                exit_date = bar_date
                exit_reason = "STOP_CONFLICT"
                break
            elif hit_stop:
                exit_price = stop_price
                exit_date = bar_date
                exit_reason = "STOP_LOSS"
                break
            elif hit_target:
                exit_price = target_price
                exit_date = bar_date
                exit_reason = "PROFIT_TARGET"
                break

        # 4. Handle Time Expiry Exit or Unresolved Status
        if exit_price is None:
            max_idx = start_eval_idx + self.horizon_days
            if max_idx < n_bars:
                # Trade concluded horizon without hitting target or stop -> Exit at Horizon Close
                exit_price = float(df["Close"].iloc[max_idx])
                exit_date = str(df.index[max_idx])[:10]
                exit_reason = "TIME_EXPIRY"
                hold_days = self.horizon_days
            else:
                # Trade is still in-progress (active open trade)
                last_idx = n_bars - 1
                curr_price = float(df["Close"].iloc[last_idx])
                unrealized_gross = ((curr_price - entry_price) / entry_price) * 100.0
                return {
                    "status": "OPEN",
                    "run_id": run_id,
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "entry_price": round(entry_price, 2),
                    "entry_model": self.entry_model,
                    "target_price": target_price,
                    "stop_price": stop_price,
                    "current_price": round(curr_price, 2),
                    "hold_days": n_bars - 1 - start_eval_idx,
                    "unrealized_return_pct": round(unrealized_gross, 2),
                    "mfe_pct": round(((peak_high - entry_price) / entry_price) * 100.0, 2),
                    "mae_pct": round(((trough_low - entry_price) / entry_price) * 100.0, 2),
                    "config_hash": self.get_config_hash()
                }

        # 5. Financial P&L & Friction Accounting
        gross_return_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
        costs = self.cost_model.calculate_costs(entry_price, exit_price, shares=100)
        total_charges_pct = costs["total_charges_pct"]
        net_return_pct = round(gross_return_pct - total_charges_pct, 2)

        # Statistical label check (+5% reached before MAE -3.5%)
        label_winner = 1 if (peak_high >= target_price and ((trough_low - entry_price) / entry_price * 100.0) >= self.stop_pct) else 0

        trade_record = {
            "status": "COMPLETED",
            "run_id": run_id,
            "symbol": symbol,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 2),
            "entry_model": self.entry_model,
            "target_price": target_price,
            "stop_price": stop_price,
            "exit_date": exit_date,
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "hold_days": hold_days,
            "gross_return_pct": gross_return_pct,
            "charges_pct": total_charges_pct,
            "net_return_pct": net_return_pct,
            "charges_breakdown": costs["breakdown"],
            "mfe_pct": round(((peak_high - entry_price) / entry_price) * 100.0, 2),
            "mae_pct": round(((trough_low - entry_price) / entry_price) * 100.0, 2),
            "is_win": 1 if net_return_pct > 0 and exit_reason in ("PROFIT_TARGET", "TARGET_GAP") else 0,
            "original_label": label_winner,
            "simulator_version": SIMULATOR_VERSION,
            "config_hash": self.get_config_hash()
        }

        return trade_record
