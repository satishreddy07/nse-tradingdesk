import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Get rich features from SHADOW entry (id 708) or latest_signals
shadow = cur.execute("SELECT * FROM quant_trade_dataset WHERE symbol = 'INDHOTEL' AND trade_type = 'SHADOW' ORDER BY id DESC LIMIT 1").fetchone()
real = cur.execute("SELECT * FROM quant_trade_dataset WHERE symbol = 'INDHOTEL' AND trade_type = 'REAL' ORDER BY id DESC LIMIT 1").fetchone()

if shadow and real:
    print(f"Updating REAL trade id {real['id']} with technical data from SHADOW id {shadow['id']}...")
    cur.execute("""
        UPDATE quant_trade_dataset
        SET score = ?, confidence = ?, rsi14 = ?, atr_pct = ?, dist_to_pivot = ?, dist_to_ema20 = ?,
            vol_ratio = ?, inst_score = ?, week_confluence = ?, is_vcp = ?, bb_sq_ratio = ?, adx14 = ?,
            features_json = ?, nifty_trend = ?, nifty_ret_5d = ?, nifty_rsi14 = ?, vix_val = ?,
            sector_excess_6m = ?, market_regime_json = ?, ml_win_prob = ?, ml_score = ?, expected_r = ?,
            ml_decision = ?, ai_decision = ?, ai_reason_code = ?, ai_summary = ?, quantity = ?
        WHERE id = ?
    """, (
        shadow["score"], shadow["confidence"], shadow["rsi14"], shadow["atr_pct"], shadow["dist_to_pivot"], shadow["dist_to_ema20"],
        shadow["vol_ratio"], shadow["inst_score"], shadow["week_confluence"], shadow["is_vcp"], shadow["bb_sq_ratio"], shadow["adx14"],
        shadow["features_json"], shadow["nifty_trend"], shadow["nifty_ret_5d"], shadow["nifty_rsi14"], shadow["vix_val"],
        shadow["sector_excess_6m"], shadow["market_regime_json"], shadow["ml_win_prob"], shadow["ml_score"], shadow["expected_r"],
        shadow["ml_decision"], shadow["ai_decision"], shadow["ai_reason_code"], shadow["ai_summary"], 132,
        real["id"]
    ))
    
    # Also update active_positions metrics_json for INDHOTEL
    cur.execute("""
        UPDATE active_positions
        SET metrics_json = ?
        WHERE symbol = 'INDHOTEL'
    """, (shadow["features_json"],))
    
    conn.commit()
    print("Update successful!")

print("\n=== VERIFIED REAL INDHOTEL IN quant_trade_dataset ===")
r = cur.execute("SELECT id, symbol, trade_type, score, confidence, rsi14, dist_to_pivot, dist_to_ema20, vol_ratio, inst_score, ml_win_prob, expected_r, ml_decision, ai_decision, ai_reason_code FROM quant_trade_dataset WHERE id = ?", (real["id"],)).fetchone()
print(dict(r))

conn.close()
