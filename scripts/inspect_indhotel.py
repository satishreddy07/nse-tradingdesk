import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== 1. active_positions for INDHOTEL ===")
rows = conn.execute("SELECT * FROM active_positions WHERE symbol LIKE '%INDHOTEL%'").fetchall()
for r in rows:
    d = dict(r)
    print(f"  Symbol: {d.get('symbol')} | Portfolio: {d.get('portfolio_type')} | Entry: {d.get('entry_price')} | Current: {d.get('current_price')} | SL: {d.get('stop_loss')} | T1: {d.get('target_1')} | T1_hit: {d.get('t1_hit')} | Entry Time: {d.get('entry_date')} | Strategy: {d.get('strategy_name')}")

print("\n=== 2. quant_trade_dataset for INDHOTEL ===")
try:
    rows = conn.execute("SELECT * FROM quant_trade_dataset WHERE symbol LIKE '%INDHOTEL%'").fetchall()
    print(f"Found {len(rows)} rows in quant_trade_dataset:")
    for r in rows:
        d = dict(r)
        print(f"  ID: {d.get('id')} | RecordType: {d.get('record_type')} | Decision: {d.get('decision')} | ML Decision: {d.get('ml_decision')} | AI Verdict: {d.get('ai_verdict')} | Score: {d.get('score')} | WinProb: {d.get('ml_win_prob')} | Outcome: {d.get('actual_outcome')} | Realized R: {d.get('actual_r_multiple')} | Created: {d.get('created_at')}")
        if d.get('ai_reasoning'):
            print(f"    AI Reasoning: {d.get('ai_reasoning')[:200]}...")
except Exception as e:
    print(f"Error querying quant_trade_dataset: {e}")

print("\n=== 3. unified_trade_learning_dataset for INDHOTEL ===")
try:
    rows = conn.execute("SELECT * FROM unified_trade_learning_dataset WHERE instrument LIKE '%INDHOTEL%'").fetchall()
    print(f"Found {len(rows)} rows in unified_trade_learning_dataset:")
    for r in rows:
        d = dict(r)
        print(f"  ID: {d.get('id')} | Type: {d.get('record_type')} | Horizon: {d.get('trading_horizon')} | Gate: {d.get('decision_gate')} | Status: {d.get('status')} | Outcome: {d.get('realized_outcome')} | R: {d.get('realized_r')} | Timestamp: {d.get('timestamp')}")
except Exception as e:
    print(f"Error querying unified_trade_learning_dataset: {e}")

print("\n=== 4. scan_signals / trade_signals / orders for INDHOTEL ===")
for tbl in ["trade_signals", "scan_history", "manual_orders", "ai_trade_reasoning"]:
    try:
        r_cnt = conn.execute(f"SELECT count(*) FROM {tbl} WHERE symbol LIKE '%INDHOTEL%'").fetchone()[0]
        print(f"  {tbl}: {r_cnt} records")
        if r_cnt > 0:
            sample = conn.execute(f"SELECT * FROM {tbl} WHERE symbol LIKE '%INDHOTEL%' ORDER BY id DESC LIMIT 2").fetchall()
            for s in sample:
                print(f"    {dict(s)}")
    except Exception as e:
        pass

conn.close()
