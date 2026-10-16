import sqlite3
import json

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== 1. SCAN LOGS AROUND 10:20 - 10:30 IST TODAY ===")
for r in cur.execute("""
    SELECT id, timestamp, scan_type, status, execution_time_sec, candidates_count, orders_count, reconciled_count, message, details_json
    FROM scan_logs
    WHERE timestamp >= '2026-10-16 10:15:00'
    ORDER BY id ASC
"""):
    print(f"[{r['id']}] {r['timestamp']} | {r['scan_type']} | {r['status']}")
    print(f"   Msg: {r['message']}")
    if r['details_json']:
        try:
            d = json.loads(r['details_json'])
            print(f"   Details: {d}")
        except Exception:
            print(f"   Details raw: {r['details_json'][:200]}")

print("\n=== 2. SWING DECISION AUDIT RECORDS TODAY ===")
for r in cur.execute("""
    SELECT id, timestamp, candidate_id, symbol, scanner_source, technical_classification,
           ml_decision, ml_win_prob, ml_score, expected_r, ai_status, ai_decision, ai_reason_code, ai_summary,
           risk_check_status, final_status, trade_id
    FROM swing_decision_audit
    WHERE timestamp >= '2026-10-16 10:00:00'
    ORDER BY id ASC
"""):
    print(dict(r))

print("\n=== 3. QUANT TRADE DATASET FOR RECENT TRADES ===")
for r in cur.execute("""
    SELECT id, symbol, portfolio_type, trade_type, status, entry_price, stop_loss, target, target_2,
           score, confidence, rsi14, dist_to_pivot, dist_to_ema20, vol_ratio, ml_win_prob, ml_decision,
           ai_decision, ai_reason_code, expected_r, created_at, notes
    FROM quant_trade_dataset
    WHERE created_at >= '2026-10-16 09:00:00' OR symbol = 'INDHOTEL'
    ORDER BY id DESC
"""):
    print(dict(r))

print("\n=== 4. ACTIVE POSITION DETAILS FOR INDHOTEL ===")
for r in cur.execute("SELECT * FROM active_positions WHERE symbol = 'INDHOTEL'"):
    for k in r.keys():
        print(f"  {k}: {r[k]}")

print("\n=== 5. KV STORE / SYSTEM CONFIG RELATED TO AUTO TRADING ===")
for r in cur.execute("SELECT key, value FROM kv_store WHERE key LIKE '%auto%' OR key LIKE '%scan%' OR key LIKE '%config%' OR key LIKE '%pillar%'"):
    print(f"  {r['key']}: {r['value']}")

conn.close()
