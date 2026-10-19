import sqlite3
import os

db_path = os.environ.get("TRADINGDESK_DB_PATH", "data/swing_trades.db")
if not os.path.exists(db_path) and os.path.exists("/opt/nse-tradingdesk/data/swing_trades.db"):
    db_path = "/opt/nse-tradingdesk/data/swing_trades.db"

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("""
UPDATE nifty_predictions_log
SET spot_at_prediction = 23146.15,
    composite_score = 0.7,
    t1_verdict = 'RANGE-BOUND / CONSOLIDATION',
    t1_expected_close = 23150.0,
    t1_low_1s = 22950.0,
    t1_high_1s = 23350.0,
    t5_verdict = 'MODERATELY BEARISH',
    actual_t1_close = 22888.95,
    t1_actual_return_pct = -1.11,
    t1_error_pts = -261.05,
    t1_validated = 'DIRECTION_MISS',
    t1_attribution = 'WRONG (MISS): Model projected RANGE-BOUND / CONSOLIDATION (+0.7 score, target ₹23,150), but Nifty moved -1.11% (-257.2 pts) to ₹22,888.95 at EOD. Error vs expected: -261.1 pts.'
WHERE prediction_date = '2026-10-19'
""")
conn.commit()
print(f"Fixed {c.rowcount} row in {db_path} for 2026-10-19.")
