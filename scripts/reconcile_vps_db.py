import sqlite3
import shutil
import datetime

db_path = "/var/lib/nse-tradingdesk/swing_trades.db"
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/var/backups/nse-tradingdesk/swing_trades_audit_reconcile_{ts}.db"

print(f"Creating backup: {backup_path}")
shutil.copy2(db_path, backup_path)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 1. Add column original_quantity if missing
cur.execute("PRAGMA table_info(active_positions)")
columns = [row["name"] for row in cur.fetchall()]
if "original_quantity" not in columns:
    print("Adding original_quantity column to active_positions...")
    cur.execute("ALTER TABLE active_positions ADD COLUMN original_quantity INTEGER")
else:
    print("original_quantity column already exists.")

# 2. Reconcile ADVANCE quantity and original_quantity
cur.execute("SELECT quantity, original_quantity, t1_hit, t1_pnl FROM active_positions WHERE symbol = 'ADVANCE'")
adv = cur.fetchone()
if adv:
    print(f"Current ADVANCE in DB: Qty={adv['quantity']}, OrigQty={adv['original_quantity']}, T1_Hit={adv['t1_hit']}, T1_PnL={adv['t1_pnl']}")
    cur.execute("UPDATE active_positions SET quantity = 168, original_quantity = 335 WHERE symbol = 'ADVANCE'")
    print("Updated ADVANCE to Qty=168, OrigQty=335.")

# 3. Reconcile UTTAMSUGAR original_quantity
cur.execute("SELECT quantity, original_quantity, t1_hit, t1_pnl FROM active_positions WHERE symbol = 'UTTAMSUGAR'")
uttam = cur.fetchone()
if uttam:
    print(f"Current UTTAMSUGAR in DB: Qty={uttam['quantity']}, OrigQty={uttam['original_quantity']}, T1_Hit={uttam['t1_hit']}, T1_PnL={uttam['t1_pnl']}")
    cur.execute("UPDATE active_positions SET original_quantity = 183 WHERE symbol = 'UTTAMSUGAR'")
    print("Updated UTTAMSUGAR to OrigQty=183.")

# 4. Fill original_quantity for all other positions
cur.execute("UPDATE active_positions SET original_quantity = quantity WHERE original_quantity IS NULL")

conn.commit()

print("\n=== RECONCILED ACTIVE POSITIONS ===")
for r in cur.execute("SELECT symbol, quantity, original_quantity, entry_price, current_price, stop_loss, target, t1_hit, t1_pnl FROM active_positions"):
    print(dict(r))

print("\n=== TRADE HISTORY ===")
for r in cur.execute("SELECT id, symbol, quantity, entry_price, exit_price, pnl, r_multiple, return_pct, exit_reason, exit_date FROM trade_history"):
    print(dict(r))

conn.close()
print("\nReconciliation complete.")
