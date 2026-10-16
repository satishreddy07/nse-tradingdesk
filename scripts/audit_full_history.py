import sqlite3
import glob

db_files = sorted(glob.glob('/var/backups/nse-tradingdesk/*.db')) + ['/var/lib/nse-tradingdesk/swing_trades.db']
print(f"Found {len(db_files)} databases.")

for f in db_files:
    try:
        conn = sqlite3.connect(f)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Check tables
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if 'trade_history' not in tables or 'active_positions' not in tables:
            continue
            
        act = cur.execute("SELECT symbol, quantity, entry_price, t1_hit, t1_pnl FROM active_positions").fetchall()
        hist = cur.execute("SELECT id, symbol, quantity, entry_price, exit_price, pnl, exit_reason, exit_date FROM trade_history").fetchall()
        
        if len(hist) > 0 or len(act) > 0:
            print(f"\nDB: {f}")
            print(f"  Active ({len(act)}): {[ (r['symbol'], r['quantity'], r['t1_hit'], r['t1_pnl']) for r in act ]}")
            print(f"  Closed History ({len(hist)}):")
            for h in hist:
                print(f"    id={h['id']} {h['symbol']} qty={h['quantity']} entry={h['entry_price']} exit={h['exit_price']} pnl={h['pnl']} reason={h['exit_reason']} date={h['exit_date']}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
