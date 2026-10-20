import sqlite3
import sys
from pathlib import Path

def reset_p3(db_file="data/swing_trades.db"):
    db_path = Path(db_file)
    if not db_path.exists():
        print(f"Error: {db_path} does not exist")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check tables
    tables = [t[0] for t in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    # 1. Trade history
    trades = cursor.execute("SELECT id, symbol, portfolio_type, pnl, exit_date, exit_reason FROM trade_history WHERE portfolio_type LIKE '%PORTFOLIO_3%' OR portfolio_type LIKE '%BREAKOUT%'").fetchall()
    print(f"Found {len(trades)} P3 trade(s) in trade_history:")
    for t in trades:
        print("  ", t)
        
    # 2. Active positions
    active = cursor.execute("SELECT id, symbol, portfolio_type, quantity, entry_price FROM active_positions WHERE portfolio_type LIKE '%PORTFOLIO_3%' OR portfolio_type LIKE '%BREAKOUT%'").fetchall()
    print(f"Found {len(active)} P3 active position(s):")
    for a in active:
        print("  ", a)
        
    # 3. Position legs
    if "p3_position_legs" in tables:
        legs = cursor.execute("SELECT id, symbol, overall_status FROM p3_position_legs").fetchall()
        print(f"Found {len(legs)} P3 legs:")
        for l in legs:
            print("  ", l)
            
    # Execute deletion
    cursor.execute("DELETE FROM trade_history WHERE portfolio_type LIKE '%PORTFOLIO_3%' OR portfolio_type LIKE '%BREAKOUT%'")
    cursor.execute("DELETE FROM active_positions WHERE portfolio_type LIKE '%PORTFOLIO_3%' OR portfolio_type LIKE '%BREAKOUT%'")
    if "p3_position_legs" in tables:
        cursor.execute("DELETE FROM p3_position_legs")
    
    conn.commit()
    conn.close()
    print("SUCCESS: Portfolio 3 database records reset cleanly.")
    return True

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/swing_trades.db"
    reset_p3(db)
