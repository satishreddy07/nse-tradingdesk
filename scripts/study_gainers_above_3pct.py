import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import sqlite3
import pandas as pd
import numpy as np
from engines.custom_screener_engine import custom_screener_engine

print("Fetching Chartink EQ-FINAL universe...")
scan_res = custom_screener_engine.run_scan(preset_name="EQ-FINAL", force_refresh=False)
stocks = scan_res.get("results", [])
print(f"Total screened stocks: {len(stocks)}")

# Filter stocks with per_chg >= 3.0%
gainers_3pct = [s for s in stocks if float(s.get("per_chg", 0.0) or 0.0) >= 3.0]
print(f"Total stocks with Friday gain >= +3.0%: {len(gainers_3pct)}")

# Sort descending by per_chg
gainers_3pct.sort(key=lambda s: float(s.get("per_chg", 0.0) or 0.0), reverse=True)

symbols = [s["symbol"] for s in gainers_3pct]
print("Symbols:", symbols)
