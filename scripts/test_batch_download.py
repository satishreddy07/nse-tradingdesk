import yfinance as yf
import pandas as pd
import numpy as np
import json
import subprocess

symbols = ['ONWARDTEC', 'HBLENGINE', 'EUROPRATIK', 'TIPSMUSIC', 'TECHM', 'SHANTIGOLD', 'MVGJL', 'SHRIPISTON', 'PNGSREVA', 'MBEL', 'THANGAMAYL', 'SMSPHARMA', 'WIPRO', 'SKYGOLD', 'RPGLIFE', 'AJMERA', 'SAMBHV', 'AIIL', 'RSL', 'SHRINGARMS', 'KPITTECH', 'LLOYDSENGG', 'SENCO', 'NITINSPIN', 'FIEMIND', 'PREMIERPOL', 'TBZ', 'RBZJEWEL', 'VENUSREM', 'TCS', 'SENORES', 'KANPRPLA', 'INFY', 'SAKSOFT', 'JAYNECOIND', 'RATNAVEER', 'ACC', 'MASTEK', 'INDHOTEL', 'TATAELXSI', 'WPIL', 'HLEGLAS', 'ROSSARI', 'VAIBHAVGBL', 'EMUDHRA', 'WAAREERTL', 'BATAINDIA', 'LTM', 'CELLO', 'GNA', 'RPEL', 'GOLDIAM', 'EXPLEOSOL', 'CGPOWER', 'WELCORP', 'TAALTECH', 'V2RETAIL', 'WAKEFIT', 'RACLGEAR', 'KERNEX', 'MIDWESTLTD', 'ROLEXRINGS', 'LINDEINDIA', 'JLHL', 'INFOBEAN', 'ARIS', 'BBOX', 'MACPOWER', 'CRISIL', 'PARAS']

print(f"Testing batch download for {len(symbols)} symbols...")
tickers = [f"{s}.NS" for s in symbols]
data = yf.download(tickers, period="1mo", interval="1d", group_by='ticker', progress=False)
print("Download finished. Columns/levels:", type(data.columns))

# Verify sample
for s in ['ONWARDTEC', 'HBLENGINE', 'TECHM', 'INDHOTEL']:
    t = f"{s}.NS"
    if t in data.columns.levels[0]:
        df = data[t].dropna()
        print(f"{s}: last 2 dates: {df.index[-2].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}, last close: {df['Close'].iloc[-1]:.2f}")
    else:
        print(f"{s}: NOT FOUND")
