import yfinance as yf
import pandas as pd
import numpy as np

# Download daily data for ONWARDTEC.NS
df = yf.download('ONWARDTEC.NS', period='6mo', interval='1d', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['Dist_EMA20_Pct'] = (df['Close'] - df['EMA20']) / df['EMA20'] * 100.0
df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7
df['Turnover_SMA50_Cr'] = (df['Close'] * df['Volume']).rolling(50).mean() / 1e7
df['Change_Pct'] = df['Close'].pct_change() * 100.0
df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA20']

# RSI 14 (Wilder's style ewm)
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI14'] = 100 - (100 / (1 + rs))

tail = df.tail(15)
print("=== ONWARDTEC.NS DAILY BAR TELEMETRY (LAST 15 TRADING SESSIONS) ===")
for idx, row in tail.iterrows():
    date_str = idx.strftime('%Y-%m-%d')
    c = row['Close']
    chg = row['Change_Pct']
    vol = int(row['Volume'])
    to_50 = row['Turnover_SMA50_Cr']
    to_day = row['Turnover_Cr']
    ema20 = row['EMA20']
    dist20 = row['Dist_EMA20_Pct']
    v_rat = row['Vol_Ratio']
    rsi = row['RSI14']
    print(f"{date_str} | Close: {c:6.2f} ({chg:+6.2f}%) | Vol: {vol:9d} | DayTO: {to_day:5.2f}Cr | 50D_TO: {to_50:5.2f}Cr | EMA20: {ema20:6.2f} | Dist20: {dist20:+6.2f}% | VolRat: {v_rat:5.2f} | RSI: {rsi:5.1f}")
