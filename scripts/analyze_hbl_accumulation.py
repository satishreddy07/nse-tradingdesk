import yfinance as yf
import pandas as pd
import numpy as np

df = yf.download('HBLENGINE.NS', period='3mo', interval='1d', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['Dist_EMA20_Pct'] = (df['Close'] - df['EMA20']) / df['EMA20'] * 100.0
df['Change_Pct'] = df['Close'].pct_change() * 100.0
df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA20']
df['Turnover_Cr'] = (df['Close'] * df['Volume']) / 1e7

# ATR 14
tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
df['ATR14'] = tr.rolling(14).mean()
df['ATR_Pct'] = df['ATR14'] / df['Close'] * 100.0

# RSI 14
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI14'] = 100 - (100 / (1 + rs))

tail = df.tail(20)
print("=== HBLENGINE.NS (HBL ENGINEERING) ACCUMULATION FORENSICS (LAST 20 SESSIONS) ===")
for idx, row in tail.iterrows():
    date_str = idx.strftime('%Y-%m-%d')
    c = row['Close']
    chg = row['Change_Pct']
    h = row['High']
    l = row['Low']
    rng = row['Daily_Range_Pct']
    vol = int(row['Volume'])
    to_day = row['Turnover_Cr']
    ema20 = row['EMA20']
    dist20 = row['Dist_EMA20_Pct']
    v_rat = row['Vol_Ratio']
    rsi = row['RSI14']
    atr_pct = row['ATR_Pct']
    print(f"{date_str} | Close: {c:6.2f} ({chg:+6.2f}%) | Range: {rng:5.2f}% | Vol: {vol:10d} | Turnover: {to_day:6.2f}Cr | VolRat: {v_rat:5.2f} | EMA20: {ema20:6.2f} | Dist20: {dist20:+6.2f}% | ATR%: {atr_pct:4.2f}% | RSI: {rsi:5.1f}")
