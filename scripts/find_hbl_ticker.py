import yfinance as yf

for sym in ['HBLENGINE.NS', 'HBLPOWER.NS', 'HBLENG.NS']:
    try:
        t = yf.Ticker(sym)
        hist = t.history(period='5d')
        if not hist.empty:
            c = hist['Close'].iloc[-1]
            print(f"Valid ticker: {sym}, last close: {c:.2f}")
    except Exception as e:
        print(f"Error for {sym}: {e}")
