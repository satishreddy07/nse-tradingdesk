import json

with open('scripts/all_latest_signals.json') as f:
    signals = json.load(f)

print(f"Total signals in DB: {len(signals)}")
gainers_sample = ['ONWARDTEC', 'HBLENGINE', 'TECHM', 'TIPSMUSIC', 'SHRIPISTON', 'SMSPHARMA', 'RPGLIFE', 'WIPRO', 'INDHOTEL']

for s in gainers_sample:
    if s in signals:
        print(f"{s}: {signals[s]}")
    else:
        print(f"{s}: NOT FOUND")
