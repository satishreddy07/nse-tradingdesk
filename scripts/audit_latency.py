import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from web.api import app

client = TestClient(app)

endpoints = [
    '/api/status',
    '/api/portfolio/INSTITUTIONAL_6PILLAR/summary',
    '/api/portfolio/INSTITUTIONAL_6PILLAR/positions',
    '/api/portfolio/INSTITUTIONAL_6PILLAR/signals',
    '/api/portfolio/NIFTY500_MOMENTUM/summary',
    '/api/portfolio/NIFTY500_MOMENTUM/positions',
    '/api/portfolio/NIFTY500_MOMENTUM/signals',
    '/api/portfolio/equity-curves',
    '/api/vault/status',
    '/api/stocks/fetched',
    '/api/execution/queue',
    '/api/nifty/predictions',
    '/api/nifty/options-chain',
    '/api/options/portfolio',
    '/api/ai/status',
    '/api/ai/analysis',
    '/api/nifty/us-market-intelligence',
    '/api/nifty/intraday-expiry',
    '/api/nifty/fao-history',
    '/api/calendar'
]

print(f"{'Endpoint':<45} | {'Status':<6} | {'Latency':<12} | {'Size':<12}")
print('-' * 85)
for ep in endpoints:
    t0 = time.time()
    try:
        res = client.get(ep)
        lat = (time.time() - t0) * 1000
        size = len(res.content)
        print(f"{ep:<45} | {res.status_code:<6} | {lat:>9.2f} ms | {size:>10} B")
    except Exception as e:
        lat = (time.time() - t0) * 1000
        print(f"{ep:<45} | ERROR  | {lat:>9.2f} ms | {str(e)[:25]}")
