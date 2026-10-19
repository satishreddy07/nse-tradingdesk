import requests
import json

resp = requests.get("http://127.0.0.1:8000/api/nifty/predictions")
data = resp.json()
print("Status:", data.get("status"))
print("Timestamp:", data.get("timestamp_ist"))
print("Spot:", data.get("spot_price"))
print("T+1 verdict:", data.get("t1_forecast", {}).get("verdict"))
print("T+5 verdict:", data.get("t5_forecast", {}).get("verdict"))

print("\nRecent Forward-Testing Ledger Entries:")
for r in data.get("prediction_ledger", [])[:5]:
    print(f"Date: {r.get('prediction_date')} | Call: {r.get('t1_verdict')} (Score: {r.get('composite_score')}) | Close: {r.get('actual_t1_close')} | Validated: {r.get('t1_validated')}")
