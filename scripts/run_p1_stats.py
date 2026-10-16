from execution.state_engine import state_engine
import json

stats = state_engine.get_portfolio_stats(portfolio_type="INSTITUTIONAL_6PILLAR")
print("=== PORTFOLIO 1 STATS ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
