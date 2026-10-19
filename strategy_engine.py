"""
========================================================================================
Portfolio 3 Strategy Engine Root Module
File: strategy_engine.py
========================================================================================
Exposes all components of Portfolio3StrategyEngine from engines/strategy_engine.py
for direct root-level imports and executions.
========================================================================================
"""

from engines.strategy_engine import (
    PORTFOLIO_TYPE,
    ENGINE_VERSION,
    Portfolio3StrategyEngine,
    portfolio3_strategy_engine,
    evaluate_prebreakout_coil,
    calculate_volume_pacing,
    _portfolio3_state
)

__all__ = [
    "PORTFOLIO_TYPE",
    "ENGINE_VERSION",
    "Portfolio3StrategyEngine",
    "portfolio3_strategy_engine",
    "evaluate_prebreakout_coil",
    "calculate_volume_pacing",
    "_portfolio3_state"
]

if __name__ == "__main__":
    print(f"Portfolio 3 Strategy Engine ({PORTFOLIO_TYPE}) v{ENGINE_VERSION} ready.")
    summary = portfolio3_strategy_engine.get_portfolio_summary()
    print("Portfolio 3 Summary:")
    for k, v in summary.items():
        if k != "stats":
            print(f"  {k}: {v}")
