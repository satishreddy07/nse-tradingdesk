from engines.swing_quant_ml_engine import swing_quant_ml_engine

print("Classifier loaded:", swing_quant_ml_engine._classifier is not None)
print("Regressor loaded:", swing_quant_ml_engine._regressor is not None)
print("Scaler loaded:", swing_quant_ml_engine._scaler is not None)

# Test evaluation on a dummy top setup
dummy = {
    'score': 92.0,
    'confidence': 5,
    'rsi14': 58.0,
    'atr_pct': 3.5,
    'dist_to_pivot': 1.0,
    'dist_to_ema20': 2.0,
    'vol_ratio': 2.5,
    'inst_score': 88.0,
    'week_confluence': 1,
    'close': 500.0,
    'sl': 475.0,
    't1': 550.0,
    't2': 600.0,
    'rr': 2.0,
    'sector': 'TECHNOLOGY'
}
eval_res = swing_quant_ml_engine.evaluate_candidate(dummy)
print("Dummy evaluation result:")
print("win_prob:", eval_res.get("win_prob"))
print("verdict:", eval_res.get("verdict"))
print("decision:", eval_res.get("decision"))
print("fatal_flags:", [f for f in eval_res.get("risk_flags", []) if "OVEREXTENDED" in f or "RSI" in f])
