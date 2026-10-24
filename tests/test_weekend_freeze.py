from engines.order_flow_structure_engine import OrderFlowStructureEngine

def test_order_flow_weekend_freeze():
    eng = OrderFlowStructureEngine()
    
    # 1. Volume Profile session stability
    vp = eng.compute_volume_profile()
    assert vp["poc"] > 0
    assert vp["vah"] >= vp["val"]
    
    # 2. CVD must be completely frozen across multiple calls outside market hours
    first_cvd = eng.compute_cumulative_volume_delta()
    for _ in range(10):
        next_cvd = eng.compute_cumulative_volume_delta()
        assert next_cvd["current_cvd"] == first_cvd["current_cvd"], "CVD must not accumulate when market is closed!"
        assert next_cvd["instantaneous_delta"] == 0, "Instantaneous delta must be 0 when market is closed!"
    
    # 3. Multi-index confluence consistency
    conf = eng.compute_multi_index_confluence()
    nifty_c = next(c for c in conf["constituents"] if c["symbol"] == "NIFTY 50")
    assert nifty_c["ltp"] > 20000
    assert nifty_c["change_pct"] == -0.39
