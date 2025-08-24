from auggie.perf import aggregate_telemetry, recommend_for_category
from providers.registry import ModelProviderRegistry as R


def test_perf_aggregate_and_recommend(monkeypatch):
    # seed telemetry
    R._telemetry.clear()
    R._telemetry.update({
        "a": {"success": 10, "failure": 0, "latency_ms": [100,120], "input_tokens": 5, "output_tokens": 10},
        "b": {"success": 5, "failure": 5, "latency_ms": [50,60], "input_tokens": 3, "output_tokens": 6},
    })
    agg = aggregate_telemetry()
    assert "a" in agg and agg["a"]["success_rate"] == 1.0
    rec = recommend_for_category("FAST_RESPONSE")
    assert rec[0] == "a"

