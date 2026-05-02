from hpm_ai_v5.adapter.store_size import PatternStoreSizeAdapter
from hpm_ai_v5.adapter import AdapterPacket


def test_store_size_responds_to_entropy() -> None:
    adapter = PatternStoreSizeAdapter(base_patterns=32, entropy_scale=2.0)
    # high entropy → recommendation grows
    packet = AdapterPacket(raw=1.0, context={"entropy": 0.8, "surprise": 0.9, "dominant_period": 0})
    packet = adapter.run(packet)
    high_rec = packet.context["recommended_max_patterns"]
    # low entropy, many stable steps → recommendation decays toward base
    adapter2 = PatternStoreSizeAdapter(base_patterns=32, entropy_scale=2.0, stability_window=2)
    adapter2.current_recommendation = 80
    for _ in range(5):
        packet2 = AdapterPacket(raw=1.0, context={"entropy": 0.1, "surprise": 0.1, "dominant_period": 0})
        packet2 = adapter2.run(packet2)
    assert packet2.context["recommended_max_patterns"] < 80
    assert high_rec > 32


def test_store_size_period_floor() -> None:
    adapter = PatternStoreSizeAdapter(base_patterns=8, min_patterns=4)
    packet = AdapterPacket(raw=1.0, context={"entropy": 0.0, "surprise": 0.0, "dominant_period": 6})
    packet = adapter.run(packet)
    assert packet.context["recommended_max_patterns"] >= 12  # 6 * 2
