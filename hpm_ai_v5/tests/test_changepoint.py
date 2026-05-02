"""Tests for ChangepointAdapter."""

from hpm_ai_v5.adapter.changepoint import ChangepointAdapter
from hpm_ai_v5.adapter import AdapterPacket


def test_changepoint_detects_mean_shift() -> None:
    adapter = ChangepointAdapter(window_size=16, threshold=1.5, cooldown=4)
    # fill with stable period-2 stream
    for v in [1.0, 2.0] * 8:
        packet = AdapterPacket(raw=v, context={})
        packet = adapter.run(packet)
    assert not packet.context["regime_changed"]
    assert packet.context["warm"]
    # now shift to a very different mean
    for v in [100.0, 200.0] * 8:
        packet = AdapterPacket(raw=v, context={})
        packet = adapter.run(packet)
    # at least one detection should have fired during the shift
    assert adapter._regime_count >= 1


def test_changepoint_cooldown_prevents_repeated_triggers() -> None:
    adapter = ChangepointAdapter(window_size=8, threshold=0.5, cooldown=6)
    for v in [1.0] * 4 + [100.0] * 12:
        packet = AdapterPacket(raw=v, context={})
        packet = adapter.run(packet)
    # cooldown should limit detections even during sustained shift
    assert adapter._regime_count <= 2


def test_changepoint_stable_stream_no_detection() -> None:
    adapter = ChangepointAdapter(window_size=16, threshold=1.5)
    for v in [1.0, 2.0] * 20:
        packet = AdapterPacket(raw=v, context={})
        packet = adapter.run(packet)
    assert adapter._regime_count == 0


def test_changepoint_reset_clears_state() -> None:
    adapter = ChangepointAdapter(window_size=8, threshold=0.5)
    for v in [1.0] * 4 + [100.0] * 8:
        packet = AdapterPacket(raw=v, context={})
        packet = adapter.run(packet)
    adapter.reset()
    assert len(adapter.window) == 0
    assert adapter._cooldown_remaining == 0
