# tests/test_atis_benchmark.py
def test_atis_loads_train_split():
    from hpm_ai_v5.adapter.atis import load_atis
    train, test = load_atis()
    assert len(train) > 1000
    assert len(test) > 100
    assert "text" in train[0]
    assert "intent" in train[0]

def test_intent_label_adapter_injects_label():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label="flight")
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert result.context.get("intent_label") == "flight"

def test_intent_label_adapter_withheld_in_inference():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label=None)
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert "intent_label" not in result.context
