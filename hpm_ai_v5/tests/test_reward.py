from hpm_ai_v5.adapter.reward import RewardAdapter
from hpm_ai_v5.adapter import AdapterPacket, NumericAdapter


def test_reward_correct_prediction_increases_utility() -> None:
    adapter = RewardAdapter(correct_reward=1.0, incorrect_penalty=0.0, decay=0.9)
    numeric = NumericAdapter()
    # simulate two steps: predict correctly
    p1 = AdapterPacket(raw=1.0, context={})
    p1 = numeric.run(p1)
    p1 = adapter.run(p1)
    # on step 2, tell reward adapter we predicted 2.0, actual is 2.0
    p2 = AdapterPacket(raw=2.0, context={"predicted_next": 2.0})
    p2 = numeric.run(p2)
    p2 = adapter.run(p2)
    assert p2.context["reward_correct"] == True
    assert p2.context["reward_running"] > 0.0


def test_reward_wrong_prediction_zero_utility() -> None:
    adapter = RewardAdapter(correct_reward=1.0, incorrect_penalty=0.0, decay=0.9)
    numeric = NumericAdapter()
    p1 = AdapterPacket(raw=1.0, context={})
    p1 = numeric.run(p1)
    p1 = adapter.run(p1)
    p2 = AdapterPacket(raw=2.0, context={"predicted_next": 99.0})  # wrong prediction
    p2 = numeric.run(p2)
    p2 = adapter.run(p2)
    assert p2.context["reward_correct"] == False


def test_reward_rolling_accuracy() -> None:
    adapter = RewardAdapter(correct_reward=1.0, incorrect_penalty=0.0, window_size=4)
    numeric = NumericAdapter()
    prev = None
    for v in [1.0, 1.0, 1.0, 1.0, 1.0]:
        ctx = {"predicted_next": prev} if prev is not None else {}
        p = AdapterPacket(raw=v, context=ctx)
        p = numeric.run(p)
        p = adapter.run(p)
        prev = v  # perfect prediction: predict current value repeats next
    assert adapter.accuracy > 0.0


def test_reward_reset() -> None:
    adapter = RewardAdapter()
    numeric = NumericAdapter()
    for v in [1.0, 2.0]:
        p = AdapterPacket(raw=v, context={})
        p = numeric.run(p)
        adapter.run(p)
    adapter.reset()
    assert adapter._total == 0
    assert adapter._running_reward == 0.0
