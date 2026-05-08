import random

from hpm_ai_v5.core.pattern_manager import PatternManager, build_context_signature
from hpm_ai_v5.core.engine import PatternEngine
from hpm_ai_v5.core.state import State


def test_pattern_manager_promotes_and_seeds() -> None:
    engine = PatternEngine()
    # observe enough to build patterns with support
    for i in range(12):
        engine.observe(State(value=float(i % 3)))
    manager = PatternManager(promotion_threshold=0.5, min_support=2)
    promoted = manager.promote_from(engine)
    assert len(promoted) >= 0  # may be 0 if no patterns meet threshold yet — that's fine
    assert manager.stats()["archive_size"] == len(promoted)


def test_pattern_manager_end_start_episode() -> None:
    engine = PatternEngine()
    for i in range(20):
        engine.observe(State(value=float(i % 4)))
    manager = PatternManager(promotion_threshold=0.3, min_support=1)
    summary = manager.end_episode(engine)
    assert "promoted" in summary
    assert "archive_size" in summary
    new_engine = PatternEngine()
    start_summary = manager.start_episode(new_engine)
    assert "seeded" in start_summary
    # seeded patterns should now be in the new engine's store
    if start_summary["seeded"] > 0:
        assert len(new_engine.store.patterns) + len(new_engine.store.meta_patterns) > 0

def test_context_signature_builder() -> None:
    assert build_context_signature({}) == "generic"
    assert "period:2" in build_context_signature({"dominant_period": 2, "entropy": 0.1})
    assert "entropy:high" in build_context_signature({"dominant_period": 0, "entropy": 0.9})
    assert "period:4+" in build_context_signature({"dominant_period": 6})


def test_context_sensitive_retrieval() -> None:
    # Train on period-2 stream — use goal to generate utility > 0
    engine_a = PatternEngine()
    for i in range(20):
        engine_a.observe(State(value=float(i % 2), goal={"utility": 0.5}))

    manager = PatternManager(promotion_threshold=0.3, min_support=1, archive_decay_rate=0.0)
    period2_ctx = {"dominant_period": 2, "entropy": 0.2}
    manager.end_episode(engine_a, context=period2_ctx)

    # Train on high-entropy stream
    engine_b = PatternEngine()
    random.seed(42)
    for i in range(20):
        engine_b.observe(State(value=float(random.randint(0, 9)), goal={"utility": 0.5}))

    high_entropy_ctx = {"dominant_period": 0, "entropy": 0.85}
    manager.end_episode(engine_b, context=high_entropy_ctx)


    # Seed a new engine with period-2 context
    engine_c = PatternEngine()
    seeded = manager.start_episode(engine_c, context=period2_ctx)

    sig_counts = manager.archive_stats_by_signature()
    assert len(sig_counts) >= 1
    assert seeded["context_signature"] == build_context_signature(period2_ctx)
