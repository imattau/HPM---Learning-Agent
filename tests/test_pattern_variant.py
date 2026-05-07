# tests/test_pattern_variant.py
from hpm_ai_v5.core.variant import PatternVariant, make_variant

def test_pattern_variant_creation():
    v = PatternVariant(
        name="variant_0",
        member_names=["pattern_1", "pattern_2"],
        centroid=(1.0, 2.0, 3.0),
        hit_count=10,
        context_signature={"domain": "nlp"},
        score=0.75,
    )
    assert v.name == "variant_0"
    assert v.member_names == ["pattern_1", "pattern_2"]
    assert v.centroid == (1.0, 2.0, 3.0)
    assert v.hit_count == 10
    assert v.score == 0.75

def test_make_variant_centroid():
    from hpm_ai_v5.core.pattern import Pattern
    p1 = Pattern(name="p1", template=(1.0, 2.0), support=3, utility=0.8)
    p2 = Pattern(name="p2", template=(1.2, 2.2), support=5, utility=0.6)
    v = make_variant([p1, p2], name="variant_0")
    assert v.member_names == ["p1", "p2"]
    assert abs(v.centroid[0] - 1.1) < 1e-6
    assert abs(v.centroid[1] - 2.1) < 1e-6
    assert v.hit_count == 8

def test_core_config_has_consolidation_threshold():
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig()
    assert hasattr(config, "consolidation_threshold")
    assert config.consolidation_threshold == 0.8

def test_core_config_custom_threshold():
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(consolidation_threshold=0.5)
    assert config.consolidation_threshold == 0.5

def test_store_has_variants_dict():
    from hpm_ai_v5.core.store import PatternStore
    store = PatternStore()
    assert hasattr(store, "variants")
    assert isinstance(store.variants, dict)

def test_store_register_variant():
    from hpm_ai_v5.core.store import PatternStore
    from hpm_ai_v5.core.variant import PatternVariant
    store = PatternStore()
    v = PatternVariant(
        name="v0", member_names=["p1"], centroid=(1.0,),
        hit_count=3, context_signature={}, score=0.5,
    )
    store.register_variant(v)
    assert "v0" in store.variants

def test_store_match_falls_back_to_variant():
    from hpm_ai_v5.core.store import PatternStore
    from hpm_ai_v5.core.variant import PatternVariant
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(near_threshold=2.0, max_patterns=32)
    store = PatternStore(config=config)
    v = PatternVariant(
        name="v0", member_names=[], centroid=(1.0, 2.0),
        hit_count=5, context_signature={}, score=0.8,
    )
    store.register_variant(v)
    result = store.match((1.1, 2.1))
    assert result.status == "variant"
    assert result.pattern is None
    assert result.variant is not None
    assert result.variant.name == "v0"

def test_consolidation_promotes_near_duplicates():
    from hpm_ai_v5.core import PatternEngine, PatternManager
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(
        max_patterns=10, near_threshold=0.5, consolidation_threshold=0.5,
    )
    engine = PatternEngine(config=config)
    manager = PatternManager(promotion_threshold=0.01)
    for i in range(6):
        engine.store.learn((1.0 + i * 0.05, 2.0 + i * 0.05))
    manager.start_episode(engine)
    manager.end_episode(engine)
    assert len(engine.store.variants) >= 1

def test_consolidation_retains_concrete_patterns():
    from hpm_ai_v5.core import PatternEngine, PatternManager
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(max_patterns=10, near_threshold=0.5, consolidation_threshold=0.5)
    engine = PatternEngine(config=config)
    manager = PatternManager(promotion_threshold=0.01)
    for i in range(6):
        engine.store.learn((1.0 + i * 0.05, 2.0 + i * 0.05))
    count_before = len(engine.store.patterns)
    manager.start_episode(engine)
    manager.end_episode(engine)
    assert len(engine.store.patterns) == count_before
