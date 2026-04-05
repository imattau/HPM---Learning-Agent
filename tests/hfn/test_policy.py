import pytest

from hfn.policy import (
    DecisionPolicy,
    CreateContext,
    AbsorptionContext,
    StructureArbitrationContext,
)


def test_expand_score_prefers_surprise_and_penalizes_weight():
    p = DecisionPolicy()
    assert p.expand_score(2.0, 0.1) > p.expand_score(1.0, 0.1)
    assert p.expand_score(2.0, 0.1) > p.expand_score(2.0, 0.9)


def test_create_score_drops_under_density_when_lacunarity_on():
    p = DecisionPolicy()
    low_density = p.create_score(3.0, 0.2, lacunarity_enabled=True)
    high_density = p.create_score(3.0, 3.0, lacunarity_enabled=True)
    assert high_density < low_density


def test_should_create_respects_density_suppression():
    p = DecisionPolicy()
    ctx = CreateContext(
        forest_size=10,
        residual_surprise=3.0,
        residual_threshold=2.0,
        lacunarity_enabled=True,
        density_ratio=3.0,
        density_factor=2.0,
    )
    assert not p.should_create(ctx)


def test_absorb_score_increases_with_miss_and_overlap():
    p = DecisionPolicy()
    s1 = p.absorb_score(miss_count=2, effective_threshold=3, overlap=0.1)
    s2 = p.absorb_score(miss_count=6, effective_threshold=3, overlap=0.6)
    assert s2 > s1
    assert p.should_absorb(s2)


def test_structure_arbitration_orders_actions():
    p = DecisionPolicy()
    actions = p.arbitrate_structure_actions(
        StructureArbitrationContext(
            create_allowed=True,
            create_score=2.0,
            compress_allowed=True,
            compress_score=1.0,
            gap_allowed=True,
        )
    )
    assert actions.create_first
    assert not actions.run_gap_query
