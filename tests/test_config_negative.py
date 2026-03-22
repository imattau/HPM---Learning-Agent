from hpm.config import AgentConfig


def test_agent_config_negative_defaults():
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.gamma_neg == 0.3
    assert cfg.neg_conflict_threshold == 0.7
    assert cfg.max_tier2_negative == 100


def test_agent_config_negative_fields_overridable():
    cfg = AgentConfig(agent_id="a", feature_dim=4,
                      gamma_neg=0.0,
                      neg_conflict_threshold=0.5,
                      max_tier2_negative=50)
    assert cfg.gamma_neg == 0.0
    assert cfg.neg_conflict_threshold == 0.5
    assert cfg.max_tier2_negative == 50


def test_neg_conflict_threshold_distinct_from_conflict_threshold():
    """neg_conflict_threshold (0.7) must not collide with conflict_threshold (0.1)."""
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.conflict_threshold == 0.1        # existing field unchanged
    assert cfg.neg_conflict_threshold == 0.7    # new field
    assert cfg.conflict_threshold != cfg.neg_conflict_threshold
