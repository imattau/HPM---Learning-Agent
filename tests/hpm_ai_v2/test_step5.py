import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem

def test_maturation_penalty_level2():
    """
    Verify that a Level 2 pattern is penalized if Level 1 foundation is weak.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=2, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    # Low density for Level 1
    p1.affective_score = 0.0
    p1.social_score = 0.0
    
    # High accuracy for Level 2 (should be high total score)
    p2.ema_loss = 0.1
    p2.affective_score = 1.0
    
    rule = MetaPatternRule(density_threshold=0.8)
    system = HPMSystem([p1, p2], rule)
    
    # Calculate scores with gate
    # p1 (level 1) penalty should be 0.0
    # p2 (level 2) penalty should be significantly > 0.0
    assert np.isclose(rule.calculate_maturation_gate(p1.level, system.patterns), 0.0)
    assert rule.calculate_maturation_gate(p2.level, system.patterns) > 1.0

def test_maturation_progression():
    """
    Verify that Level 2 pattern can stabilize once Level 1 foundation is strong.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=2, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    p1.weight = 0.5
    p2.weight = 0.5
    
    # Strong Level 1 foundation
    p1.affective_score = 1.0
    p1.social_score = 1.0
    
    # Level 2 is also strong
    p2.ema_loss = 0.1
    p2.affective_score = 1.0
    p2.social_score = 1.0
    
    rule = MetaPatternRule(density_threshold=0.8)
    system = HPMSystem([p1, p2], rule)
    
    # Penalty for level 2 should be 0.0 now
    assert np.isclose(rule.calculate_maturation_gate(p2.level, system.patterns), 0.0)
    
    # Both should be competing fairly
    system.step(surface_loss=np.array([p1.ema_loss, p2.ema_loss]))
    
    # Both should have significant weight
    assert p1.weight > 0.4
    assert p2.weight > 0.4
