import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule

def test_meta_pattern_rule_weight_update():
    # Setup two patterns, one high accuracy, one low
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    p1.ema_loss = 0.5 # Better accuracy
    p2.ema_loss = 2.0 # Worse accuracy
    
    # Starting weights
    p1.weight = 0.5
    p2.weight = 0.5
    
    rule = MetaPatternRule(learning_rate=0.1, conflict_scale=0.0) # No inhibition
    
    # We expect p1 to gain weight and p2 to lose weight
    rule.update_weights([p1, p2], np.array([]))
    
    assert p1.weight > 0.5
    assert p2.weight < 0.5
    assert np.isclose(p1.weight + p2.weight, 1.0)

def test_substrate_shift_logic():
    p = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    rule = MetaPatternRule(density_threshold=0.8)
    
    # Density below threshold
    shifted = rule.check_substrate_shift(p, connectivity=0.5, social_freq=0.1)
    assert not shifted
    assert p.substrate_id == SubstrateID.INTERNAL_FLEX
    
    # Density above threshold
    # D(h) = 0.4*1.0 + 0.3*(1.0 + 1.0) + 0.3*0.5 = 0.4 + 0.6 + 0.15 = 1.15
    p.affective_score = 1.0
    p.social_score = 1.0
    shifted = rule.check_substrate_shift(p, connectivity=1.0, social_freq=0.5)
    assert shifted
    assert p.substrate_id == SubstrateID.INTERNAL_PROC

def test_conflict_inhibition():
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    p1.ema_loss = 1.0
    p2.ema_loss = 1.0
    p1.weight = 0.5
    p2.weight = 0.5
    
    # With zero conflict, weights should remain 0.5 (assuming equal decay)
    rule_no_conflict = MetaPatternRule(learning_rate=0.1, conflict_scale=0.0)
    rule_no_conflict.update_weights([p1, p2], np.array([]))
    assert np.isclose(p1.weight, 0.5)
    
    # With high conflict inhibition, and high kappa, weights should stay near equal but decay?
    # Actually if they both have same score and same weight, replicator term is 0.
    # Inhibition term is beta_c * kappa * w1 * w2.
    kappa = np.array([[0, 1], [1, 0]])
    rule_conflict = MetaPatternRule(learning_rate=0.1, conflict_scale=0.5)
    rule_conflict.update_weights([p1, p2], kappa)
    
    # They should both be 0.5 because of normalization
    assert np.isclose(p1.weight, 0.5)
    assert np.isclose(p2.weight, 0.5)
