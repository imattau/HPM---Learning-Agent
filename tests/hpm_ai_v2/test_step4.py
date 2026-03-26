import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem

def test_internalization_settling():
    """
    Verify that a successful pattern (high weight/density)
    shifts from INTERNAL_FLEX to INTERNAL_PROC.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p1.weight = 1.0 # High weight
    
    # We want to boost affective score to increase density
    # p.calculate_density(connectivity, social_frequency)
    # Density D(h) = 0.4*connectivity + 0.3*(affective + social) + 0.3*social_freq
    
    rule = MetaPatternRule(learning_rate=0.1, density_threshold=0.8)
    system = HPMSystem([p1], rule, ema_lambda=0.1)
    
    # Simulate a pattern that gains high density
    # If connectivity=1.0, and affective=1.0, social=1.0
    # D(h) = 0.4*1.0 + 0.3*(1.0 + 1.0) + 0.3*0.0 = 0.4 + 0.6 = 1.0
    
    p1.affective_score = 1.0
    p1.social_score = 1.0
    
    # Step in the system (this calls check_substrate_shift)
    system.step(surface_loss=np.array([0.0]))
    
    assert p1.substrate_id == SubstrateID.INTERNAL_PROC
    # After shifting, verify its resource cost has dropped
    assert p1.resource_cost == 0.1 # 10x cheaper
