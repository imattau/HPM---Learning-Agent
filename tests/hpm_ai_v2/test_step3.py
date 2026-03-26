import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem

def test_level1_stabilization():
    """
    Simulate a stream of observations that favor pattern p1.
    Verify its weight converges to a higher value than p2.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    # Initialize with equal weights
    p1.weight = 0.5
    p2.weight = 0.5
    
    rule = MetaPatternRule(learning_rate=0.2, conflict_scale=0.1)
    system = HPMSystem([p1, p2], rule, ema_lambda=0.1)
    
    # Simulate a stream of 50 steps
    for t in range(50):
        # Observation favors p1 (lower loss)
        loss1 = 0.2
        loss2 = 2.0
        
        # Take a step in the system (updates ema_loss and weights)
        system.step(surface_loss=np.array([loss1, loss2]))
        
    # After stabilization, p1 should have a much higher weight
    assert p1.weight > 0.8
    assert p2.weight < 0.2
    assert np.isclose(p1.weight + p2.weight, 1.0)

def test_level1_stabilization_with_conflict():
    """
    Verify that conflict inhibition speeds up pruning of the worse pattern.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    p1.weight = 0.5
    p2.weight = 0.5
    
    rule = MetaPatternRule(learning_rate=0.2, conflict_scale=0.5) # High conflict
    system = HPMSystem([p1, p2], rule, ema_lambda=0.1)
    # Fully incompatible
    system.kappa_matrix = np.array([[0, 1], [1, 0]])
    
    # Simulate fewer steps than before
    for t in range(20):
        # Observation favors p1 (lower loss)
        loss1 = 0.2
        loss2 = 2.0
        
        system.step(surface_loss=np.array([loss1, loss2]))
        
    # With conflict, it should stabilize even faster
    assert p1.weight > 0.9
    assert p2.weight < 0.1
