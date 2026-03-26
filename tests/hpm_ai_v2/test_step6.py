import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem

def test_external_sym_persistence():
    """
    Verify that EXTERNAL_SYM patterns persist (0 decay)
    even if accuracy is low, provided density remains.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.EXTERNAL_SYM)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    # Initialize weights
    p1.weight = 0.5
    p2.weight = 0.5
    
    # p1 has MODERATE accuracy loss but high social score (Superstition in a field)
    p1.ema_loss = 2.0
    p1.social_score = 1.0
    
    # p2 has BETTER accuracy but high decay
    p2.ema_loss = 0.5
    
    rule = MetaPatternRule(learning_rate=0.1, stability_kappa=2.0)
    system = HPMSystem([p1, p2], rule)
    
    # Run for 20 steps
    for t in range(20):
        system.step(surface_loss=np.array([p1.ema_loss, p2.ema_loss]))
        
    # p1 should still have significant weight because of 0 decay and social support
    # even with -2.0 accuracy.
    assert p1.weight > 0.1
    
def test_externalization_shift():
    """
    Verify the path INTERNAL_FLEX -> INTERNAL_PROC -> EXTERNAL_SYM.
    """
    p = CompositePattern(id="p", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    rule = MetaPatternRule(density_threshold=0.8)
    system = HPMSystem([p], rule)
    
    # Internalization
    p.affective_score = 1.0
    p.social_score = 1.0
    system.step(surface_loss=np.array([0.0]))
    assert p.substrate_id == SubstrateID.INTERNAL_PROC
    
    # Externalization (requires high social score)
    system.step(surface_loss=np.array([0.0]))
    assert p.substrate_id == SubstrateID.EXTERNAL_SYM
