import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem
from hpm_ai_v2.recombination import StructuralRecombinator

def test_level4_surface_noise_immunity():
    """
    Verify Level 4 immunity to surface noise compared to Level 1.
    """
    p_l1 = CompositePattern(id="l1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX)
    p_l4 = CompositePattern(id="l4", level=4, substrate_id=SubstrateID.INTERNAL_FLEX)
    
    # Setup foundations for L4 so it doesn't get penalized
    p_l3 = CompositePattern(id="l3", level=3, substrate_id=SubstrateID.INTERNAL_FLEX)
    p_l3.affective_score = 1.0 # high density foundation
    
    rule = MetaPatternRule(learning_rate=0.5)
    system = HPMSystem([p_l1, p_l3, p_l4], rule)
    
    # High surface noise, stable structure
    surf_loss = np.array([5.0, 5.0, 5.0])
    struct_loss = np.array([0.0, 0.1, 0.1])
    
    for _ in range(10):
        system.step(surf_loss, struct_loss)
        
    # Level 4 should have much better total score than Level 1 
    # because alpha_4 = 0.05 vs alpha_1 = 0.9
    assert p_l4.total_score > p_l1.total_score

def test_level5_generative_utility():
    """
    Verify Level 5 patterns gain generative utility from contributions.
    """
    p_l5 = CompositePattern(id="l5", level=5, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"CORE"})
    p_l1 = CompositePattern(id="l1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"EXT"})
    
    recombinator = StructuralRecombinator()
    
    # Contribute to a successful merge
    p_star = recombinator.recombine(p_l5, p_l1)
    
    # Level 5 pattern should now have generative utility
    assert p_l5.generative_utility > 0
    # This utility should reflect in its total score
    assert p_l5.total_score > -p_l5.ema_loss - p_l5.resource_cost
