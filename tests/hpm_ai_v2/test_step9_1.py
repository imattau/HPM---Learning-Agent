import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem

def test_step_9_1_structural_sensitivity():
    """
    HPM Step 9.1: Compare Level 1 (Surface) vs Level 3 (Relational).
    Prediction: Level 3 is more robust to surface noise but more sensitive to structural changes.
    """
    # 1. Setup
    p_l1 = CompositePattern(id="l1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A"})
    p_l2 = CompositePattern(id="l2", level=2, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A", "B"})
    p_l3 = CompositePattern(id="l3", level=3, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"REL_X"})
    
    # Mature L1 and L2 so L3 can exist without penalty
    p_l1.affective_score = 2.0
    p_l2.affective_score = 2.0
    p_l3.affective_score = 2.5
    
    p_l1.weight = 0.45
    p_l2.weight = 0.1
    p_l3.weight = 0.45
    
    rule = MetaPatternRule(learning_rate=0.2)
    system = HPMSystem([p_l1, p_l2, p_l3], rule)
    
    # 2. Phase 1: Surface Noise
    # Feed data where surface is noisy but structure is preserved
    for _ in range(20):
        # Slightly less noise so L3 can overcome its higher level-based cost
        surf_loss = np.array([1.5, 1.5, 1.5]) 
        struct_loss = np.array([0.0, 0.0, 0.05]) 
        
        system.step(surface_loss=surf_loss, structural_loss=struct_loss)
        
    # Prediction: L3 should have higher total score than L1 because it weights 
    # structural fit (0.1) at 80% and surface noise (2.0) at only 20%.
    # L1 weights surface noise (2.0) at 90%.
    
    assert p_l3.total_score > p_l1.total_score
    assert p_l3.weight > p_l1.weight
    
    # 3. Phase 2: Structural Perturbation
    # Reset weights to ensure L1 is not dead
    p_l1.weight = 0.45
    p_l2.weight = 0.1
    p_l3.weight = 0.45
    
    # Feed data where surface is clean but structure is broken
    for _ in range(500):
        # L1 is now significantly more accurate than L3's surface mapping
        surf_loss = np.array([0.01, 0.01, 0.5])
        struct_loss = np.array([0.0, 0.0, 5.0]) # Structure is broken
        
        system.step(surface_loss=surf_loss, structural_loss=struct_loss)
        
    # Prediction: L3 should now collapse because it is highly sensitive to structure (80% weight)
    assert p_l3.total_score < p_l1.total_score
    assert p_l1.weight > p_l3.weight
