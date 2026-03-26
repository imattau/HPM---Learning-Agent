import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem
from hpm_ai_v2.evaluators import InstitutionalField
from hpm_ai_v2.recombination import StructuralRecombinator

def test_milestone_sweep():
    """
    Final HPM Validation: Full trajectory sweep.
    """
    # 1. Milestone A: Level 1 vs. Level 2 emergence
    # Setup Level 1 (Surface) and Level 2 (Local Structural)
    p1_l1 = CompositePattern(id="l1_p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A"})
    p1_l2 = CompositePattern(id="l2_p1", level=2, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A", "B"})
    
    p1_l1.weight = 0.5
    p1_l2.weight = 0.5
    
    rule = MetaPatternRule(learning_rate=0.2, density_threshold=0.8, stability_kappa=1.0)
    system = HPMSystem([p1_l1, p1_l2], rule)
    
    # Early phase: Level 1 is favored by maturation gate
    # Even if accuracies are equal, Level 1 should win because of lower cost and gate
    p1_l1.ema_loss = 0.1
    p1_l2.ema_loss = 0.1
    
    system.step(surface_loss=np.array([p1_l1.ema_loss, p1_l2.ema_loss]))
    # After one step, Level 1 weight should increase because L2 is gated
    assert p1_l1.weight > 0.5
    assert p1_l2.weight < 0.5
    
    # 2. Milestone B: Superstition Persistence
    # Let's make Level 1 pattern "sticky" (INTERNAL_PROC)
    p1_l1.affective_score = 1.0
    p1_l1.social_score = 1.0
    system.step(surface_loss=np.array([p1_l1.ema_loss, p1_l2.ema_loss]))
    assert p1_l1.substrate_id == SubstrateID.INTERNAL_PROC
    
    # Now Level 1 accuracy drops (it's inaccurate but dense)
    p1_l1.ema_loss = 2.0
    # It should still persist for a while because of Stability Bias and 0.01 decay vs 0.1 decay
    old_weight = p1_l1.weight
    system.step(surface_loss=np.array([p1_l1.ema_loss, p1_l2.ema_loss]))
    assert p1_l1.weight > 0.1 # Persists despite accuracy drop
    
    # 3. Milestone C: Scientific Refinement
    # Introduce an Institutional Field that prunes the superstitious p1_l1
    field = InstitutionalField(replication_threshold=0.5)
    
    # Simulate a few steps in the scientific field
    for _ in range(5):
        # Apply filter to p1_l1 (high loss -> low multiplier)
        mult = field.apply_filter(p1_l1, 2.0)
        # Apply filter to p1_l2 (low loss -> high multiplier)
        mult_l2 = field.apply_filter(p1_l2, 0.1)
        
    # Apply scientific refinement to social scores
    p1_l1.social_score *= mult
    p1_l2.social_score *= mult_l2
    
    system.step(surface_loss=np.array([p1_l1.ema_loss, p1_l2.ema_loss]))
    # Now Level 2 should start gaining ground as Level 1's field support is stripped
    assert p1_l2.weight > 0.0 # It survived
    
    # 4. Milestone D: Recombinative Insight
    recombinator = StructuralRecombinator()
    # Merge Level 1 and Level 2 to find a better structure
    p_star = recombinator.recombine(p1_l1, p1_l2)
    assert p_star is not None
    assert p_star.level == 2
    assert p_star.affective_score > 0.5 # Insight boost
    
    # Add p_star to population
    system.patterns.append(p_star)
    # Re-normalize weights
    total_w = sum(p.weight for p in system.patterns)
    for p in system.patterns:
        p.weight /= total_w
        
    # Step again
    system.step(surface_loss=np.array([p1_l1.ema_loss, p1_l2.ema_loss, p_star.ema_loss]))
    # p_star should be competitive
    assert p_star.weight > 0.0
