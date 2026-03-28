import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.dynamics import MetaPatternRule
from hpm_ai_v2.system import HPMSystem
from hpm_ai_v2.evaluators import InstitutionalField
from hpm_ai_v2.recombination import StructuralRecombinator

def test_hpm_v1_25_e2e_fidelity():
    """
    E2E Fidelity Test: Simulates a full developmental trajectory.
    Identifies gaps in the 4 Roles (Substrates, Dynamics, Evaluators, Fields)
    and the 5 Levels of Progression.
    """
    # --- SETUP ---
    # Start with a diverse pool of potential patterns
    p_surface = CompositePattern(id="surface_cue", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"S1"})
    p_relational = CompositePattern(id="rel_abstract", level=3, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"R1"})
    p_superstitious = CompositePattern(id="superstition", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"SP1"})
    
    rule = MetaPatternRule(learning_rate=0.5, density_threshold=0.8, stability_kappa=0.2)
    system = HPMSystem([p_surface, p_relational, p_superstitious], rule)
    
    # --- PHASE 1: NOVICE STABILIZATION (Section 7.4.1) ---
    # Stimulus: Stable surface cues, but the relational structure is not yet obvious.
    for _ in range(50):
        # Surface is clear (0.1 loss), Relational is noisy (1.0 loss)
        system.step(surface_loss=np.array([0.1, 1.0, 0.5]), 
                    structural_loss=np.array([0.0, 1.0, 0.0]))
    
    # DRIFT CHECK: Level 1 should dominate. Level 3 should be gated.
    assert p_surface.weight > p_relational.weight, "Drift: Level 3 emerged without Level 1 foundation."
    
    # RESET WEIGHTS: Once foundation is stable, relational patterns 'emerge' (Section 7.4.3)
    # This prevents patterns from being permanently 'dead' from the maturation gate
    p_surface.weight = 0.5
    p_relational.weight = 0.5
    p_superstitious.weight = 0.0
    # ENSURE FOUNDATION: Surface must be dense to open the gate for L3
    p_surface.affective_score = 1.0
    p_surface.social_score = 1.0
    
    # --- PHASE 2: SKILL INTERNALIZATION (Section 2.5.1) ---
    # Reward the surface pattern to drive it into a procedural routine
    p_surface.affective_score = 1.0
    p_surface.social_score = 1.0
    # Step just once to catch the first transition
    system.step(surface_loss=np.array([0.1, 1.0, 0.5]))
        
    # FIDELITY CHECK: Verify shift to INTERNAL_PROC
    assert p_surface.substrate_id == SubstrateID.INTERNAL_PROC, "Gap: Pattern failed to internalize into procedural routine."
    
    # Run a few more steps to allow externalization for future phases
    for _ in range(9):
        system.step(surface_loss=np.array([0.1, 1.0, 0.5]))
    
    # --- PHASE 3: RELATIONAL ABSTRACTION (Section 9.1) ---
    # Stimulus: Surface becomes noisy, but structure becomes stable.
    # Level 1 foundation is now dense (stabilized), so Level 3 gate is open.
    for _ in range(100):
        # High surface noise (2.0), clean structure (0.05)
        system.step(surface_loss=np.array([2.0, 2.0, 2.0]), 
                    structural_loss=np.array([0.0, 0.05, 0.0]))
        
    # FIDELITY CHECK: Level 3 should now surpass Level 1 due to structural sensitivity
    assert p_relational.total_score > p_surface.total_score, "Drift: System values surface over structure in expert phase."
    assert p_relational.weight > p_surface.weight, "Drift: Relational patterns failed to displace noisy surface patterns."

    # --- PHASE 4: INSTITUTIONAL GATEKEEPING (Section 5.3 & 9.7) ---
    # Make the superstition "sticky" via social frequency (Media amplification)
    # But pass it through an Institutional (Scientific) Field.
    sci_field = InstitutionalField(replication_threshold=0.6)
    for _ in range(50):
        # Superstition has high social freq but high accuracy variance
        social_freq = np.array([0.1, 0.5, 1.0])
        curr_losses = [0.1, 0.05, np.random.uniform(0, 5)] # Random/Non-replicable loss for superstition
        
        # Apply science filter
        mult = sci_field.apply_filter(p_superstitious, curr_losses[2])
        p_superstitious.social_score = mult * 1.5 # Boosted by field, then filtered
        
        system.step(surface_loss=np.array(curr_losses), social_freq=social_freq)
        
    # FIDELITY CHECK: Science should have pruned the superstition despite its high frequency
    assert p_relational.weight > p_superstitious.weight, "Gap: Institutional fields failed to prune superstitious density."

    # --- PHASE 5: EXPERT RECOMBINATION (Section 7.4.5) ---
    # Final check: Level 5 emergence via recombination
    p_expert = CompositePattern(id="expert_base", level=5, substrate_id=SubstrateID.EXTERNAL_SYM, constituent_features={"CORE"})
    p_new_data = CompositePattern(id="new_motif", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"EXT"})
    
    recombinator = StructuralRecombinator()
    p_star = recombinator.recombine(p_expert, p_new_data)
    
    # FIDELITY CHECK: Level 5 contribution results in Generative Utility
    assert p_expert.generative_utility > 0, "Gap: Expert patterns not rewarded for generative contributions."
    assert p_star.level == 5, "Drift: Recombination did not preserve expert level."

    print("\n[HPM E2E SUCCESS] Codebase aligned to HPM v1.25 trajectories.")
