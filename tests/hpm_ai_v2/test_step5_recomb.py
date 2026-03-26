import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.recombination import StructuralRecombinator

def test_recombination_merging():
    """
    Verify that merging two patterns results in a new pattern
    with combined constituent features.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A", "B"})
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"C", "D"})
    
    recombinator = StructuralRecombinator()
    p_star = recombinator.recombine(p1, p2)
    
    assert p_star is not None
    assert p_star.constituent_features == {"A", "B", "C", "D"}
    assert p_star.id == "recomb_p1_p2"
    assert p_star.level == 1
    
def test_insight_calculation():
    """
    Verify that higher novelty results in a higher insight score.
    """
    # High novelty (no common features)
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A"})
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"B"})
    
    recombinator = StructuralRecombinator()
    p_star_novel = recombinator.recombine(p1, p2)
    insight_novel = p_star_novel.affective_score
    
    # Low novelty (some common features)
    p3 = CompositePattern(id="p3", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A", "B"})
    p4 = CompositePattern(id="p4", level=1, substrate_id=SubstrateID.INTERNAL_FLEX, constituent_features={"A", "C"})
    p_star_redundant = recombinator.recombine(p3, p4)
    insight_redundant = p_star_redundant.affective_score
    
    assert insight_novel > insight_redundant
    assert p_star_novel.weight > p_star_redundant.weight
