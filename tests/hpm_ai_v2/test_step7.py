import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID
from hpm_ai_v2.evaluators import InstitutionalField

def test_institutional_field_replication_filter():
    """
    Verify that InstitutionalField prunes (penalizes)
    patterns with inconsistent accuracy.
    """
    p1 = CompositePattern(id="p1", level=1, substrate_id=SubstrateID.EXTERNAL_SYM)
    p2 = CompositePattern(id="p2", level=1, substrate_id=SubstrateID.EXTERNAL_SYM)
    
    field = InstitutionalField(replication_threshold=0.5)
    
    # Pattern 1 is consistent and accurate (low loss)
    for _ in range(5):
        mult1 = field.apply_filter(p1, 0.1) # consistent loss
    
    # Pattern 2 is inconsistent (high variance)
    field.apply_filter(p2, 0.1)
    field.apply_filter(p2, 2.0)
    field.apply_filter(p2, 0.1)
    field.apply_filter(p2, 2.0)
    mult2 = field.apply_filter(p2, 0.1)
    
    assert mult1 == 1.0 # High replication score
    assert mult2 == 0.1 # Low replication score due to variance
    
def test_institutional_field_low_accuracy_filter():
    """
    Verify that patterns with consistently low accuracy (high loss)
    are also penalized.
    """
    p = CompositePattern(id="p", level=1, substrate_id=SubstrateID.EXTERNAL_SYM)
    field = InstitutionalField(replication_threshold=0.5)
    
    # Consistently high loss (e.g., superstitious belief in a hard domain)
    for _ in range(5):
        mult = field.apply_filter(p, 2.0)
        
    assert mult == 0.1 # High mean loss -> low replication score
