import tempfile
import time
import numpy as np
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.density import PatternDensityTracker


def test_density_tracker_basic():
    forest = Forest(D=2)
    observer = Observer(forest)
    tracker = PatternDensityTracker(observer)

    # Create a node
    node = HFN(mu=np.zeros(2), sigma=np.ones(2), id="test_node")
    observer.register(node)

    # Update structural connectivity
    tracker.update_structural_connectivity(node)
    res = tracker.get_density("test_node")
    assert res is not None
    c, e, f, total = res
    assert c > 0  # should have some connectivity (node is leaf, so low but >0)

    # Update evaluator reinforcement
    tracker.update_evaluator_reinforcement("test_node", success=True)
    res = tracker.get_density("test_node")
    assert res is not None
    c, e, f, total = res
    assert e > 0

    # Update field amplification
    tracker.update_field_amplification("test_node", time.time())
    res = tracker.get_density("test_node")
    assert res is not None
    c, e, f, total = res
    assert f > 0

    # Pruning decision
    should_prune = tracker.should_prune("test_node", threshold=0.3)
    # New node with low density but no epistemic loss yet → should not prune
    assert not should_prune


def test_density_pruning():
    forest = Forest(D=2)
    observer = Observer(forest)
    tracker = PatternDensityTracker(observer)

    node = HFN(mu=np.zeros(2), sigma=np.ones(2), id="weak_node")
    observer.register(node)
    
    # Initialize connectivity
    tracker.update_structural_connectivity(node)

    # Artificially lower weight and score
    state_node = observer._get_state("weak_node")
    state_node.mu[0] = 0.1  # weight
    state_node.mu[1] = 0.0  # score

    # Density will be low (0.1*0.4 = 0.04)
    should_prune = tracker.should_prune("weak_node", threshold=0.3)
    # Epistemic loss high (weight+score low), density low → prune
    assert should_prune
