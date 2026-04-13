import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.density import PatternDensityTracker
from hfn.affective import AffectiveEvaluator, AffectiveState

def test_observer_density_integration():
    D = 2
    forest = Forest(D=D)
    # Register a prior to have something to match
    node = HFN(mu=np.zeros(D), sigma=np.ones(D), id="node_0", use_diag=True)
    forest.register(node)
    
    obs = Observer(forest, use_density_tracker=True)
    assert obs.density_tracker is not None
    
    # Observe a point close to node_0
    obs.observe(np.array([0.1, 0.1]))
    
    # Check that density tracker has state for node_0
    dens = obs.density_tracker.get_density("node_0")
    assert dens is not None
    c, e, f, total = dens
    # e and f should be > 0 due to updates in _update_weights
    assert e > 0
    assert f > 0

def test_observer_affective_integration():
    D = 2
    forest = Forest(D=D)
    node = HFN(mu=np.zeros(D), sigma=np.ones(D), id="node_0", use_diag=True)
    forest.register(node)
    
    obs = Observer(forest, use_affective_evaluator=True)
    assert isinstance(obs.evaluator, AffectiveEvaluator)
    
    # Start neutral
    _, _, state = obs.evaluator._get_global_state()
    assert state == AffectiveState.NEUTRAL
    
    # Observer uses the evaluator
    # Success update should happen if we match node_0
    obs.observe(np.array([0.0, 0.0]))
    
    # State might still be neutral but Valence should have increased
    a, v, s = obs.evaluator._get_global_state()
    assert v > 0.5 # valence should have increased from initial 0.5
    
def test_observer_node_creation_density():
    D = 2
    forest = Forest(D=D)
    obs = Observer(forest, use_density_tracker=True, tau=0.1) # low tau to force creation
    
    # Observe something surprising to trigger node creation
    obs.observe(np.array([10.0, 10.0]))
    
    # Find the created node
    new_nodes = [n for n in forest.active_nodes() if n.id.startswith("leaf_")]
    assert len(new_nodes) > 0
    new_node_id = new_nodes[0].id
    
    # Check that density tracker has connectivity for it
    dens = obs.density_tracker.get_density(new_node_id)
    assert dens is not None
    assert dens[0] > 0 # connectivity should be initialized
