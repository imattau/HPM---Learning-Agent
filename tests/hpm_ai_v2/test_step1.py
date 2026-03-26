import pytest
import numpy as np
from hpm_ai_v2.patterns import CompositePattern, SubstrateID, SUBSTRATE_REGISTRY
from hpm_ai_v2.distributions import GaussianDistribution, CategoricalDistribution

def test_substrate_registry():
    assert SubstrateID.INTERNAL_FLEX in SUBSTRATE_REGISTRY
    assert SubstrateID.INTERNAL_PROC in SUBSTRATE_REGISTRY
    assert SubstrateID.EXTERNAL_SYM in SUBSTRATE_REGISTRY
    
    flex_props = SUBSTRATE_REGISTRY[SubstrateID.INTERNAL_FLEX]
    proc_props = SUBSTRATE_REGISTRY[SubstrateID.INTERNAL_PROC]
    
    assert flex_props.resource_cost > proc_props.resource_cost
    assert flex_props.recombination_flexibility > proc_props.recombination_flexibility

def test_composite_pattern_init():
    pattern = CompositePattern(
        id="test_p1",
        level=1,
        substrate_id=SubstrateID.INTERNAL_FLEX,
        constituent_features={"feat1", "feat2"}
    )
    
    assert pattern.id == "test_p1"
    assert pattern.level == 1
    assert pattern.substrate_id == SubstrateID.INTERNAL_FLEX
    assert pattern.resource_cost == 1.0
    assert pattern.accuracy == 0.0

def test_connectivity_calculation():
    pattern = CompositePattern(
        id="test_p1",
        level=1,
        substrate_id=SubstrateID.INTERNAL_FLEX,
        constituent_features={"feat1", "feat2"}
    )
    
    # Fully connected directed graph (2 nodes, 2 edges: 1->2, 2->1)
    matrix = np.array([[0, 1], [1, 0]])
    conn = pattern.calculate_connectivity(matrix)
    assert conn == 1.0
    
    # Partially connected
    matrix_half = np.array([[0, 1], [0, 0]])
    conn_half = pattern.calculate_connectivity(matrix_half)
    assert conn_half == 0.5

def test_gaussian_distribution():
    mu = np.array([0.0, 0.0])
    sigma = np.eye(2)
    dist = GaussianDistribution(mu, sigma)
    
    # Log prob at the mean
    lp = dist.log_prob(np.array([0.0, 0.0]))
    # -0.5 * (2 * log(2pi) + log(1)) = -log(2pi)
    assert np.isclose(lp, -np.log(2 * np.pi))
    
    # Entropy
    ent = dist.entropy()
    # 0.5 * (2 * (1 + log(2pi)) + 0) = 1 + log(2pi)
    assert np.isclose(ent, 1 + np.log(2 * np.pi))

def test_categorical_distribution():
    probs = np.array([0.8, 0.2])
    dist = CategoricalDistribution(probs)
    
    assert np.isclose(dist.probs[0], 0.8)
    assert np.isclose(dist.probs[1], 0.2)
    
    lp0 = dist.log_prob(0)
    assert np.isclose(lp0, np.log(0.8))
    
    lp1 = dist.log_prob(1)
    assert np.isclose(lp1, np.log(0.2))
