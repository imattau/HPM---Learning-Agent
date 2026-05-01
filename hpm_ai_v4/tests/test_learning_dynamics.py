import numpy as np
import pytest
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.evaluators.metrics import compression_gate
from hpm_ai_v4.field import PatternField
from hpm_ai_v4.operators.dynamics import meta_pattern_update

class FixedEnvironment:
    """A simple environment with a fixed structure: alternating 0s and 1s."""
    def __init__(self):
        self.step_count = 0
    def step(self):
        obs = self.step_count % 2
        self.step_count += 1
        return obs

def test_within_pattern_learning_convergence():
    """Verify that hierarchical patterns can converge towards environmental dynamics."""
    env = FixedEnvironment()
    target_seq = [0, 1, 0, 1, 0, 1, 0, 1]
    
    # Try 5 different random initializations (like an HPM agent pool)
    patterns = [HierarchicalPattern(pattern_id=i, latent_dim=2, obs_dim=2) for i in range(5)]
    
    for p in patterns:
        env.step_count = 0 # Reset env for each pattern
        buffer = []
        for step in range(300):
            obs = env.step()
            buffer.append(obs)
            if len(buffer) >= 5:
                p.update_parameters_online(buffer)
                
    best_ll = max(p.log_likelihood(target_seq) for p in patterns)
    
    print(f"\nMulti-Pattern Convergence Test:")
    print(f"  Best Final LL: {best_ll:.2f}")
    
    # For a perfect model of [0, 1, 0, 1...], LL would be 0.
    # Random model LL would be ~ -0.69 * 8 = -5.5
    assert best_ll > -5.0, "Patterns should improve significantly from random initialization"

def test_adaptive_vs_static_performance():
    """Verify that an agent with within-pattern learning adapts faster than one without."""
    env = FixedEnvironment()
    agent_adaptive = HPMAgent(num_initial_patterns=3)
    
    agent_static = HPMAgent(num_initial_patterns=3)
    # Mock update_parameters_online to be static
    for p in agent_static.patterns:
        if hasattr(p, 'update_parameters_online'):
            p.update_parameters_online = lambda obs, window_size=0: None
        if hasattr(p, 'observe'):
            p.observe = lambda obs, learning_rate=0: None
        
    # Run both for 100 steps (more steps for better signal)
    for _ in range(100):
        obs = env.step()
        agent_adaptive.perceive_and_learn(obs)
        agent_static.perceive_and_learn(obs)
        
    test_seq = [0, 1, 0, 1, 0, 1, 0, 1]
    
    adaptive_ll = np.mean([p.log_likelihood(test_seq) for p in agent_adaptive.patterns])
    static_ll = np.mean([p.log_likelihood(test_seq) for p in agent_static.patterns])
    
    print(f"\nAdaptive vs Static Test:")
    print(f"  Adaptive LL: {adaptive_ll:.2f}")
    print(f"  Static LL: {static_ll:.2f}")
    
    assert adaptive_ll > static_ll, "Adaptive agent should outperform static agent"


def test_pattern_update_recovers_from_nan_parameters():
    p = HierarchicalPattern(pattern_id=99, latent_dim=2, obs_dim=2)
    p.A[:] = np.nan
    p.B[:] = np.nan
    p.pi[:] = np.nan

    ll = p.update_parameters_online([0, 1, 0, 1, 0, 1])
    dist = p.predict_next_distribution([0, 1, 0, 1])

    assert np.isfinite(ll)
    assert np.all(np.isfinite(p.A))
    assert np.all(np.isfinite(p.B))
    assert np.all(np.isfinite(p.pi))
    assert np.all(np.isfinite(dist))
    assert abs(dist.sum() - 1.0) < 1e-6


def test_compression_gate_activates_earlier():
    p = HierarchicalPattern(pattern_id=100, latent_dim=2, obs_dim=2)
    p.running_loss = 0.3
    low_gate = compression_gate(p)
    p.running_loss = 0.9
    high_gate = compression_gate(p)

    assert low_gate > high_gate
    assert low_gate > 0.4


def test_meta_pattern_update_prefers_dense_patterns():
    dense = HierarchicalPattern(pattern_id=1, latent_dim=2, obs_dim=2)
    sparse = HierarchicalPattern(pattern_id=2, latent_dim=2, obs_dim=2)
    dense.weight = 0.5
    sparse.weight = 0.5
    dense.density_at_save = 1.0
    sparse.density_at_save = 0.1
    dense.A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    sparse.A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    dense.B = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
    sparse.B = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
    dense.pi = np.array([0.6, 0.4], dtype=np.float32)
    sparse.pi = np.array([0.6, 0.4], dtype=np.float32)
    dense._refresh_log_cache()
    sparse._refresh_log_cache()

    meta_pattern_update([dense, sparse], {1: 1.0, 2: 1.0}, eta=0.1, beta_c=0.0, decay=0.0)

    assert dense.weight > sparse.weight


def _stage_pattern(pattern_id: int, running_loss: float) -> HierarchicalPattern:
    p = HierarchicalPattern(pattern_id=pattern_id, latent_dim=2, obs_dim=2)
    p.A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    p.B = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=np.float32)
    p.pi = np.array([0.6, 0.4], dtype=np.float32)
    p.running_loss = running_loss
    p.weight = 1.0
    p._refresh_log_cache()
    return p


def test_developmental_stage_advances_on_saturated_evaluator_signal():
    agent = HPMAgent(num_initial_patterns=1)
    patterns = [_stage_pattern(i, 0.12) for i in range(5)]

    for step in range(30, 34):
        agent.development.update(patterns, global_step=step)

    assert agent.development.level_idx >= 1
    assert agent.development.level in {"local", "relational", "abstract", "generative"}


def test_developmental_stage_holds_back_when_evaluator_signal_is_noisy():
    agent = HPMAgent(num_initial_patterns=1)
    patterns = [
        _stage_pattern(0, 0.05),
        _stage_pattern(1, 0.95),
        _stage_pattern(2, 0.10),
        _stage_pattern(3, 0.90),
        _stage_pattern(4, 0.20),
    ]

    for step in range(30, 34):
        agent.development.update(patterns, global_step=step)

    assert agent.development.level_idx == 0


def test_density_weight_adapts_to_loss_trend():
    dense = HierarchicalPattern(pattern_id=1, latent_dim=2, obs_dim=2)
    sparse = HierarchicalPattern(pattern_id=2, latent_dim=2, obs_dim=2)
    dense.weight = 0.5
    sparse.weight = 0.5
    dense.density_at_save = 1.0
    sparse.density_at_save = 0.1

    rising_state = {"density_weight": 0.2}
    for loss in [0.50, 0.60, 0.70, 0.80]:
        meta_pattern_update(
            [dense, sparse],
            {1: 1.0, 2: 1.0},
            eta=0.0,
            beta_c=0.0,
            decay=0.0,
            density_state=rising_state,
            mean_running_loss=loss,
        )

    falling_state = {"density_weight": 0.2}
    for loss in [0.80, 0.70, 0.60, 0.50]:
        meta_pattern_update(
            [dense, sparse],
            {1: 1.0, 2: 1.0},
            eta=0.0,
            beta_c=0.0,
            decay=0.0,
            density_state=falling_state,
            mean_running_loss=loss,
        )

    assert rising_state["density_weight"] < 0.2
    assert falling_state["density_weight"] > 0.2


def test_pattern_field_biases_toward_generalising_patterns():
    field = PatternField()
    generalising = HierarchicalPattern(pattern_id=1, latent_dim=2, obs_dim=2)
    overfit = HierarchicalPattern(pattern_id=2, latent_dim=2, obs_dim=2)
    generalising.weight = 0.5
    overfit.weight = 0.5

    field.update(
        [generalising, overfit],
        episode_stats={
            1: {"train_ll": -8.0, "holdout_ll": -2.0, "chunk_size": 8},
            2: {"train_ll": -2.0, "holdout_ll": -8.0, "chunk_size": 8},
        },
    )

    assert field.affinity_for(generalising) > field.affinity_for(overfit)
    assert field.frequencies[1] > field.frequencies[2]
