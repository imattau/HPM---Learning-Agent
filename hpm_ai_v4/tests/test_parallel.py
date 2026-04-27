# hpm_ai_v4/tests/test_parallel.py
import numpy as np
import pytest
from hpm_ai_v4.operators.parallel import pattern_worker

def make_state_dict(complexity=1, latent_dim=2, obs_dim=2):
    K, D = latent_dim, obs_dim
    def rand_trans(r, c):
        m = np.random.dirichlet(np.ones(c), size=r)
        return m
    d = {
        'pattern_id': 0,
        'complexity': complexity,
        'latent_dim': K,
        'obs_dim': D,
        'B':   rand_trans(K, D),
        'running_loss': 0.5,
        'weight': 0.2,
    }
    if latent_dim > 1:
        d['A'] = rand_trans(K, K)
        d['pi'] = np.random.dirichlet(np.ones(K))
    return d

def make_params(pattern_id=0):
    return {
        'learning_rate': 0.02,
        'lambda_l': 0.1,
        'adapt_window': 10,
        'beta_aff': 0.4,
        'gamma_soc': 0.3,
        'external_soc': 0.5,
    }

def test_worker_returns_required_keys():
    np.random.seed(42)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.3}
    state = make_state_dict()
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    required = {'pattern_id', 'A', 'B', 'pi',
                'running_loss', 'ep_score', 'aff_score',
                'soc_score', 'total_score'}
    assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

def test_worker_scores_are_finite():
    np.random.seed(7)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.2}
    state = make_state_dict()
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    for key in ('ep_score', 'aff_score', 'soc_score', 'total_score'):
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

def test_worker_flat_pattern():
    np.random.seed(3)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {0: 0.1}
    state = make_state_dict(complexity=1, latent_dim=1, obs_dim=2)
    params = make_params()
    result = pattern_worker(state, obs_buffer, field_freq, params)
    required = {'pattern_id', 'running_loss', 'total_score'}
    assert required.issubset(result.keys())

from hpm_ai_v4.operators.parallel import ParallelPatternPool
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern

def make_pattern_population(n=4, obs_dim=2):
    patterns = []
    for i in range(n - 1):
        p = HierarchicalPattern(pattern_id=i, latent_dim=2, obs_dim=obs_dim)
        p.weight = 1.0 / n
        patterns.append(p)
    flat = FlatPattern(pattern_id=n - 1, obs_dim=obs_dim)
    flat.weight = 1.0 / n
    patterns.append(flat)
    return patterns

def test_pool_map_returns_n_results():
    np.random.seed(1)
    patterns = make_pattern_population(n=4)
    obs_buffer = list(np.random.randint(0, 2, size=20))
    field_freq = {p.id: 0.25 for p in patterns}
    params = {'learning_rate': 0.02, 'lambda_l': 0.1, 'adapt_window': 10,
              'beta_aff': 0.4, 'gamma_soc': 0.3}
    pool = ParallelPatternPool(num_workers=1)
    results = pool.map_patterns(patterns, obs_buffer, field_freq, params)
    assert len(results) == 4
    pool.close()

def test_pool_sequential_no_subprocess():
    """num_workers=1 must not spawn a pool (checked via _pool attribute)."""
    pool = ParallelPatternPool(num_workers=1)
    assert pool._pool is None
    pool.close()

def test_pool_result_ids_match_input_order():
    np.random.seed(2)
    patterns = make_pattern_population(n=3)
    obs_buffer = list(np.random.randint(0, 2, size=15))
    field_freq = {}
    params = {'learning_rate': 0.02, 'lambda_l': 0.1, 'adapt_window': 10,
              'beta_aff': 0.4, 'gamma_soc': 0.3}
    pool = ParallelPatternPool(num_workers=1)
    results = pool.map_patterns(patterns, obs_buffer, field_freq, params)
    for p, r in zip(patterns, results):
        assert p.id == r['pattern_id']
    pool.close()


def test_pool_map_truncates_obs_buffer(monkeypatch):
    import hpm_ai_v4.operators.parallel as parallel_module

    np.random.seed(4)
    patterns = make_pattern_population(n=3)
    obs_buffer = list(np.random.randint(0, 2, size=120))
    field_freq = {}
    params = {'learning_rate': 0.02, 'lambda_l': 0.1, 'adapt_window': 15,
              'beta_aff': 0.4, 'gamma_soc': 0.3}
    seen = {}
    original_worker = parallel_module.pattern_worker

    def fake_pattern_worker(state_dict, obs_buffer_arg, field_freq_arg, params_arg):
        seen["len"] = len(obs_buffer_arg)
        return original_worker(state_dict, obs_buffer_arg, field_freq_arg, params_arg)

    monkeypatch.setattr(parallel_module, "pattern_worker", fake_pattern_worker)
    pool = ParallelPatternPool(num_workers=1)
    results = pool.map_patterns(patterns, obs_buffer, field_freq, params)
    assert len(results) == 3
    assert seen["len"] == 30
    pool.close()

from hpm_ai_v4.agents.agent import HPMAgent

def test_agent_parallel_no_error():
    """HPMAgent with num_workers=2 should run without error."""
    np.random.seed(99)
    agent = HPMAgent(num_initial_patterns=3, obs_dim=2, num_workers=2)
    for _ in range(5):
        agent.perceive_and_learn(int(np.random.randint(0, 2)))
    agent._pool.close()

def test_agent_weights_sum_to_one():
    """Pattern weights must sum to ~1.0 after parallel update."""
    np.random.seed(11)
    agent = HPMAgent(num_initial_patterns=4, obs_dim=2, num_workers=1)
    for _ in range(10):
        agent.perceive_and_learn(int(np.random.randint(0, 2)))
    weights = sum(p.weight for p in agent.patterns)
    assert abs(weights - 1.0) < 1e-6, f"Weights sum to {weights}, not 1.0"
