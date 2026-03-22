"""
Integration test: 3-agent ensemble (Gaussian + Laplace + Categorical).

Validates that:
- A 3-agent ensemble constructs without error
- Categorical agent receives integer-encoded observations; others receive float vectors
- orch.step() runs 10 steps without error
- ensemble_score returns finite values for both float and integer candidate vectors
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern
from hpm.patterns.categorical import CategoricalPattern
from hpm.patterns.factory import make_pattern
from hpm.store.memory import InMemoryStore


# Categorical agent uses D dimensions over K symbols
CAT_D = 10   # number of categorical dimensions (e.g. flattened grid cells)
CAT_K = 10   # alphabet size (ARC colours 0-9)

# Continuous agents use float feature vectors
FLOAT_D = 16


def make_gaussian_agent(agent_id: str, field=None) -> Agent:
    config = AgentConfig(
        agent_id=agent_id,
        feature_dim=FLOAT_D,
        pattern_type="gaussian",
        init_sigma=1.0,
        gamma_soc=0.0,
        T_recomb=1000,   # suppress recombination during test
    )
    return Agent(config, field=field)


def make_laplace_agent(agent_id: str, field=None) -> Agent:
    config = AgentConfig(
        agent_id=agent_id,
        feature_dim=FLOAT_D,
        pattern_type="laplace",
        init_sigma=1.0,
        gamma_soc=0.0,
        T_recomb=1000,
    )
    return Agent(config, field=field)


def make_categorical_agent(agent_id: str, D: int = CAT_D, K: int = CAT_K) -> Agent:
    """Create a categorical agent. No field — cannot share patterns with continuous agents."""
    config = AgentConfig(
        agent_id=agent_id,
        feature_dim=D,
        pattern_type="categorical",
        alphabet_size=K,
        gamma_soc=0.0,
        T_recomb=1000,
    )
    return Agent(config)


class TestCategoricalEnsembleConstruction:
    def test_three_agents_construct_without_error(self):
        gauss = make_gaussian_agent("gauss_a")
        laplace = make_laplace_agent("laplace_b")
        cat = make_categorical_agent("cat_c")
        assert gauss is not None
        assert laplace is not None
        assert cat is not None

    def test_categorical_agent_has_categorical_pattern_in_store(self):
        cat = make_categorical_agent("cat_c")
        records = cat.store.query("cat_c")
        assert len(records) > 0
        p, _ = records[0]
        assert isinstance(p, CategoricalPattern)

    def test_gaussian_agent_has_gaussian_pattern_in_store(self):
        gauss = make_gaussian_agent("gauss_a")
        records = gauss.store.query("gauss_a")
        assert len(records) > 0
        p, _ = records[0]
        assert isinstance(p, GaussianPattern)

    def test_laplace_agent_has_laplace_pattern_in_store(self):
        laplace = make_laplace_agent("laplace_b")
        records = laplace.store.query("laplace_b")
        assert len(records) > 0
        p, _ = records[0]
        assert isinstance(p, LaplacePattern)

    def test_categorical_agent_pattern_has_correct_shape(self):
        cat = make_categorical_agent("cat_c", D=CAT_D, K=CAT_K)
        records = cat.store.query("cat_c")
        p, _ = records[0]
        assert p.probs.shape == (CAT_D, CAT_K)
        assert p.K == CAT_K


class TestCategoricalAgentStep:
    def test_categorical_agent_step_no_error(self):
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(42)
        for _ in range(10):
            obs = rng.integers(0, CAT_K, size=CAT_D)
            result = cat.step(obs)
            assert isinstance(result, dict)
            assert result['t'] >= 1

    def test_continuous_agents_step_no_error(self):
        gauss = make_gaussian_agent("gauss_a")
        laplace = make_laplace_agent("laplace_b")
        rng = np.random.default_rng(7)
        for _ in range(10):
            obs_float = rng.standard_normal(FLOAT_D)
            gauss.step(obs_float)
            laplace.step(obs_float)

    def test_10_steps_mixed_ensemble_no_error(self):
        """All 3 agents run 10 steps receiving their respective observation types."""
        gauss = make_gaussian_agent("gauss_a")
        laplace = make_laplace_agent("laplace_b")
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(123)

        for _ in range(10):
            obs_float = rng.standard_normal(FLOAT_D)
            obs_int = rng.integers(0, CAT_K, size=CAT_D)
            gauss.step(obs_float)
            laplace.step(obs_float)
            cat.step(obs_int)

        # All agents should have surviving patterns
        assert len(gauss.store.query("gauss_a")) > 0
        assert len(laplace.store.query("laplace_b")) > 0
        assert len(cat.store.query("cat_c")) > 0

    def test_step_returns_finite_accuracy(self):
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(0)
        obs = rng.integers(0, CAT_K, size=CAT_D)
        result = cat.step(obs)
        assert np.isfinite(result['mean_accuracy'])


class TestEnsembleScore:
    def _ensemble_score(self, agents, vec):
        """Compute weighted NLL ensemble score (same convention as benchmarks)."""
        total = 0.0
        any_records = False
        for agent in agents:
            records = agent.store.query(agent.agent_id)
            if records:
                any_records = True
                for p, w in records:
                    total += w * p.log_prob(vec)
        return total if any_records else 0.0

    def test_ensemble_score_finite_for_float_agents(self):
        gauss = make_gaussian_agent("gauss_a")
        laplace = make_laplace_agent("laplace_b")
        rng = np.random.default_rng(42)
        # Train for a few steps
        for _ in range(5):
            gauss.step(rng.standard_normal(FLOAT_D))
            laplace.step(rng.standard_normal(FLOAT_D))
        # Score a candidate vector
        candidate = rng.standard_normal(FLOAT_D)
        score = self._ensemble_score([gauss, laplace], candidate)
        assert np.isfinite(score)

    def test_ensemble_score_finite_for_categorical_agent(self):
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(42)
        for _ in range(5):
            cat.step(rng.integers(0, CAT_K, size=CAT_D))
        # Score an integer candidate
        candidate_int = rng.integers(0, CAT_K, size=CAT_D)
        score = self._ensemble_score([cat], candidate_int)
        assert np.isfinite(score)

    def test_categorical_lower_nll_for_trained_symbol(self):
        """After training on symbol 0 repeatedly, NLL for all-0 vector < all-9 vector."""
        cat = make_categorical_agent("cat_c", D=5, K=10)
        x_trained = np.zeros(5, dtype=int)
        for _ in range(30):
            cat.step(x_trained)
        x_untrained = np.full(5, 9, dtype=int)
        nll_trained = self._ensemble_score([cat], x_trained)
        nll_untrained = self._ensemble_score([cat], x_untrained)
        # Trained vector should have lower NLL (more probable)
        assert nll_trained < nll_untrained

    def test_ensemble_score_separate_per_type(self):
        """Float ensemble and categorical ensemble each return finite values."""
        gauss = make_gaussian_agent("gauss_a")
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(99)
        for _ in range(5):
            gauss.step(rng.standard_normal(FLOAT_D))
            cat.step(rng.integers(0, CAT_K, size=CAT_D))

        float_candidate = rng.standard_normal(FLOAT_D)
        int_candidate = rng.integers(0, CAT_K, size=CAT_D)

        float_score = self._ensemble_score([gauss], float_candidate)
        int_score = self._ensemble_score([cat], int_candidate)

        assert np.isfinite(float_score)
        assert np.isfinite(int_score)


class TestCategoricalPatternProtocol:
    def test_no_sigma_attribute_triggers_mc_kl_branch(self):
        """Categorical patterns have no sigma; MetaPatternRule uses MC KL branch."""
        cat_pattern = CategoricalPattern(np.ones((3, 4)) / 4, K=4)
        assert not hasattr(cat_pattern, 'sigma')

    def test_categorical_sample_for_mc_kl(self):
        """sample() required by MC KL branch returns correct shape and dtype."""
        cat_pattern = CategoricalPattern(np.ones((3, 4)) / 4, K=4)
        rng = np.random.default_rng(0)
        samples = cat_pattern.sample(50, rng)
        assert samples.shape == (50, 3)
        assert np.issubdtype(samples.dtype, np.integer)
        assert np.all(samples >= 0) and np.all(samples < 4)

    def test_is_structurally_valid_after_training(self):
        cat = make_categorical_agent("cat_c")
        rng = np.random.default_rng(5)
        for _ in range(10):
            cat.step(rng.integers(0, CAT_K, size=CAT_D))
        records = cat.store.query("cat_c")
        for p, _ in records:
            assert p.is_structurally_valid()
