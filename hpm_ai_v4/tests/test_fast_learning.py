import numpy as np
import pytest
from hpm_ai_v4.pattern import HierarchicalPattern


class TestFloat32Storage:
    def test_A_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.A.dtype == np.float32

    def test_B_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.B.dtype == np.float32

    def test_pi_is_float32(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.pi.dtype == np.float32

    def test_logA_shape(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA.shape == (2, 2)

    def test_logA_dtype(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert p.logA.dtype == np.float32

    def test_logA_consistent_with_A(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logA), p.A, atol=1e-5)

    def test_logB_consistent_with_B(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        assert np.allclose(np.exp(p.logB), p.B, atol=1e-5)

    def test_log_cache_refreshed_after_update(self):
        p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        obs_seq = [0, 1, 2, 1, 0, 1, 2, 3, 4, 0] * 5
        p.update_parameters_online(obs_seq)
        assert np.allclose(np.exp(p.logA), p.A, atol=1e-4)


class TestOnlineUpdate:
    def setup_method(self):
        np.random.seed(42)
        self.p = HierarchicalPattern(1, latent_dim=2, obs_dim=5)
        self.obs_seq = [0, 1, 2, 1, 0, 3, 4, 2, 1, 0] * 10  # length 100

    def test_runs_without_error(self):
        self.p.update_parameters_online(self.obs_seq)

    def test_A_rows_sum_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.A.sum(axis=1), 1.0, atol=1e-5)

    def test_B_rows_sum_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.B.sum(axis=1), 1.0, atol=1e-5)

    def test_pi_sums_to_one(self):
        self.p.update_parameters_online(self.obs_seq)
        assert np.allclose(self.p.pi.sum(), 1.0, atol=1e-5)

    def test_log_likelihood_finite_after_updates(self):
        for _ in range(10):
            self.p.update_parameters_online(self.obs_seq)
        ll = self.p.log_likelihood(self.obs_seq)
        assert np.isfinite(ll)

    def test_running_loss_updates(self):
        initial = self.p.running_loss
        self.p.update_running_loss(self.obs_seq[:10])
        assert self.p.running_loss != initial


class TestAgentFastPath:
    def setup_method(self):
        np.random.seed(7)
        from hpm_ai_v4.agents.agent import HPMAgent
        self.agent = HPMAgent(num_initial_patterns=3, obs_dim=5)

    def test_perceive_and_learn_runs(self):
        self.agent.perceive_and_learn(0)

    def test_200_steps_without_error(self):
        for i in range(200):
            self.agent.perceive_and_learn(i % 5)

    def test_pattern_weights_positive_after_200_steps(self):
        for i in range(200):
            self.agent.perceive_and_learn(i % 5)
        weights = np.array([p.weight for p in self.agent.patterns])
        assert weights.sum() > 0.1
