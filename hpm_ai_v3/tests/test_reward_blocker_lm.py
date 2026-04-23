"""Test LanguageModelPattern online update and log_prob cleanup."""
import pytest
import torch
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern


@pytest.fixture
def lm():
    return LanguageModelPattern(vocab_size=128, embed_dim=16, hidden_dim=32, n_layers=1)


def test_lm_log_prob_no_heuristic(lm):
    """log_prob must return verified reward or 0.0, no heuristics."""
    # With reward
    obs_reward = {"reward": torch.tensor([0.7])}
    assert lm.log_prob(obs_reward).item() == pytest.approx(0.7)
    
    # Without reward, with result (previously had heuristic)
    obs_result = {"result": [1, 2, 3]}
    assert lm.log_prob(obs_result).item() == 0.0
    
    # Empty
    assert lm.log_prob({}).item() == 0.0


def test_lm_update_parameters_positive_reward(lm):
    """update_parameters should trigger fine_tune if reward > 0."""
    obs = {
        "reward": torch.tensor([1.0]),
        "text": "The quick brown fox jumps over the lazy dog"
    }
    
    # Initial loss
    initial_loss = lm.last_loss
    
    # Run update
    lm.update_parameters(obs, learning_rate=1e-3)
    
    # Check that fine_tune was called (last_loss updated)
    assert lm.last_loss < float('inf')
    assert lm.loss_ema is not None


def test_lm_update_parameters_negative_reward(lm):
    """update_parameters should NOT trigger fine_tune if reward <= 0."""
    obs = {
        "reward": torch.tensor([-0.5]),
        "text": "The quick brown fox jumps over the lazy dog"
    }
    
    initial_loss = lm.last_loss
    lm.update_parameters(obs)
    
    assert lm.last_loss == initial_loss
    assert lm.loss_ema is None
