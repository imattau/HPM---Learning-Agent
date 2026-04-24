import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.pattern import HierarchicalPattern


def make_mock_pattern(compression_val, running_loss_val, latent_dim=2):
    p = MagicMock(spec=HierarchicalPattern)
    p.compression.return_value = compression_val
    p.running_loss = running_loss_val
    p.latent_dim = latent_dim
    p.weight = 0.5
    p.id = 0
    p.complexity = 3
    return p


def test_grow_called_when_conditions_met():
    """compression high + loss high + below max_K => grow_latent() called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=2.0, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=8, loss_threshold=1.0)

    pattern.grow_latent.assert_called_once()


def test_grow_not_called_when_compression_low():
    """compression low => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.1, running_loss_val=2.0, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=8, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_grow_not_called_when_loss_low():
    """loss low => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=0.5, latent_dim=2)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=8, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_grow_not_called_at_max_k():
    """pattern at max_K => grow_latent() NOT called."""
    agent = HPMAgent.__new__(HPMAgent)
    agent.obs_buffer = list(range(20))
    agent.step_counter = 0

    pattern = make_mock_pattern(compression_val=0.5, running_loss_val=2.0, latent_dim=8)
    agent.patterns = [pattern]

    agent._maybe_grow_patterns(max_K=8, loss_threshold=1.0)

    pattern.grow_latent.assert_not_called()


def test_growth_checked_every_500_steps(monkeypatch):
    """_maybe_grow_patterns is only called at step multiples of 500."""
    called_at = []

    original_maybe_grow = HPMAgent._maybe_grow_patterns

    def tracking_maybe_grow(self, **kwargs):
        called_at.append(self.step_counter)

    monkeypatch.setattr(HPMAgent, '_maybe_grow_patterns', tracking_maybe_grow)

    # Build a minimal agent that won't error on perceive_and_learn internals
    agent = HPMAgent(num_initial_patterns=1, obs_dim=2)

    # Run 1001 steps, track which steps trigger growth check
    # We patch the expensive parts
    with patch.object(agent._pool, 'map_patterns') as mock_map, \
         patch.object(agent, 'gossip_with_substrate'):
        
        def side_effect(patterns, obs_buffer, field_freq, params):
            return [
                {'pattern_id': p.id, 'A3': p.A3, 'A32': p.A32, 'A21': p.A21,
                 'B': p.B, 'pi3': p.pi3, 'SS_A3': p.SS_A3, 'SS_A32': p.SS_A32,
                 'SS_A21': p.SS_A21, 'SS_B': p.SS_B, 'running_loss': 0.0,
                 'total_score': 0.0}
                for p in patterns
            ]
        mock_map.side_effect = side_effect
        
        for i in range(1001):
            agent.perceive_and_learn(i % 2)

    # Growth should have been checked at steps 500 and 1000
    assert 500 in called_at
    assert 1000 in called_at
    # Should NOT be checked at step 1
    assert 1 not in called_at
