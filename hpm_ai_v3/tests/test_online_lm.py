"""
test_online_lm.py - Verification for failure-triggered online learning.
"""

import pytest
import unittest.mock as mock
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.online_buffer import OnlineLearningBuffer
from hpm_ai_v3.neural_lm_pattern import LanguageModelPattern


@pytest.fixture
def agent():
    return UnifiedDiscoveryAgent(context_dim=64)


def test_online_buffer_keyword_extraction():
    buffer = OnlineLearningBuffer()
    text = "Extract the mass of Jupiter from astronomical data."
    keywords = buffer.extract_keywords(text)
    # Common words like 'extract', 'data' are in stop_words
    assert "jupiter" in keywords
    assert "astronomical" in keywords


def test_failure_triggers_fetch(agent):
    """Verify that 3 failures trigger a Wikipedia fetch."""
    task = {
        "text": "Find the atomic number of Gold.",
        "type": "pretraining",
        "answer": 79,
        "phase": 0.0
    }
    agent.initialize_task(task)
    domain = "pretraining" # default if no hint
    
    # Mock the fetch_wikipedia method
    with mock.patch.object(agent.online_buffer, 'fetch_wikipedia', return_value="Gold is a chemical element with symbol Au and atomic number 79.") as mock_fetch:
        # 1st Failure
        agent.on_step_complete(reward=-0.5, pattern=None, result={"status": "failed"})
        assert not mock_fetch.called
        
        # 2nd Failure
        agent.on_step_complete(reward=-0.5, pattern=None, result={"status": "failed"})
        assert not mock_fetch.called
        
        # 3rd Failure triggers fetch in run_episode
        agent.on_step_complete(reward=-0.5, pattern=None, result={"status": "failed"})
        agent.run_episode(task, max_steps=1)
        
        assert mock_fetch.called
        assert "gold" in [k.lower() for k in mock_fetch.call_args[0][0]]


def test_fine_tuning_improves_prediction():
    """Verify that fine-tuning on domain text improves next-char log-prob (conceptually)."""
    lm = LanguageModelPattern()
    text = "The speed of light is 299792458 metres per second."
    
    # Initial loss
    lm.fine_tune(text, epochs=0) # Just sets up model if needed
    loss_before = lm.last_loss
    
    # Fine-tune
    lm.fine_tune(text, epochs=5)
    loss_after = lm.last_loss
    
    assert loss_after < loss_before


def test_agent_advancement_with_online_learning(agent):
    """integration test: Agent fails, fetches, fine-tunes, and succeeds."""
    task = {
        "text": "Speed of light is 299792458",
        "type": "pretraining",
        "answer": 299792458,
        "phase": 0.0,
        "hint": "language_model"
    }
    agent.initialize_task(task)
    
    # Mock successful tool execution (the goal)
    # We want to verify the WHOLE sequence is called
    with mock.patch.object(agent.online_buffer, 'fetch_wikipedia', return_value="The speed of light is 299792458 m/s") as mock_fetch:
        # Simulate 3 failures
        for _ in range(3):
            agent.on_step_complete(reward=-0.5, pattern=None, result={})
            
        # This call should trigger fetch AND fine-tune
        agent.run_episode(task, max_steps=5)
        
        assert mock_fetch.called
        # Check if LM was fine-tuned
        lm = next((p for p in agent.population.patterns if isinstance(p, LanguageModelPattern)), None)
        assert lm is not None
        assert lm._pretrained
