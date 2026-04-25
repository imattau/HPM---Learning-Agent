# hpm_ai_v4/tests/test_layered_agent.py
import numpy as np
from hpm_ai_v4.simulations.layered_agent import LayeredAgent

def test_layered_agent_perceive_runs():
    agent = LayeredAgent(num_workers=1)
    for i in range(20):
        agent.perceive(i % 95)

def test_layered_agent_obs_dims():
    agent = LayeredAgent(num_workers=1)
    assert agent.l1.obs_dim == 5
    assert agent.l2.obs_dim == 95

def test_layered_agent_equal_weights():
    agent = LayeredAgent(num_workers=1)
    # Check that initial weights are as expected (not 1.0)
    assert max(p.weight for p in agent.l1.patterns) < 0.2
    assert max(p.weight for p in agent.l2.patterns) < 0.2

def test_layered_agent_generate_returns_string():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    assert isinstance(result, str)
    assert len(result) <= 20

def test_layered_agent_generate_printable():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    # 0-94 range corresponds to 32-126 ASCII
    assert all(32 <= ord(ch) <= 126 for ch in result)

def test_predict_next_chars_returns_list():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    # use a real context
    context = list(range(10))
    preds = agent.predict_next_chars(context, top_k=5)
    assert len(preds) <= 5
    assert all(isinstance(ch, str) and isinstance(prob, (float, np.float32, np.float64)) for ch, prob in preds)
