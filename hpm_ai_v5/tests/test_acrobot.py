"""Tests for Acrobot benchmark."""

import pytest
import numpy as np
from hpm_ai_v5.planning.acrobot import AcrobotEnv, AcrobotBenchmark
from hpm_ai_v5.adapter.physics import AcrobotStateAdapter
from hpm_ai_v5.adapter.packet import AdapterPacket

def test_acrobot_env_reset():
    env = AcrobotEnv()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "theta1" in obs
    assert "theta2" in obs
    assert "theta1_dot" in obs
    assert "theta2_dot" in obs

def test_acrobot_env_step():
    env = AcrobotEnv()
    env.reset()
    obs, reward, done = env.step(1.0)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_acrobot_state_adapter():
    adapter = AcrobotStateAdapter()
    obs = {"theta1": 0.1, "theta2": -0.1, "theta1_dot": 0.5, "theta2_dot": -0.5}
    packet = AdapterPacket(raw=obs, context={"last_action": 1.0})
    packet = adapter.run(packet)
    
    assert len(packet.states) == 1
    state_val = packet.states[0].value
    # 3 actions + 4 sin/cos + 2 vels + 1 error = 10
    assert len(state_val) == 10
    assert packet.context["task"] == "acrobot"

def test_acrobot_benchmark_init():
    benchmark = AcrobotBenchmark()
    assert benchmark.env is not None
    assert benchmark.pipeline is not None

def test_acrobot_smoke_run():
    # Very short run to ensure no crashes
    benchmark = AcrobotBenchmark()
    res = benchmark.run(episodes=2, max_steps=10)
    assert res.total_episodes == 2
    assert len(res.episode_lengths) == 2
