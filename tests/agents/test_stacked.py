import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.stacked import LevelConfig, StackedOrchestrator, make_stacked_orchestrator


def test_level_config_defaults():
    cfg = LevelConfig(n_agents=2)
    assert cfg.pattern_type == "gaussian"
    assert cfg.K == 1
    assert cfg.agent_ids is None


def test_make_stacked_orchestrator_dims_3level():
    configs = [
        LevelConfig(n_agents=2),
        LevelConfig(n_agents=2, K=3),
        LevelConfig(n_agents=1, K=3),
    ]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=64, level_configs=configs)
    assert all_agents[0][0].config.feature_dim == 64
    assert all_agents[1][0].config.feature_dim == 66
    assert all_agents[2][0].config.feature_dim == 68


def test_make_stacked_orchestrator_returns_correct_agent_counts():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=3, K=2)]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=8, level_configs=configs)
    assert len(all_agents[0]) == 2
    assert len(all_agents[1]) == 3


def test_stacked_orchestrator_level_agents_public():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=1, K=2)]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert len(orch.level_agents) == 2
    assert orch.level_agents[0] is all_agents[0]
    assert orch.level_agents[1] is all_agents[1]


def test_stacked_orchestrator_level_agents_shape():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=2, K=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert len(orch.level_agents) == 3
    assert len(orch.level_agents[0]) == 2
    assert len(orch.level_agents[1]) == 2
    assert len(orch.level_agents[2]) == 1


def test_stacked_orchestrator_level_ks():
    configs = [
        LevelConfig(n_agents=1, K=99),  # K=99 on L1 must be ignored
        LevelConfig(n_agents=1, K=3),
        LevelConfig(n_agents=1, K=5),
    ]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert orch.level_Ks[0] == 1
    assert orch.level_Ks[1] == 3
    assert orch.level_Ks[2] == 5


def test_stacked_orchestrator_ticks_initial():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert orch._level_ticks == [0, 0]


def test_cadence_k1_all_levels_fire_every_step():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = orch.step(rng.standard_normal(4))
        assert result["level2"] != {}


def test_cadence_k1_ticks_equal():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(15):
        orch.step(rng.standard_normal(4))
    assert orch._level_ticks[1] == orch._level_ticks[0]


def test_cadence_k1_l2_feature_dim():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, agents = make_stacked_orchestrator(l1_feature_dim=8, level_configs=configs)
    assert agents[1][0].config.feature_dim == 10


def test_cadence_k3_l2_fires_at_correct_steps():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    fire_steps = []
    for _ in range(12):
        result = orch.step(rng.standard_normal(4))
        if result["level2"] != {}:
            fire_steps.append(result["t"])
    assert fire_steps == [3, 6, 9, 12]


def test_cadence_k3_tick_counters():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(9):
        orch.step(rng.standard_normal(4))
    assert orch._level_ticks[0] == 9
    assert orch._level_ticks[1] == 3


def test_cadence_3level_k3_l3_fires_at_t9_18_27():
    configs = [
        LevelConfig(n_agents=2),
        LevelConfig(n_agents=2, K=3),
        LevelConfig(n_agents=1, K=3),
    ]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    l3_fire_steps = []
    for _ in range(30):
        result = orch.step(rng.standard_normal(4))
        if result["level3"] != {}:
            l3_fire_steps.append(result["t"])
    assert l3_fire_steps == [9, 18, 27]


def test_non_cadence_step_returns_empty():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=5)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    result = orch.step(np.zeros(4))  # t=1
    assert result["level2"] == {}


def test_t_counter_increments():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    for i in range(1, 6):
        result = orch.step(np.zeros(4))
        assert result["t"] == i


def test_k_larger_than_steps_no_error():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=100)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = orch.step(rng.standard_normal(4))
    assert result["level2"] == {}
    assert orch._level_ticks[1] == 0
