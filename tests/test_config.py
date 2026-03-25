from hpm.config import AgentConfig


def test_config_defaults():
    cfg = AgentConfig(agent_id="a1", feature_dim=4)
    assert cfg.eta == 0.01
    assert cfg.beta_c == 0.1
    assert cfg.epsilon == 1e-4
    assert cfg.lambda_L == 0.1
    assert cfg.beta_aff == 0.5
    assert cfg.gamma_soc == 0.0   # off by default (single agent)
    assert cfg.structural_message_eta == 0.05
    assert cfg.feature_dim == 4


def test_level_classifier_thresholds_have_defaults():
    cfg = AgentConfig(agent_id='a', feature_dim=4)
    assert cfg.l5_density == 0.85
    assert cfg.l5_conn == 0.80
    assert cfg.l5_comp == 0.70
    assert cfg.l4_conn == 0.70
    assert cfg.l4_comp == 0.60
    assert cfg.l3_conn == 0.50
    assert cfg.l3_comp == 0.40
    assert cfg.l2_conn == 0.30


def test_kappa_d_levels_default_is_five_zeros():
    cfg = AgentConfig(agent_id='a', feature_dim=4)
    assert cfg.kappa_d_levels == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_kappa_d_levels_instances_are_independent():
    cfg1 = AgentConfig(agent_id='a', feature_dim=4)
    cfg2 = AgentConfig(agent_id='b', feature_dim=4)
    cfg1.kappa_d_levels[0] = 99.0
    assert cfg2.kappa_d_levels[0] == 0.0   # no shared mutable default


def test_recombination_config_defaults():
    cfg = AgentConfig(agent_id='a', feature_dim=2)
    assert cfg.T_recomb == 100
    assert cfg.N_recomb == 3
    assert cfg.kappa_max == 0.5
    assert cfg.conflict_threshold == 0.1
    assert cfg.recomb_cooldown == 10
    assert cfg.obs_buffer_size == 50
    assert cfg.beta_orig == 1.0
    assert cfg.alpha_nov == 0.5
    assert cfg.alpha_eff == 0.5
    assert cfg.kappa_0 == 0.1
    assert cfg.recomb_temp == 1.0
