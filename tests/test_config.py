from hpm.config import AgentConfig


def test_config_defaults():
    cfg = AgentConfig(agent_id="a1", feature_dim=4)
    assert cfg.eta == 0.01
    assert cfg.beta_c == 0.1
    assert cfg.epsilon == 1e-4
    assert cfg.lambda_L == 0.1
    assert cfg.beta_aff == 0.5
    assert cfg.gamma_soc == 0.0   # off by default (single agent)
    assert cfg.feature_dim == 4
