from dataclasses import dataclass


@dataclass
class AgentConfig:
    agent_id: str
    feature_dim: int
    # Dynamics (D5)
    eta: float = 0.01
    beta_c: float = 0.1
    epsilon: float = 1e-4
    # Evaluators
    lambda_L: float = 0.1        # EMA decay (D2)
    beta_aff: float = 0.5        # affective weight in J_i
    gamma_soc: float = 0.0       # social weight (0 = single agent)
    # Affective evaluator shape (§9.4)
    k: float = 1.0               # sigmoid sharpness
    c_opt: float = 10.0          # optimal complexity
    sigma_c: float = 5.0         # complexity bandwidth
    alpha_r: float = 0.0         # external reward weight
    # Social evaluator (Phase 2+)
    rho: float = 1.0             # field frequency amplification scale (D6)
    # Pattern initialisation
    init_sigma: float = 1.0      # initial covariance scale
