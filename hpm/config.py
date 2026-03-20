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
    alpha_int: float = 0.8       # internal/external field blend (1.0 = agents only, §3.8)
    # Pattern initialisation
    init_sigma: float = 1.0      # initial covariance scale
    # Resource cost evaluator
    delta_cost: float = 0.0     # weight of E_cost in J_i (0 = off, backward compatible)
    lambda_cost: float = 1.0    # penalty scale inside ResourceCostEvaluator
    w_mem: float = 0.5          # memory weight in pressure scalar
    w_cpu: float = 0.5          # CPU weight in pressure scalar
    # Pattern density (density bias in MetaPatternRule, §A.8)
    kappa_D: float = 0.0     # density bias weight (0 = off, backward compatible)
    alpha_conn: float = 0.33  # weight of structural connectivity in D(h)
    alpha_sat: float = 0.33   # weight of evaluator saturation in D(h)
    alpha_amp: float = 0.34   # weight of field amplification in D(h)
