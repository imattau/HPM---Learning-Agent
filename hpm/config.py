from dataclasses import dataclass, field


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
    delta_cost: float = 0.0    # weight of E_cost in J_i (0 = off, backward compatible)
    beta_comp: float = 0.0    # compression bonus weight in hierarchical total score (D7)
    lambda_cost: float = 1.0    # penalty scale inside ResourceCostEvaluator
    w_mem: float = 0.5          # memory weight in pressure scalar
    w_cpu: float = 0.5          # CPU weight in pressure scalar
    # Pattern density (density bias in MetaPatternRule, §A.8)
    kappa_D: float = 0.0     # density bias weight (0 = off, backward compatible)
    alpha_conn: float = 0.33  # weight of structural connectivity in D(h)
    alpha_sat: float = 0.33   # weight of evaluator saturation in D(h)
    alpha_amp: float = 0.34   # weight of field amplification in D(h)
    # Level classifier thresholds (Gap 2)
    l5_density: float = 0.85
    l5_conn: float = 0.80
    l5_comp: float = 0.70
    l4_conn: float = 0.70
    l4_comp: float = 0.60
    l3_conn: float = 0.50
    l3_comp: float = 0.40
    l2_conn: float = 0.30
    # Per-level kappa_D table (index 0 = Level 1, index 4 = Level 5)
    kappa_d_levels: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # Recombination Operator (Gap 3 / Appendix E)
    T_recomb: int = 100               # steps between time-triggered recombinations
    N_recomb: int = 3                 # max pair draw attempts per trigger
    kappa_max: float = 0.5            # max KL incompatibility for eligible pair
    conflict_threshold: float = 0.1   # total_conflict level that fires conflict trigger
    recomb_cooldown: int = 10         # min steps between any two recombinations
    obs_buffer_size: int = 50         # ring buffer capacity (recent observations)
    beta_orig: float = 1.0            # insight score scale
    alpha_nov: float = 0.5            # novelty weight in I(h*)
    alpha_eff: float = 0.5            # efficacy weight in I(h*)
    kappa_0: float = 0.1              # entry weight scale for accepted h*
    recomb_temp: float = 1.0          # softmax temperature for pair sampling
    conflict_stress_scale: float = 0.0  # multiplier: temp *= (1 + scale * total_conflict); 0 = off
