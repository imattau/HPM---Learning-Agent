import numpy as np
import copy
from collections import deque
from hpm_ai_v4.pattern import HierarchicalPattern

def compute_conflict_matrix(patterns):
    """k_ij = 1 - cosine similarity between parameter vectors (Vectorized)."""
    K = len(patterns)
    if K == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # 1. Extract and flatten parameters for all patterns
    def get_params(pat):
        if pat.complexity >= 2 or pat.latent_dim > 1:
            return np.concatenate([
                pat.A.flatten(), pat.B.flatten(), pat.pi.flatten()
            ])
        else:
            return pat.B.flatten()

    param_list = [get_params(p) for p in patterns]
    max_len = max(len(p) for p in param_list)
    
    params = np.zeros((K, max_len), dtype=np.float32)
    for i, p in enumerate(param_list):
        params[i, :len(p)] = p
        
    # 2. Compute Cosine Similarity Matrix
    norms = np.linalg.norm(params, axis=1) # (K,)
    dot_products = params @ params.T # (K, K)
    
    norm_outer = np.outer(norms, norms) + 1e-12
    cos_sim = dot_products / norm_outer
    
    k_mat = 1.0 - np.maximum(0, cos_sim)
    np.fill_diagonal(k_mat, 0.0)
    return k_mat

def _update_density_weight(
    density_weight: float,
    density_state: dict | None,
    mean_running_loss: float | None,
) -> float:
    """Adapt density pressure from the recent loss trend.

    The density prior should become stickier only when it is helping the
    population reduce loss over time. If the recent trend moves the wrong way,
    the pressure is softened.
    """
    if density_state is None:
        return float(density_weight)

    recent = density_state.setdefault("recent_mean_loss", deque(maxlen=8))
    current = float(density_state.get("density_weight", density_weight))

    if mean_running_loss is not None and np.isfinite(float(mean_running_loss)):
        recent.append(float(mean_running_loss))

    if len(recent) >= 4:
        recent_list = list(recent)
        midpoint = max(1, len(recent_list) // 2)
        early = float(np.mean(recent_list[:midpoint]))
        late = float(np.mean(recent_list[midpoint:]))
        trend = late - early
        density_state["recent_loss_trend"] = float(trend)
        if trend > 0:
            current = max(0.05, current - 0.01)
        elif trend < 0:
            current = min(0.4, current + 0.005)

    density_state["density_weight"] = float(current)
    return float(current)


def meta_pattern_update(
    patterns,
    totals,
    eta=0.1,
    beta_c=0.05,
    k_matrix=None,
    decay=0.01,
    density_weight=0.1,
    density_state=None,
    mean_running_loss=None,
):
    """
    Update pattern weights using replicator dynamics with inhibition and decay (Vectorized).
    """
    if not patterns: return

    weights = np.array([p.weight for p in patterns], dtype=np.float32)
    total_vec = np.array([totals.get(p.id, 0.0) for p in patterns], dtype=np.float32)
    density_weight = _update_density_weight(density_weight, density_state, mean_running_loss)
    density_prior = np.array([
        float(getattr(p, "compression", lambda: 0.0)()) * max(0.1, float(getattr(p, "density_at_save", 0.1)))
        if getattr(p, "latent_dim", 1) > 1 else 0.05
        for p in patterns
    ], dtype=np.float32)
    if density_prior.size:
        density_prior = density_prior / (density_prior.sum() + 1e-12)
        effective_fitness = (1.0 - density_weight) * total_vec + density_weight * (density_prior * len(patterns))
    else:
        effective_fitness = total_vec
    
    avg_total = np.sum(weights * effective_fitness) / (np.sum(weights) + 1e-12)
    rep = eta * (effective_fitness - avg_total) * weights
    
    inhib = np.zeros_like(weights)
    if k_matrix is not None and k_matrix.shape == (len(patterns), len(patterns)):
        inhib = beta_c * weights * (k_matrix @ weights)
        
    new_weights = weights * (1 - decay) + rep - inhib
    new_weights = np.maximum(new_weights, 0.0)

    # Enforce minimum weight for HierarchicalPatterns (latent_dim > 1) so
    # FlatPatterns cannot crowd them out entirely before structure can emerge.
    for i, p in enumerate(patterns):
        if p.latent_dim > 1:
            new_weights[i] = max(new_weights[i], 0.02)

    total_w = np.sum(new_weights) + 1e-12
    new_weights = new_weights / total_w

    for i, p in enumerate(patterns):
        p.weight = float(new_weights[i])

def recombine(parent_a, parent_b, constraints=None):
    """
    Create a new pattern by structural crossover of parent matrices.
    """
    def crossover_mat(m1, m2):
        mask = np.random.rand(*m1.shape) < 0.5
        new = np.where(mask, m1, m2)
        row_sums = new.sum(axis=1, keepdims=True)
        return (new / (row_sums + 1e-12)).astype(np.float32)

    def crossover_vec(v1, v2):
        mask = np.random.rand(*v1.shape) < 0.5
        new = np.where(mask, v1, v2)
        return (new / (new.sum() + 1e-12)).astype(np.float32)

    if (parent_a.complexity >= 2 or parent_a.latent_dim > 1) and \
       (parent_b.complexity >= 2 or parent_b.latent_dim > 1):
        
        child = HierarchicalPattern(pattern_id=None, latent_dim=parent_a.latent_dim, obs_dim=parent_a.obs_dim)
        child.A = crossover_mat(parent_a.A, parent_b.A)
        child.B = crossover_mat(parent_a.B, parent_b.B)
        child.pi = crossover_vec(parent_a.pi, parent_b.pi)
    else:
        # Promotion logic
        child = HierarchicalPattern(pattern_id=None, latent_dim=parent_a.latent_dim, obs_dim=parent_a.obs_dim)
        if parent_a.latent_dim == 1:
            child.B[0] = parent_a.B[0]
            if child.latent_dim > 1:
                for k in range(1, child.latent_dim):
                    child.B[k] = parent_a.B[0]

    if constraints is not None and not constraints(child):
        return None
        
    child._refresh_log_cache()
    return child
