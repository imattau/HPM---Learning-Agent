import numpy as np
import copy
from hpm_ai_v4.pattern import HierarchicalPattern

def compute_conflict_matrix(patterns):
    """k_ij = 1 - cosine similarity between parameter vectors (Vectorized)."""
    K = len(patterns)
    if K == 0:
        return np.array([[]])
    
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

def meta_pattern_update(patterns, totals, eta=0.1, beta_c=0.05, k_matrix=None, decay=0.01):
    """
    Update pattern weights using replicator dynamics with inhibition and decay (Vectorized).
    """
    if not patterns: return
    
    weights = np.array([p.weight for p in patterns], dtype=np.float32)
    total_vec = np.array([totals.get(p.id, 0.0) for p in patterns], dtype=np.float32)
    
    avg_total = np.sum(weights * total_vec) / (np.sum(weights) + 1e-12)
    rep = eta * (total_vec - avg_total) * weights
    
    inhib = np.zeros_like(weights)
    if k_matrix is not None and k_matrix.shape == (len(patterns), len(patterns)):
        inhib = beta_c * weights * (k_matrix @ weights)
        
    new_weights = weights * (1 - decay) + rep - inhib
    new_weights = np.maximum(new_weights, 0.0)

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
