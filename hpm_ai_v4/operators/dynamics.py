import numpy as np
import copy
from hpm_ai_v4.pattern import HierarchicalPattern

def compute_conflict_matrix(patterns):
    """k_ij = 1 - cosine similarity between parameter vectors (structural incompatibility)."""
    K = len(patterns)
    mat = np.zeros((K, K))
    for i, p in enumerate(patterns):
        for j, q in enumerate(patterns):
            if i == j:
                mat[i, j] = 0.0
            else:
                # Extract parameter vector for similarity check
                def get_params(pat):
                    if pat.complexity >= 2:
                        return np.concatenate([
                            pat.A3.flatten(), pat.A32.flatten(), 
                            pat.A21.flatten(), pat.B.flatten()
                        ])
                    else:
                        return np.array([getattr(pat, 'theta', 0.5)])
                
                params_i = get_params(p)
                params_j = get_params(q)
                
                # Align lengths if different (e.g., flat vs hierarchical)
                if len(params_i) != len(params_j):
                    # For comparison, zero-pad the shorter one or use a proxy
                    max_len = max(len(params_i), len(params_j))
                    p_i = np.zeros(max_len)
                    p_j = np.zeros(max_len)
                    p_i[:len(params_i)] = params_i
                    p_j[:len(params_j)] = params_j
                else:
                    p_i, p_j = params_i, params_j
                    
                cos_sim = np.dot(p_i, p_j) / (np.linalg.norm(p_i)*np.linalg.norm(p_j) + 1e-12)
                mat[i, j] = 1 - max(0, cos_sim)
    return mat

def meta_pattern_update(patterns, totals, eta=0.1, beta_c=0.05, k_matrix=None, decay=0.01):
    """
    Update pattern weights using replicator dynamics with inhibition and decay.
    As defined in HPM Framework Appendix D.
    """
    weights = np.array([p.weight for p in patterns])
    total_vec = np.array([totals[p.id] for p in patterns])
    
    # Average population utility
    avg_total = np.sum(weights * total_vec) / (np.sum(weights) + 1e-12)

    # Apply forgetting / decay
    weights = weights * (1 - decay)

    new_weights = weights.copy()
    for i, p in enumerate(patterns):
        # Replication term (based on individual vs average utility)
        rep = eta * (total_vec[i] - avg_total) * weights[i]
        
        # Inhibition term (based on structural conflict)
        inhib = 0.0
        if k_matrix is not None:
            for j, q in enumerate(patterns):
                if i != j:
                    inhib += beta_c * k_matrix[i, j] * weights[i] * weights[j]
                    
        new_weights[i] = weights[i] + rep - inhib
        new_weights[i] = max(new_weights[i], 0.0)

    # Normalize weights to sum to 1
    total_w = np.sum(new_weights) + 1e-12
    new_weights = new_weights / total_w
    
    for i, p in enumerate(patterns):
        p.weight = new_weights[i]

def recombine(parent_a, parent_b, constraints=None):
    """
    Create a new pattern by structural crossover of parent matrices.
    """
    if parent_a.complexity >= 2 and parent_b.complexity >= 2:
        def crossover_mat(m1, m2):
            mask = np.random.rand(*m1.shape) < 0.5
            new = np.where(mask, m1, m2)
            # Row normalization (must be a valid transition matrix)
            row_sums = new.sum(axis=1, keepdims=True)
            return new / (row_sums + 1e-12)

        child = HierarchicalPattern(pattern_id=None)
        child.A3 = crossover_mat(parent_a.A3, parent_b.A3)
        child.A32 = crossover_mat(parent_a.A32, parent_b.A32)
        child.A21 = crossover_mat(parent_a.A21, parent_b.A21)
        child.B = crossover_mat(parent_a.B, parent_b.B)
        
        # Initial distributions
        child.pi3 = (parent_a.pi3 + parent_b.pi3) / 2
        child.pi3 /= child.pi3.sum()
    else:
        # If one is flat, promote to hierarchical with a mix of random and inherited
        child = HierarchicalPattern(pattern_id=None, obs_dim=parent_a.obs_dim)
        if parent_a.complexity == 1:
            # Inherit emission distribution from flat parent
            child.B[0] = parent_a.B[0]
            child.B[1] = parent_a.B[0] # Duplicate for both initial states

    if constraints is not None and not constraints(child):
        return None
        
    return child
