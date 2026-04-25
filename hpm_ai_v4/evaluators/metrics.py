import numpy as np

def epistemic_score(pattern):
    """Ai(t) = - Li(t)  (running loss, negative)"""
    return -pattern.running_loss

def affective_score(pattern, obs_seq, target_entropy=0.5):
    """Goldilocks curiosity + compression bonus (non-epistemic)"""
    if pattern.latent_dim > 1:
        # target loss ~0.5 means intermediate predictability (Goldilocks zone)
        recent_loss = pattern.running_loss
        curiosity = 1.0 - min(1.0, abs(recent_loss - 0.5)/0.5)
        
        # Additional compression bonus (mutual information between levels)
        comp = pattern.compression()
        return curiosity + 0.2 * comp
    else:
        # Flat pattern: curiosity based on entropy of its emission distribution
        # Use B[0, :] since flat patterns always assume z1=0
        p_row = pattern.B[0]
        ent = -np.sum(p_row * np.log(p_row + 1e-12))
        return 1.0 - abs(ent - target_entropy)/target_entropy

def social_score(pattern, field_frequencies):
    """Based on how common the pattern is in the population (field)."""
    return field_frequencies.get(pattern.id, 0.0)

def pattern_density(pattern, obs_seq, evaluator_values, field_amplification=0.0):
    """
    D(h) = alpha * C(h) + beta * sum(e_k(h)) + gamma * F(h)
    with C(h) = (1/n) * sum(w_ij) (structural connectivity).
    """
    if pattern.latent_dim > 1:
        # Structural connectivity: normalised number of parameters in the hierarchy
        # (1 transition matrix + 1 emission matrix + 1 initial distribution)
        n_params = (pattern.latent_dim**2) + (pattern.latent_dim * pattern.obs_dim) + pattern.latent_dim
        C = n_params / 100.0   # normalising constant for visualization/scaling (approx 0.2)
    else:
        C = 0.05   # flat patterns have minimal internal connectivity
        
    # Evaluator saturation: sum of non-epistemic evaluators (affective, social)
    E = sum(evaluator_values)
    
    # Field amplification (pre-computed social influence)
    F = field_amplification
    
    alpha, beta, gamma = 0.3, 0.5, 0.2
    return alpha * C + beta * E + gamma * F

def total_score(pattern, obs_seq, field_freq, beta_aff=0.4, gamma_soc=0.3,
                gamma_field=0.2, density_weight=0.1, external_soc=0.5):
    """Combines all evaluators and density into a final utility score."""
    ep = epistemic_score(pattern)
    aff = affective_score(pattern, obs_seq)
    soc_local = social_score(pattern, field_freq)
    
    # Blended social score: mix of local field frequency and external reliability
    soc = 0.5 * soc_local + 0.5 * external_soc

    # Non-epistemic evaluator sum J = beta*aff + gamma*soc
    J = beta_aff * aff + gamma_soc * soc
    
    # Field influence (external frequency)
    field_infl = gamma_field * field_freq.get(pattern.id, 0.0)
    
    # Base utility = epistemic + non-epistemic + field
    total = ep + J + field_infl
    
    # Add density after computing (density uses non-epistemic evaluators)
    density = pattern_density(pattern, obs_seq, [aff, soc], field_amplification=field_infl)
    total += density_weight * density
    
    return total
