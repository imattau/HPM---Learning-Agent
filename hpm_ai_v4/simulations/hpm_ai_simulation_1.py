```python
import numpy as np
from scipy.special import logsumexp
from collections import defaultdict
import copy

np.random.seed(42)   # for reproducibility

# ============================================================================
# Environment: True 2‑state HMM (the "world" the agent learns)
# ============================================================================
class TrueEnvironment:
    def __init__(self):
        # True transition and emission matrices
        self.A_true = np.array([[0.8, 0.2],
                                [0.2, 0.8]])
        self.B_true = np.array([[0.9, 0.1],
                                [0.1, 0.9]])
        self.state = 0

    def reset(self):
        self.state = np.random.choice([0, 1], p=[0.5, 0.5])

    def step(self):
        obs = np.random.choice([0, 1], p=self.B_true[self.state])
        self.state = np.random.choice([0, 1], p=self.A_true[self.state])
        return obs

# ============================================================================
# Hierarchical Pattern (2‑level HMM) with online EM (sliding window batch EM)
# ============================================================================
class HierarchicalPattern:
    def __init__(self, pattern_id, latent_dim=2, obs_dim=2):
        self.id = pattern_id
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        # Random initialisation (structured priors would be better, but random is fine)
        self.A = np.random.dirichlet(np.ones(latent_dim), size=latent_dim)
        self.B = np.random.dirichlet(np.ones(obs_dim), size=latent_dim)
        self.pi = np.random.dirichlet(np.ones(latent_dim))

        # Learning state
        self.running_loss = 0.0
        self.obs_buffer = []          # for EM window
        self.weight = 1.0
        self.creation_step = 0

    # ------------------------------------------------------------------------
    # HMM inference (forward / forward‑backward)
    # ------------------------------------------------------------------------
    def log_likelihood(self, obs_seq):
        """Compute log p(obs_seq) using forward algorithm."""
        if len(obs_seq) == 0:
            return 0.0
        T = len(obs_seq)
        alpha = np.zeros((T, self.latent_dim))
        # initialisation
        for s in range(self.latent_dim):
            alpha[0, s] = np.log(self.pi[s] + 1e-12) + np.log(self.B[s, obs_seq[0]] + 1e-12)
        # recursion
        for t in range(1, T):
            for s in range(self.latent_dim):
                log_sum = logsumexp(alpha[t-1] + np.log(self.A[:, s] + 1e-12))
                alpha[t, s] = log_sum + np.log(self.B[s, obs_seq[t]] + 1e-12)
        return logsumexp(alpha[-1])

    def _forward_backward(self, obs_seq):
        """Return posterior probabilities gamma[t, s] and xi[t, i, j]."""
        T = len(obs_seq)
        K = self.latent_dim
        # forward
        alpha = np.zeros((T, K))
        for s in range(K):
            alpha[0, s] = np.log(self.pi[s]) + np.log(self.B[s, obs_seq[0]])
        for t in range(1, T):
            for s in range(K):
                alpha[t, s] = logsumexp(alpha[t-1] + np.log(self.A[:, s])) + np.log(self.B[s, obs_seq[t]])
        log_lik = logsumexp(alpha[-1])
        # backward
        beta = np.zeros((T, K))
        beta[-1, :] = 0.0
        for t in range(T-2, -1, -1):
            for s in range(K):
                beta[t, s] = logsumexp(np.log(self.A[s, :]) + beta[t+1, :] + np.log(self.B[:, obs_seq[t+1]]))
        # gamma
        gamma = np.zeros((T, K))
        for t in range(T):
            log_gamma = alpha[t] + beta[t] - log_lik
            gamma[t] = np.exp(log_gamma - logsumexp(log_gamma))
        # xi (transition posteriors)
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.exp(alpha[t, i] + np.log(self.A[i, j]) +
                                          np.log(self.B[j, obs_seq[t+1]]) + beta[t+1, j] - log_lik)
        return gamma, xi

    # ------------------------------------------------------------------------
    # Online EM (sliding window batch EM, updated every step)
    # ------------------------------------------------------------------------
    def update_parameters_online(self, all_obs, window_size=30):
        """Use last `window_size` observations to re‑estimate parameters via batch EM."""
        if len(all_obs) < window_size:
            return
        window = all_obs[-window_size:]
        gamma, xi = self._forward_backward(window)

        # Accumulate sufficient statistics
        K = self.latent_dim
        new_A = np.zeros((K, K))
        new_B = np.zeros((K, self.obs_dim))
        new_pi = gamma[0]

        # Transition counts
        for t in range(len(window)-1):
            for i in range(K):
                for j in range(K):
                    new_A[i, j] += xi[t, i, j]

        # Emission counts
        for t, obs in enumerate(window):
            for s in range(K):
                new_B[s, obs] += gamma[t, s]

        # Normalise
        for i in range(K):
            row_sum = new_A[i].sum()
            if row_sum > 0:
                new_A[i] /= row_sum
        for s in range(K):
            row_sum = new_B[s].sum()
            if row_sum > 0:
                new_B[s] /= row_sum
        if new_pi.sum() > 0:
            new_pi /= new_pi.sum()

        # Apply update (smooth with old parameters to avoid overfitting)
        alpha_smooth = 0.7
        self.A = alpha_smooth * new_A + (1-alpha_smooth) * self.A
        self.B = alpha_smooth * new_B + (1-alpha_smooth) * self.B
        self.pi = alpha_smooth * new_pi + (1-alpha_smooth) * self.pi

        # Keep rows stochastic
        self.A = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-12)
        self.B = self.B / (self.B.sum(axis=1, keepdims=True) + 1e-12)
        self.pi = self.pi / (self.pi.sum() + 1e-12)

    # ------------------------------------------------------------------------
    # Updating running loss (exponential moving average of negative log‑likelihood)
    # ------------------------------------------------------------------------
    def update_running_loss(self, obs_seq, lambda_l=0.1):
        if len(obs_seq) == 0:
            return
        ll = self.log_likelihood(obs_seq[-30:])   # use recent window for speed
        avg_loss = -ll / max(1, len(obs_seq[-30:]))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * avg_loss

    # ------------------------------------------------------------------------
    # Compression: mutual information I(z_t; z_{t-1}) approximated by stationary entropy
    # (simplified for 2‑level HMM)
    # ------------------------------------------------------------------------
    def compression(self):
        # For a simple HMM, compression = H(state) - H(state|previous) = Markov chain mutual info
        # Compute stationary distribution of A
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        stationary = np.real(eigvecs[:, np.isclose(eigvals, 1.0)])
        if stationary.size == 0:
            stat = np.ones(self.latent_dim) / self.latent_dim
        else:
            stat = stationary[:, 0]
            stat = np.abs(stat) / np.sum(np.abs(stat))
        H_state = -np.sum(stat * np.log(stat + 1e-12))
        # Conditional entropy H(state_t | state_{t-1}) = average over rows of H(row)
        H_cond = 0.0
        for i in range(self.latent_dim):
            row = self.A[i, :]
            H_cond += stat[i] * (-np.sum(row * np.log(row + 1e-12)))
        mi = max(0, H_state - H_cond)
        return mi

    # ------------------------------------------------------------------------
    # Predictive entropy (used in affective evaluator)
    # ------------------------------------------------------------------------
    def predictive_entropy(self, obs_seq):
        # One‑step ahead entropy, approximated using stationary distribution
        if len(obs_seq) == 0:
            # Use stationary distribution of state → predictive emission entropy
            eigvals, eigvecs = np.linalg.eig(self.A.T)
            stationary = np.real(eigvecs[:, np.isclose(eigvals, 1.0)])
            if stationary.size == 0:
                stat = np.ones(self.latent_dim) / self.latent_dim
            else:
                stat = stationary[:, 0]
                stat = np.abs(stat) / np.sum(np.abs(stat))
        else:
            # Use last observation to update belief (simple forward step)
            # For simplicity, approximate using running belief from last forward pass
            # We'll just use stationary for stability
            stat = None   # fallback to stationary
        # Emission distribution mixture
        p_obs = stat @ self.B
        ent = -np.sum(p_obs * np.log(p_obs + 1e-12))
        return ent

# ============================================================================
# Flat pattern (Bernoulli) for baseline
# ============================================================================
class FlatPattern:
    def __init__(self, pattern_id, theta=0.5):
        self.id = pattern_id
        self.theta = theta
        self.running_loss = 0.0
        self.weight = 1.0
        self.obs_buffer = []
        self.complexity = 1

    def log_likelihood(self, obs_seq):
        if len(obs_seq) == 0:
            return 0.0
        p = self.theta if obs_seq[-1] == 1 else (1 - self.theta)
        return np.log(p + 1e-12)

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        if len(obs_seq) == 0:
            return
        ll = self.log_likelihood([obs_seq[-1]])   # only last observation
        loss = -ll
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * loss

    def compression(self):
        return 0.0

    def predictive_entropy(self, obs_seq):
        p1 = self.theta
        return -p1 * np.log(p1+1e-12) - (1-p1) * np.log(1-p1+1e-12)

# ============================================================================
# Evaluators and total score
# ============================================================================
def epistemic_score(pattern):
    return -pattern.running_loss

def affective_score(pattern, obs_seq, target_entropy=0.5):
    ent = pattern.predictive_entropy(obs_seq)
    curiosity = 1.0 - min(1.0, abs(ent - target_entropy) / target_entropy)
    comp = pattern.compression()
    bonus = 0.2 * comp
    return curiosity + bonus

def social_score(pattern, field_freq):
    return field_freq.get(pattern.id, 0.0)

def pattern_density(pattern, aff_score, soc_score, field_amp):
    # Structural connectivity: approximate by number of parameters (log scale)
    if hasattr(pattern, 'complexity') and pattern.complexity == 1:
        C = 0.2
    else:
        C = np.log(pattern.latent_dim * pattern.latent_dim * 2)   # A + B + pi
    E = aff_score + soc_score
    F = field_amp
    alpha, beta, gamma = 0.3, 0.4, 0.3
    return alpha * C + beta * E + gamma * F

def total_score(pattern, obs_seq, field_freq, beta_aff=0.4, gamma_soc=0.3,
                gamma_field=0.2, density_weight=0.1):
    ep = epistemic_score(pattern)
    aff = affective_score(pattern, obs_seq)
    soc = social_score(pattern, field_freq)
    field_infl = gamma_field * field_freq.get(pattern.id, 0.0)
    J = beta_aff * aff + gamma_soc * soc
    total = ep + J + field_infl
    density = pattern_density(pattern, aff, soc, field_infl)
    total += density_weight * density
    return total

# ============================================================================
# Replicator dynamics with conflict matrix and decay
# ============================================================================
def compute_conflict_matrix(patterns):
    K = len(patterns)
    mat = np.zeros((K, K))
    for i, p in enumerate(patterns):
        for j, q in enumerate(patterns):
            if i == j:
                mat[i, j] = 0.0
            else:
                # Conflict = 1 - cosine similarity of parameter vectors (if both hierarchical)
                if hasattr(p, 'A') and hasattr(q, 'A'):
                    vec_p = np.concatenate([p.A.flatten(), p.B.flatten(), p.pi])
                    vec_q = np.concatenate([q.A.flatten(), q.B.flatten(), q.pi])
                else:
                    # flat patterns: use theta
                    vec_p = np.array([p.theta]) if hasattr(p, 'theta') else np.array([0])
                    vec_q = np.array([q.theta]) if hasattr(q, 'theta') else np.array([0])
                norm_p = np.linalg.norm(vec_p) + 1e-12
                norm_q = np.linalg.norm(vec_q) + 1e-12
                cos_sim = np.dot(vec_p, vec_q) / (norm_p * norm_q)
                mat[i, j] = 1 - max(0, cos_sim)
    return mat

def meta_pattern_update(patterns, totals, eta=0.1, beta_c=0.03, k_matrix=None, decay=0.005):
    weights = np.array([p.weight for p in patterns])
    total_vec = np.array([totals[p.id] for p in patterns])
    avg_total = np.sum(weights * total_vec) / (np.sum(weights) + 1e-12)

    # Forgetting / decay
    weights = weights * (1 - decay)

    new_weights = weights.copy()
    for i in range(len(patterns)):
        rep = eta * (total_vec[i] - avg_total) * weights[i]
        inhib = 0.0
        if k_matrix is not None:
            for j in range(len(patterns)):
                if i != j:
                    inhib += beta_c * k_matrix[i, j] * weights[i] * weights[j]
        new_weights[i] = weights[i] + rep - inhib
        new_weights[i] = max(new_weights[i], 0.0)

    total_w = np.sum(new_weights) + 1e-12
    new_weights /= total_w
    for i, p in enumerate(patterns):
        p.weight = new_weights[i]

# ============================================================================
# Recombination (crossover) for hierarchical patterns
# ============================================================================
def recombine(parent_a, parent_b):
    """Row‑wise crossover of A, B, pi matrices. Enforces row stochasticity."""
    if hasattr(parent_a, 'A') and hasattr(parent_b, 'A'):
        # hierarchical patterns
        K = parent_a.latent_dim
        A_new = np.zeros_like(parent_a.A)
        B_new = np.zeros_like(parent_a.B)
        pi_new = np.zeros_like(parent_a.pi)
        for i in range(K):
            choose = np.random.rand() < 0.5
            A_new[i, :] = parent_a.A[i, :] if choose else parent_b.A[i, :]
            B_new[i, :] = parent_a.B[i, :] if choose else parent_b.B[i, :]
        # pi: choose whole vector from one parent
        pi_new = parent_a.pi if np.random.rand() < 0.5 else parent_b.pi
        # Normalise rows (already stochastic from parents but ensure)
        A_new = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-12)
        B_new = B_new / (B_new.sum(axis=1, keepdims=True) + 1e-12)
        pi_new = pi_new / (pi_new.sum() + 1e-12)
        child = HierarchicalPattern(pattern_id=None, latent_dim=K)
        child.A, child.B, child.pi = A_new, B_new, pi_new
        child.weight = 0.05
        return child
    else:
        # flat patterns: average theta
        theta_new = (parent_a.theta + parent_b.theta) / 2
        child = FlatPattern(pattern_id=None, theta=theta_new)
        child.weight = 0.05
        return child

# ============================================================================
# Agent (contains the pattern population)
# ============================================================================
class HPMAgent:
    def __init__(self, num_hier=3, num_flat=3):
        self.patterns = []
        # hierarchical patterns
        for i in range(num_hier):
            p = HierarchicalPattern(i)
            p.weight = 1.0 / (num_hier + num_flat)
            self.patterns.append(p)
        # flat patterns
        for i in range(num_hier, num_hier+num_flat):
            theta = np.random.uniform(0.3, 0.7)
            p = FlatPattern(i, theta=theta)
            p.weight = 1.0 / (num_hier + num_flat)
            self.patterns.append(p)
        self.obs_buffer = []
        self.step_counter = 0

    def perceive_and_learn(self, obs, field_freq):
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) > 200:
            self.obs_buffer = self.obs_buffer[-200:]

        # 1) Online EM for each hierarchical pattern
        for p in self.patterns:
            if hasattr(p, 'update_parameters_online'):
                p.update_parameters_online(self.obs_buffer, window_size=30)

        # 2) Update running loss for all patterns (using last 30 obs for efficiency)
        for p in self.patterns:
            p.update_running_loss(self.obs_buffer[-30:])

        # 3) Compute total scores
        totals = {}
        for p in self.patterns:
            totals[p.id] = total_score(p, self.obs_buffer, field_freq)

        # 4) Compute conflict matrix
        k_mat = compute_conflict_matrix(self.patterns)

        # 5) Replicator update
        meta_pattern_update(self.patterns, totals, eta=0.1, beta_c=0.03,
                            k_matrix=k_mat, decay=0.005)

        # 6) Recombination every 25 steps
        if self.step_counter > 0 and self.step_counter % 25 == 0:
            weights = np.array([p.weight for p in self.patterns])
            if np.sum(weights) > 0:
                probs = weights / np.sum(weights)
                idx = np.random.choice(len(self.patterns), size=2, p=probs, replace=False)
                parent_a = self.patterns[idx[0]]
                parent_b = self.patterns[idx[1]]
                child = recombine(parent_a, parent_b)
                if child is not None:
                    child.id = max([p.id for p in self.patterns] + [0]) + 1
                    child.creation_step = self.step_counter
                    self.patterns.append(child)

        # 7) Prune low‑weight patterns
        self.patterns = [p for p in self.patterns if p.weight > 0.001]

        self.step_counter += 1

# ============================================================================
# Main simulation
# ============================================================================
def run_simulation(steps=500):
    env = TrueEnvironment()
    env.reset()
    agent = HPMAgent(num_hier=3, num_flat=3)

    # For logging
    history = []

    for step in range(steps):
        obs = env.step()
        # Field frequencies: simply the current weight distribution
        weights = np.array([p.weight for p in agent.patterns])
        field_freq = {p.id: w / (np.sum(weights)+1e-12) for p, w in zip(agent.patterns, weights)}
        agent.perceive_and_learn(obs, field_freq)

        if step % 50 == 0 or step == steps-1:
            # Get best pattern
            best = max(agent.patterns, key=lambda p: p.weight)
            avg_ep = np.mean([epistemic_score(p) for p in agent.patterns])
            print(f"Step {step:3d}: best pattern id={best.id}, "
                  f"type={'hier' if hasattr(best,'A') else 'flat'}, "
                  f"weight={best.weight:.3f}, ep_score={epistemic_score(best):.3f}, "
                  f"comp={best.compression():.3f if hasattr(best,'A') else 0:.3f}, "
                  f"pop_size={len(agent.patterns)}")
            history.append((step, best.id, best.weight, epistemic_score(best)))

    # Final analysis
    print("\n--- Final population ---")
    for p in agent.patterns:
        print(f"ID {p.id}: weight={p.weight:.3f}, "
              f"type={'hier' if hasattr(p,'A') else 'flat'}, "
              f"ep_score={epistemic_score(p):.3f}")
    return history

if __name__ == "__main__":
    history = run_simulation(500)
```
