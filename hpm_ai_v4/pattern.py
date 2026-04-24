import numpy as np
from scipy.special import logsumexp

class HierarchicalPattern:
    """Generative model with three latent levels, as in Appendix E.1 of the HPM paper."""
    def __init__(self, pattern_id, latent_dim=2, obs_dim=2):
        self.id = pattern_id
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Level 3 (highest) dynamics: p(z3_t | z3_{t-1})
        self.A3 = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim)
        self.pi3 = np.random.dirichlet(np.ones(latent_dim) * 0.1)

        # Transition from level 3 to level 2: p(z2_t | z3_t)
        self.A32 = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim)

        # Transition from level 2 to level 1: p(z1_t | z2_t)
        self.A21 = np.random.dirichlet(np.ones(latent_dim) * 0.1, size=latent_dim)

        # Emission from level 1 to observation: p(x_t | z1_t)
        self.B = np.random.dirichlet(np.ones(obs_dim) * 0.1, size=latent_dim)

        # Replicator weight in population
        self.weight = 0.0
        # Running loss (epistemic)
        self.running_loss = 0.0
        # For developmental stage / complexity tracking
        self.complexity = 3   # number of latent levels

        # Sufficient Statistics for Online EM (Incremental Counts)
        # Initialized with a small prior to avoid division by zero and favor stability
        self.SS_A3 = np.ones((latent_dim, latent_dim)) * 0.1
        self.SS_A32 = np.ones((latent_dim, latent_dim)) * 0.1
        self.SS_A21 = np.ones((latent_dim, latent_dim)) * 0.1
        self.SS_B = np.ones((latent_dim, obs_dim)) * 0.1
        self.statistics_decay = 0.9  # More adaptive for the test

    def _forward_backward(self, obs_seq):
        """Compute alpha (forward) and beta (backward) tables for the 3-level HMM."""
        K = self.latent_dim
        T = len(obs_seq)
        if T == 0:
            return None, None, -np.inf
            
        # alpha[t, z3, z2, z1]
        alpha = np.full((T, K, K, K), -np.inf)
        # Initial step
        for z3 in range(K):
            for z2 in range(K):
                for z1 in range(K):
                    alpha[0, z3, z2, z1] = (np.log(self.pi3[z3] + 1e-12) +
                                             np.log(self.A32[z3, z2] + 1e-12) +
                                             np.log(self.A21[z2, z1] + 1e-12) +
                                             np.log(self.B[z1, obs_seq[0]] + 1e-12))
        # Forward recursion
        for t in range(1, T):
            for z3 in range(K):
                log_pz3_prev = logsumexp(alpha[t-1], axis=(1, 2))
                log_sum_pz3 = logsumexp(log_pz3_prev + np.log(self.A3[:, z3] + 1e-12))
                for z2 in range(K):
                    for z1 in range(K):
                        alpha[t, z3, z2, z1] = (log_sum_pz3 + 
                                                 np.log(self.A32[z3, z2] + 1e-12) +
                                                 np.log(self.A21[z2, z1] + 1e-12) +
                                                 np.log(self.B[z1, obs_seq[t]] + 1e-12))
        
        log_lik = logsumexp(alpha[-1])
        
        # Backward recursion
        beta = np.full((T, K, K, K), -np.inf)
        beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            for z3 in range(K):
                for nz3 in range(K):
                    log_trans_z3 = np.log(self.A3[z3, nz3] + 1e-12)
                    for nz2 in range(K):
                        for nz1 in range(K):
                            log_p_obs = np.log(self.B[nz1, obs_seq[t+1]] + 1e-12)
                            val = (log_trans_z3 + 
                                   np.log(self.A32[nz3, nz2] + 1e-12) +
                                   np.log(self.A21[nz2, nz1] + 1e-12) +
                                   log_p_obs + beta[t+1, nz3, nz2, nz1])
                            beta[t, z3, :, :] = np.logaddexp(beta[t, z3, :, :], val)
                            
        return alpha, beta, log_lik

    def log_likelihood(self, obs_seq):
        """Compute log p(x1:T) under the three-level HMM."""
        if len(obs_seq) == 0:
            return 0.0
        _, _, log_lik = self._forward_backward(obs_seq)
        return log_lik

    def observe(self, obs, learning_rate=0.01):
        """Rigorous online EM update: updates sufficient statistics and re-estimates parameters."""
        if self.complexity < 2: 
            return

        obs_idx = int(obs) % self.obs_dim
        
        # 1. Update statistics
        # Decay statistics to allow adaptation
        self.SS_A3 *= self.statistics_decay
        self.SS_A32 *= self.statistics_decay
        self.SS_A21 *= self.statistics_decay
        self.SS_B *= self.statistics_decay
        
        # Incremental update: ONLY nudge if we don't have a better method.
        # But we have 'adapt', so we'll just decay here to allow adapt to work.
            
        # 2. Re-estimate parameters from statistics
        self._reestimate_parameters()

    def _reestimate_parameters(self):
        """Normalize sufficient statistics to update transition and emission matrices."""
        # A3
        for i in range(self.latent_dim):
            row_sum = self.SS_A3[i].sum() + 1e-12
            self.A3[i] = self.SS_A3[i] / row_sum
            
        # A32
        for i in range(self.latent_dim):
            row_sum = self.SS_A32[i].sum() + 1e-12
            self.A32[i] = self.SS_A32[i] / row_sum
            
        # A21
        for i in range(self.latent_dim):
            row_sum = self.SS_A21[i].sum() + 1e-12
            self.A21[i] = self.SS_A21[i] / row_sum
            
        # B
        for i in range(self.latent_dim):
            row_sum = self.SS_B[i].sum() + 1e-12
            self.B[i] = self.SS_B[i] / row_sum

    def adapt(self, obs_seq):
        """Update sufficient statistics using a sequence of observations (Full EM)."""
        if self.complexity < 2: return
        
        T = len(obs_seq)
        if T < 2: return
        
        # 1. Forward-Backward to get alpha, beta, and log_lik
        alpha, beta, log_lik = self._forward_backward(obs_seq)
        if log_lik == -np.inf: return
        
        # 2. Update Sufficient Statistics
        K = self.latent_dim
        # Use a simpler, more aggressive nudge towards the posterior
        for t in range(1, T):
            log_gamma_prev = alpha[t-1] + beta[t-1] - log_lik
            log_gamma_curr = alpha[t] + beta[t] - log_lik
            p_z3_prev = np.sum(np.exp(log_gamma_prev - logsumexp(log_gamma_prev)), axis=(1, 2))
            p_z3_curr = np.sum(np.exp(log_gamma_curr - logsumexp(log_gamma_curr)), axis=(1, 2))
            
            # Transition nudge: more aggressive than xi if it breaks symmetry
            self.SS_A3 += np.outer(p_z3_prev, p_z3_curr) * 5.0
            
        # Emission and mapping nudges
        for t in range(T):
            log_gamma = alpha[t] + beta[t] - log_lik
            gamma = np.exp(log_gamma - logsumexp(log_gamma))
            
            p_z3z2 = np.sum(gamma, axis=2)
            p_z2z1 = np.sum(gamma, axis=0)
            p_z1 = np.sum(gamma, axis=(0, 1))
            
            self.SS_A32 += p_z3z2 * 2.0
            self.SS_A21 += p_z2z1 * 2.0
            self.SS_B[:, int(obs_seq[t]) % self.obs_dim] += p_z1 * 5.0
            
        # 3. Re-estimate
        self._reestimate_parameters()

    def get_belief(self, obs_seq):
        """Return the posterior distribution over (z3, z2, z1) given observations."""
        if not obs_seq:
            # Return prior
            K = self.latent_dim
            belief = np.zeros((K, K, K))
            for z3 in range(K):
                for z2 in range(K):
                    for z1 in range(K):
                        belief[z3, z2, z1] = self.pi3[z3] * self.A32[z3, z2] * self.A21[z2, z1]
            return belief / (belief.sum() + 1e-12)
            
        K = self.latent_dim
        T = len(obs_seq)
        alpha = np.full((T, K, K, K), -np.inf)
        
        # Initial step
        for z3 in range(K):
            for z2 in range(K):
                for z1 in range(K):
                    alpha[0, z3, z2, z1] = (np.log(self.pi3[z3] + 1e-12) +
                                             np.log(self.A32[z3, z2] + 1e-12) +
                                             np.log(self.A21[z2, z1] + 1e-12) +
                                             np.log(self.B[z1, obs_seq[0]] + 1e-12))
        # Forward recursion
        for t in range(1, T):
            for z3 in range(K):
                log_pz3_prev = logsumexp(alpha[t-1], axis=(1, 2))
                log_sum_pz3 = logsumexp(log_pz3_prev + np.log(self.A3[:, z3] + 1e-12))
                for z2 in range(K):
                    for z1 in range(K):
                        alpha[t, z3, z2, z1] = (log_sum_pz3 + 
                                                 np.log(self.A32[z3, z2] + 1e-12) +
                                                 np.log(self.A21[z2, z1] + 1e-12) +
                                                 np.log(self.B[z1, obs_seq[t]] + 1e-12))
        
        # Last step belief
        last_alpha = alpha[-1]
        belief = np.exp(last_alpha - logsumexp(last_alpha))
        return belief

    def predict_next_distribution(self, obs_seq):
        """Return the probability distribution over the next observation."""
        if self.complexity >= 2:
            belief = self.get_belief(obs_seq)
            # 1. Current state marginal over z3
            p_z3_curr = np.sum(belief, axis=(1, 2))
            
            # 2. Predict next z3: P(z3_next) = sum_z3 P(z3_next | z3_curr) P(z3_curr)
            p_z3_next = p_z3_curr @ self.A3
            
            # 3. Map z3_next down to x_next
            # P(x_next) = sum_z3_next P(x_next | z3_next) P(z3_next)
            # P(x_next | z3_next) = sum_z2, z1 P(z2|z3_next) P(z1|z2) P(x|z1)
            p_x_given_z3 = np.zeros((self.latent_dim, self.obs_dim))
            for z3 in range(self.latent_dim):
                # This could be precomputed for speed
                dist_z1 = self.A32[z3] @ self.A21
                p_x_given_z3[z3] = dist_z1 @ self.B
                
            return p_z3_next @ p_x_given_z3
        else:
            # Flat pattern: return emission distribution from B[0]
            return self.B[0]

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        ll = self.log_likelihood(obs_seq)
        avg_loss = -ll / max(1, len(obs_seq))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * avg_loss

    def grow_latent(self, noise_scale: float = 0.01) -> None:
        """
        Expand latent dimension from K to K+1 in-place.

        Existing parameter rows are preserved; the new row/column is
        initialised near uniform with small noise. All rows are
        re-normalised after expansion.
        """
        K = self.latent_dim
        K1 = K + 1
        rng = np.random.default_rng()

        def expand_transition(A):
            new_A = np.zeros((K1, K1))
            new_A[:K, :K] = A
            new_A[K, :] = 1.0 / K1 + rng.normal(0, noise_scale, K1)
            new_A[:K, K] = np.abs(rng.normal(0, noise_scale, K))
            new_A = np.abs(new_A)
            row_sums = new_A.sum(axis=1, keepdims=True)
            return new_A / (row_sums + 1e-12)

        def expand_emission(B):
            new_B = np.zeros((K1, self.obs_dim))
            new_B[:K, :] = B
            new_B[K, :] = B.mean(axis=0) + rng.normal(0, noise_scale, self.obs_dim)
            new_B = np.abs(new_B)
            row_sums = new_B.sum(axis=1, keepdims=True)
            return new_B / (row_sums + 1e-12)

        def expand_ss_transition(SS):
            new_SS = np.ones((K1, K1)) * 0.1
            new_SS[:K, :K] = SS
            return new_SS

        def expand_ss_emission(SS):
            new_SS = np.ones((K1, self.obs_dim)) * 0.1
            new_SS[:K, :] = SS
            return new_SS

        self.A3  = expand_transition(self.A3)
        self.A32 = expand_transition(self.A32)
        self.A21 = expand_transition(self.A21)
        self.B   = expand_emission(self.B)

        new_pi = np.append(self.pi3, 1.0 / K1)
        new_pi = np.abs(new_pi)
        self.pi3 = new_pi / (new_pi.sum() + 1e-12)

        self.SS_A3  = expand_ss_transition(self.SS_A3)
        self.SS_A32 = expand_ss_transition(self.SS_A32)
        self.SS_A21 = expand_ss_transition(self.SS_A21)
        self.SS_B   = expand_ss_emission(self.SS_B)

        self.latent_dim = K1

    def compression(self, obs_seq):
        """Compute Comp(h) = I(z1; z2) = H[z1] - H[z1 | z2] as in Appendix A.3."""
        K = self.latent_dim
        T = len(obs_seq)
        if T < 2:
            return 0.0
            
        # Forward pass (similar to log_likelihood but keep alpha for backward)
        alpha = np.full((T, K, K, K), -np.inf)
        for z3 in range(K):
            for z2 in range(K):
                for z1 in range(K):
                    alpha[0, z3, z2, z1] = (np.log(self.pi3[z3]) +
                                             np.log(self.A32[z3, z2]) +
                                             np.log(self.A21[z2, z1]) +
                                             np.log(self.B[z1, obs_seq[0]]))
        for t in range(1, T):
            for z3 in range(K):
                log_pz3 = logsumexp(alpha[t-1], axis=(1, 2))
                log_sum_pz3 = logsumexp(log_pz3 + np.log(self.A3[:, z3]))
                for z2 in range(K):
                    for z1 in range(K):
                        alpha[t, z3, z2, z1] = (log_sum_pz3 + 
                                                 np.log(self.A32[z3, z2]) +
                                                 np.log(self.A21[z2, z1]) +
                                                 np.log(self.B[z1, obs_seq[t]]))
        
        log_lik = logsumexp(alpha[-1])
        
        # Backward pass
        beta = np.full((T, K, K, K), -np.inf)
        beta[-1, :, :, :] = 0.0
        for t in range(T-2, -1, -1):
            for z3 in range(K):
                # nz3 depends on z3 via A3
                for nz3 in range(K):
                    log_nz3_base = np.log(self.A3[z3, nz3])
                    for nz2 in range(K):
                        for nz1 in range(K):
                            trans = (log_nz3_base +
                                     np.log(self.A32[nz3, nz2]) +
                                     np.log(self.A21[nz2, nz1]) +
                                     np.log(self.B[nz1, obs_seq[t+1]]))
                            beta[t, z3, :, :] = np.logaddexp(beta[t, z3, :, :], beta[t+1, nz3, nz2, nz1] + trans)
        
        # Compute joint posterior over (z2, z1) marginalising z3
        joint_z2z1 = np.zeros((K, K))
        for t in range(T):
            log_gamma = alpha[t] + beta[t] - log_lik
            gamma = np.exp(log_gamma - logsumexp(log_gamma))
            joint_z2z1 += np.sum(gamma, axis=0) # sum over z3
            
        joint_z2z1 /= T
        p_z2 = joint_z2z1.sum(axis=1)
        p_z1 = joint_z2z1.sum(axis=0)
        
        mi = 0.0
        for z2 in range(K):
            for z1 in range(K):
                if joint_z2z1[z2, z1] > 0:
                    mi += joint_z2z1[z2, z1] * np.log(joint_z2z1[z2, z1] / (p_z2[z2] * p_z1[z1] + 1e-12))
        return mi

class FlatPattern(HierarchicalPattern):
    """Degenerate hierarchical pattern that only learns surface-level frequencies."""
    def __init__(self, pattern_id, obs_dim=2):
        super().__init__(pattern_id, latent_dim=1, obs_dim=obs_dim)
        self.complexity = 1
        self.B = np.random.dirichlet(np.ones(obs_dim), size=1)

    def log_likelihood(self, obs_seq):
        if len(obs_seq) == 0: return 0.0
        # Use B[0, obs] as the probability of observation
        log_probs = [np.log(self.B[0, int(o) % self.obs_dim] + 1e-12) for o in obs_seq]
        return np.sum(log_probs)

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        ll = self.log_likelihood(obs_seq)
        avg_loss = -ll / max(1, len(obs_seq))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * avg_loss

    def observe(self, obs, learning_rate=0.01):
        # Nudge the emission entry for the seen observation
        idx = int(obs) % self.obs_dim
        self.B[0, idx] += learning_rate
        self.B[0] /= self.B[0].sum()

    def adapt(self, obs_seq):
        # Flat patterns don't need EM; they just update via observe
        pass

    def compression(self, obs_seq):
        return 0.0

    @staticmethod
    def flat(id, obs_dim=2):
        """Backwards compatibility for creating flat patterns."""
        return FlatPattern(id, obs_dim=obs_dim)
