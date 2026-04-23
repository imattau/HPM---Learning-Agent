import numpy as np
from scipy.special import logsumexp

class HierarchicalPattern:
    """Generative model with three latent levels, as in Appendix E.1 of the HPM paper."""
    def __init__(self, pattern_id, latent_dim=2, obs_dim=2):
        self.id = pattern_id
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Level 3 (highest) dynamics: p(z3_t | z3_{t-1})
        self.A3 = np.random.dirichlet(np.ones(latent_dim), size=latent_dim)
        self.pi3 = np.random.dirichlet(np.ones(latent_dim))

        # Transition from level 3 to level 2: p(z2_t | z3_t)
        self.A32 = np.random.dirichlet(np.ones(latent_dim), size=latent_dim)

        # Transition from level 2 to level 1: p(z1_t | z2_t)
        self.A21 = np.random.dirichlet(np.ones(latent_dim), size=latent_dim)

        # Emission from level 1 to observation: p(x_t | z1_t)
        self.B = np.random.dirichlet(np.ones(obs_dim), size=latent_dim)

        # Replicator weight in population
        self.weight = 0.0
        # Running loss (epistemic)
        self.running_loss = 0.0
        # For developmental stage / complexity tracking
        self.complexity = 3   # number of latent levels

    def log_likelihood(self, obs_seq):
        """Compute log p(x1:T) under the three-level HMM using dynamic programming."""
        if len(obs_seq) == 0:
            return 0.0
        
        K = self.latent_dim
        T = len(obs_seq)
        # alpha[t, z3, z2, z1]
        alpha = np.full((T, K, K, K), -np.inf)
        
        # Initial step
        for z3 in range(K):
            for z2 in range(K):
                for z1 in range(K):
                    logp = (np.log(self.pi3[z3] + 1e-12) +
                            np.log(self.A32[z3, z2] + 1e-12) +
                            np.log(self.A21[z2, z1] + 1e-12) +
                            np.log(self.B[z1, obs_seq[0]] + 1e-12))
                    alpha[0, z3, z2, z1] = logp
                    
        # Recursion
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
                        
        return logsumexp(alpha[-1])

    def observe(self, obs, learning_rate=0.01):
        """Enhanced online update: nudges parameters toward observations and inferred latent states."""
        if self.complexity < 2: 
            return

        obs_idx = int(obs) % self.obs_dim
        
        # 1. Nudge Emission Matrix (B)
        # We don't know the latent state z1, so we nudge all z1 rows slightly
        # but favor those that already predict this observation (reinforcement)
        for z1 in range(self.latent_dim):
            # Reinforcement nudge
            influence = self.B[z1, obs_idx] + 0.1
            self.B[z1, obs_idx] += learning_rate * influence
            self.B[z1] /= self.B[z1].sum()
            
        # 2. Nudge Transition Matrices (A3, A32, A21)
        # Favor stability and consistent transitions
        for i in range(self.latent_dim):
            # Nudge self-transitions to favor persistence (common in structured data)
            self.A3[i, i] += learning_rate * 0.5
            self.A3[i] /= self.A3[i].sum()
            
            self.A32[i, i] += learning_rate * 0.2
            self.A32[i] /= self.A32[i].sum()
            
            self.A21[i, i] += learning_rate * 0.5
            self.A21[i] /= self.A21[i].sum()

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
            # Marginalise over z3, z2 to get p(z1)
            p_z1 = np.sum(belief, axis=(0, 1))
            # Next observation distribution: p(x) = sum_z1 p(x|z1)p(z1)
            return p_z1 @ self.B
        else:
            # Flat pattern: return emission distribution from B[0]
            return self.B[0]

    def predict_next(self, obs_seq):
        """Predict the most likely next observation."""
        dist = self.predict_next_distribution(obs_seq)
        return int(np.argmax(dist))

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        ll = self.log_likelihood(obs_seq)
        avg_loss = -ll / max(1, len(obs_seq))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * avg_loss

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

    @staticmethod
    def flat(id, obs_dim=2):
        """Create a flat categorical pattern for baseline comparison."""
        p = HierarchicalPattern(id, obs_dim=obs_dim)
        p.complexity = 1
        # Emissions: p(x | z1), but z1 is always 0 for flat
        p.B = np.random.dirichlet(np.ones(obs_dim), size=p.latent_dim)
        
        def flat_ll(obs_seq):
            if len(obs_seq) == 0: return 0.0
            # Use B[0, obs] as the probability of observation
            log_probs = [np.log(p.B[0, int(o) % p.obs_dim] + 1e-12) for o in obs_seq]
            return np.sum(log_probs)
            
        def flat_update_loss(obs_seq, lambda_l=0.1):
            ll = flat_ll(obs_seq)
            avg_loss = -ll / max(1, len(obs_seq))
            p.running_loss = (1 - lambda_l) * p.running_loss + lambda_l * avg_loss

        def flat_observe(obs, learning_rate=0.01):
            # Nudge the emission entry for the seen observation
            idx = int(obs) % p.obs_dim
            p.B[0, idx] += learning_rate
            p.B[0] /= p.B[0].sum()

        p.log_likelihood = flat_ll
        p.update_running_loss = flat_update_loss
        p.observe = flat_observe
        p.compression = lambda obs_seq: 0.0
        return p
