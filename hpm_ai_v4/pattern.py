import numpy as np
import warnings


class HierarchicalPattern:
    """Single-level HMM. Hierarchy emerges from stacking patterns via get_top_state()."""

    def __init__(self, pattern_id, latent_dim=2, obs_dim=6):
        if latent_dim > 4:
            warnings.warn(
                f"HierarchicalPattern created with latent_dim={latent_dim} > 4. "
                "HPM architecture requires small-K (max 4). Use depth, not width.",
                stacklevel=2,
            )
        self.id = pattern_id
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.complexity = 1  # single-level; kept for DevelopmentalStage compatibility

        self.A = np.random.dirichlet(np.ones(latent_dim), size=latent_dim).astype(np.float32)
        self.B = np.random.dirichlet(np.ones(obs_dim), size=latent_dim).astype(np.float32)
        self.pi = np.random.dirichlet(np.ones(latent_dim)).astype(np.float32)

        self.running_loss = 0.0
        self.weight = 1.0
        self.creation_step = 0

        self._refresh_log_cache()

    def _refresh_log_cache(self):
        self.logA = np.log(self.A + 1e-12).astype(np.float32)
        self.logB = np.log(self.B + 1e-12).astype(np.float32)

    def _forward(self, obs_seq):
        """Scaled forward pass. Returns alpha (T, K) and scales (T,)."""
        T, K = len(obs_seq), self.latent_dim
        alpha = np.zeros((T, K), dtype=np.float32)
        alpha[0] = self.pi * self.B[:, int(obs_seq[0]) % self.obs_dim]
        s = alpha[0].sum()
        alpha[0] /= s + 1e-12
        scales = [s]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, int(obs_seq[t]) % self.obs_dim]
            s = alpha[t].sum()
            alpha[t] /= s + 1e-12
            scales.append(s)
        return alpha, np.array(scales, dtype=np.float32)

    def _forward_backward(self, obs_seq):
        """Standard Baum-Welch. Returns gamma (T, K) and xi (T-1, K, K)."""
        T, K = len(obs_seq), self.latent_dim
        alpha, scales = self._forward(obs_seq)

        beta = np.ones((T, K), dtype=np.float32)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * (beta[t + 1] * self.B[:, int(obs_seq[t + 1]) % self.obs_dim])).sum(axis=1)
            s = beta[t].sum()
            beta[t] /= s + 1e-12

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

        xi = np.zeros((T - 1, K, K), dtype=np.float32)
        for t in range(T - 1):
            obs_next = int(obs_seq[t + 1]) % self.obs_dim
            xi[t] = alpha[t:t + 1].T * self.A * (beta[t + 1] * self.B[:, obs_next])
            xi[t] /= xi[t].sum() + 1e-12

        return gamma, xi

    def log_likelihood(self, obs_seq):
        """Compute log p(x_{1:T}) via scaled forward algorithm."""
        if len(obs_seq) == 0:
            return 0.0
        _, scales = self._forward(obs_seq)
        return float(np.sum(np.log(scales + 1e-12)))

    def update_parameters_online(self, obs_seq, window_size=100):
        """Windowed Baum-Welch EM update with exponential smoothing."""
        if len(obs_seq) < 5:
            return
        window = list(obs_seq[-window_size:])
        gamma, xi = self._forward_backward(window)
        K = self.latent_dim

        new_A = xi.sum(axis=0)
        new_B = np.zeros((K, self.obs_dim), dtype=np.float32)
        for t, o in enumerate(window):
            new_B[:, int(o) % self.obs_dim] += gamma[t]
        new_pi = gamma[0].copy()

        # Normalise
        new_A /= new_A.sum(axis=1, keepdims=True) + 1e-12
        new_B /= new_B.sum(axis=1, keepdims=True) + 1e-12
        new_pi /= new_pi.sum() + 1e-12

        # Exponential smoothing blend
        alpha = 0.7
        self.A = (alpha * new_A + (1 - alpha) * self.A).astype(np.float32)
        self.B = (alpha * new_B + (1 - alpha) * self.B).astype(np.float32)
        self.pi = (alpha * new_pi + (1 - alpha) * self.pi).astype(np.float32)

        # Re-normalise
        self.A /= self.A.sum(axis=1, keepdims=True) + 1e-12
        self.B /= self.B.sum(axis=1, keepdims=True) + 1e-12
        self.pi /= self.pi.sum() + 1e-12

        self._refresh_log_cache()

    def update_running_loss(self, obs_seq, lambda_l=0.1):
        if len(obs_seq) == 0:
            return
        ll = self.log_likelihood(obs_seq[-30:])
        loss = -ll / max(1, len(obs_seq[-30:]))
        self.running_loss = (1 - lambda_l) * self.running_loss + lambda_l * loss

    def get_top_state(self, obs_seq):
        """Most probable current latent state given obs_seq."""
        if not obs_seq:
            return int(np.argmax(self.pi))
        alpha, _ = self._forward(obs_seq[-20:])
        return int(np.argmax(alpha[-1]))

    def predict_next(self, obs_seq):
        """Argmax of predictive distribution over next observation."""
        return int(np.argmax(self.predict_next_distribution(obs_seq)))

    def predict_next_distribution(self, obs_seq):
        """Predictive distribution P(x_{t+1} | x_{1:t}) as (obs_dim,) array."""
        if not obs_seq:
            next_state = self.pi @ self.A
        else:
            alpha, _ = self._forward(obs_seq[-20:])
            next_state = alpha[-1] @ self.A
        pred = next_state @ self.B
        pred /= pred.sum() + 1e-12
        return pred.astype(np.float32)

    def compression(self):
        """
        Mutual information I(z_t; z_{t+1}) under stationary distribution of A.
        Measures how much the transition structure compresses state uncertainty.
        No obs_seq argument — computed analytically from A.
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        idx = np.where(np.isclose(np.real(eigvals), 1.0))[0]
        if len(idx) == 0:
            stat = np.ones(self.latent_dim, dtype=np.float32) / self.latent_dim
        else:
            stat = np.abs(np.real(eigvecs[:, idx[0]])).astype(np.float32)
            stat /= stat.sum() + 1e-12

        H_state = float(-np.sum(stat * np.log(stat + 1e-12)))
        H_cond = float(sum(
            stat[i] * (-np.sum(self.A[i] * np.log(self.A[i] + 1e-12)))
            for i in range(self.latent_dim)
        ))
        return max(0.0, H_state - H_cond)

    def predictive_entropy(self, obs_seq):
        """Entropy of the predictive distribution over the next observation."""
        pred = self.predict_next_distribution(obs_seq)
        return float(-np.sum(pred * np.log(pred + 1e-12)))

    @staticmethod
    def flat(id, obs_dim=2):
        """Factory for creating FlatPattern instances."""
        return FlatPattern(id, obs_dim=obs_dim)


class FlatPattern(HierarchicalPattern):
    """Degenerate hierarchical pattern that only learns surface-level frequencies."""
    def __init__(self, pattern_id, obs_dim=2):
        super().__init__(pattern_id, latent_dim=1, obs_dim=obs_dim)
        self.complexity = 1
        self.B = np.random.dirichlet(np.ones(obs_dim), size=1).astype(np.float32)

    def log_likelihood(self, obs_seq):
        if len(obs_seq) == 0: return 0.0
        # Use B[0, obs] as the probability of observation
        log_probs = [np.log(self.B[0, int(o) % self.obs_dim] + 1e-12) for o in obs_seq]
        return float(np.sum(log_probs))

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

    def compression(self):
        return 0.0

    @staticmethod
    def flat(id, obs_dim=2):
        """Backwards compatibility for creating flat patterns."""
        return FlatPattern(id, obs_dim=obs_dim)
