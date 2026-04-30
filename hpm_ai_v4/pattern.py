import numpy as np
import warnings


class HierarchicalPattern:
    """Single-level HMM. Hierarchy emerges from stacking patterns via get_top_state()."""

    def __init__(self, pattern_id, latent_dim=2, obs_dim=6):
        if latent_dim > 16:
            warnings.warn(
                f"HierarchicalPattern created with latent_dim={latent_dim} > 16. "
                "Very wide HMMs are experimental; prefer moderate widths and benchmark the effect.",
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

        self._compression_cache: float | None = None
        self._forward_cache_key: tuple[int, ...] | None = None
        self._forward_cache_alpha: np.ndarray | None = None
        self._forward_cache_scales: np.ndarray | None = None
        self._refresh_log_cache()

    def _refresh_log_cache(self):
        self.logA = np.log(self.A + 1e-12).astype(np.float32)
        self.logB = np.log(self.B + 1e-12).astype(np.float32)
        self._compression_cache = None
        self._forward_cache_key = None
        self._forward_cache_alpha = None
        self._forward_cache_scales = None

    def _obs_key(self, obs_seq):
        return tuple(int(o) % self.obs_dim for o in obs_seq)

    def _forward(self, obs_seq):
        """Scaled forward pass. Returns alpha (T, K) and scales (T,)."""
        if len(obs_seq) == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32), np.zeros(0, dtype=np.float32)

        key = self._obs_key(obs_seq)
        if key == self._forward_cache_key and self._forward_cache_alpha is not None and self._forward_cache_scales is not None:
            return self._forward_cache_alpha, self._forward_cache_scales

        obs = np.asarray(key, dtype=np.int32)
        T, K = len(obs), self.latent_dim
        alpha = np.zeros((T, K), dtype=np.float32)
        A = self.A
        B = self.B
        alpha[0] = self.pi * B[:, obs[0]]
        s = alpha[0].sum()
        if not np.isfinite(s) or s <= 0.0:
            alpha[0] = np.ones(K, dtype=np.float32) / K
            s = 1.0
        alpha[0] /= s + 1e-12
        scales = np.empty(T, dtype=np.float32)
        scales[0] = s
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
            s = alpha[t].sum()
            if not np.isfinite(s) or s <= 0.0:
                alpha[t] = np.ones(K, dtype=np.float32) / K
                s = 1.0
            alpha[t] /= s + 1e-12
            scales[t] = s

        self._forward_cache_key = key
        self._forward_cache_alpha = alpha
        self._forward_cache_scales = scales
        return alpha, scales

    def _forward_backward(self, obs_seq):
        """Standard Baum-Welch. Returns gamma (T, K) and xi (T-1, K, K)."""
        T, K = len(obs_seq), self.latent_dim
        alpha, scales = self._forward(obs_seq)

        beta = np.ones((T, K), dtype=np.float32)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * (beta[t + 1] * self.B[:, int(obs_seq[t + 1]) % self.obs_dim])).sum(axis=1)
            s = beta[t].sum()
            if not np.isfinite(s) or s <= 0.0:
                beta[t] = np.ones(K, dtype=np.float32) / K
                continue
            beta[t] /= s + 1e-12

        gamma = alpha * beta
        gamma = np.nan_to_num(gamma, nan=1.0 / max(1, K), posinf=1.0 / max(1, K), neginf=1.0 / max(1, K))
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

        xi = np.zeros((T - 1, K, K), dtype=np.float32)
        for t in range(T - 1):
            obs_next = int(obs_seq[t + 1]) % self.obs_dim
            xi[t] = alpha[t:t + 1].T * self.A * (beta[t + 1] * self.B[:, obs_next])
            xi[t] = np.nan_to_num(xi[t], nan=0.0, posinf=0.0, neginf=0.0)
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
            return 0.0
        window = list(obs_seq[-window_size:])
        gamma, xi = self._forward_backward(window)
        K = self.latent_dim

        new_A = xi.sum(axis=0)
        new_B = np.zeros((K, self.obs_dim), dtype=np.float32)
        for t, o in enumerate(window):
            new_B[:, int(o) % self.obs_dim] += gamma[t]
        new_pi = gamma[0].copy()

        # Normalise
        new_A = np.nan_to_num(new_A, nan=1.0 / max(1, K), posinf=1.0 / max(1, K), neginf=0.0)
        new_B = np.nan_to_num(new_B, nan=0.0, posinf=0.0, neginf=0.0)
        new_pi = np.nan_to_num(new_pi, nan=1.0 / max(1, K), posinf=1.0 / max(1, K), neginf=0.0)
        new_A /= new_A.sum(axis=1, keepdims=True) + 1e-12
        new_B /= new_B.sum(axis=1, keepdims=True) + 1e-12
        new_pi /= new_pi.sum() + 1e-12

        # Exponential smoothing blend
        alpha = 0.7
        self.A = (alpha * new_A + (1 - alpha) * self.A).astype(np.float32)
        self.B = (alpha * new_B + (1 - alpha) * self.B).astype(np.float32)
        self.pi = (alpha * new_pi + (1 - alpha) * self.pi).astype(np.float32)

        # Re-normalise
        self.A = np.nan_to_num(self.A, nan=1.0 / max(1, K), posinf=1.0 / max(1, K), neginf=0.0)
        self.B = np.nan_to_num(self.B, nan=1.0 / max(1, self.obs_dim), posinf=1.0 / max(1, self.obs_dim), neginf=0.0)
        self.pi = np.nan_to_num(self.pi, nan=1.0 / max(1, K), posinf=1.0 / max(1, K), neginf=0.0)
        self.A /= self.A.sum(axis=1, keepdims=True) + 1e-12
        self.B /= self.B.sum(axis=1, keepdims=True) + 1e-12
        self.pi /= self.pi.sum() + 1e-12

        self._refresh_log_cache()
        return self.log_likelihood(window)

    def update_running_loss(self, obs_seq, lambda_l=0.1, ll: float | None = None):
        if len(obs_seq) == 0:
            return
        if ll is None:
            ll = self.log_likelihood(obs_seq[-30:])
        if not np.isfinite(ll):
            ll = 0.0
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
        pred = np.nan_to_num(pred, nan=1.0 / max(1, self.obs_dim), posinf=1.0 / max(1, self.obs_dim), neginf=0.0)
        pred /= pred.sum() + 1e-12
        return pred.astype(np.float32)

    def compression(self):
        """
        Mutual information I(z_t; z_{t+1}) under stationary distribution of A.
        Measures how much the transition structure compresses state uncertainty.
        No obs_seq argument — computed analytically from A.
        """
        if self._compression_cache is not None:
            return self._compression_cache

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
        self._compression_cache = max(0.0, H_state - H_cond)
        return self._compression_cache

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

    def update_running_loss(self, obs_seq, lambda_l=0.1, ll: float | None = None):
        if ll is None:
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
