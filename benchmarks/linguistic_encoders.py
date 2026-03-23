"""Linguistic Encoders for Register Shift Benchmark (SP14)."""

import numpy as np

class LinguisticL1Encoder:
    """L1: Syntax (Character Distribution)."""
    feature_dim = 32
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, product = observation
        text = reactant.text + product.text
        vec = np.zeros(self.feature_dim)
        for char in text:
            idx = ord(char) % self.feature_dim
            vec[idx] += 1.0
        total = vec.sum()
        if total > 0: vec /= total
        return [vec]

class LinguisticL2Encoder:
    """L2: Structural Anatomy (Semantic Root).
    Threads L1 epistemic state.
    """
    feature_dim = 16
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, _ = observation # Semantic anatomy of the root
        l1_w, l1_loss = epistemic if epistemic else (1.0, 0.0)
        
        # Root features (5-dim)
        # Pad to 14, then append 2 epistemic
        vec = np.pad(reactant.features, (0, 9))
        vec = np.concatenate([vec, [l1_w, l1_loss]])
        return [vec]

class LinguisticL3Encoder:
    """L3: Relational Law (The Register Transformation)."""
    feature_dim = 20
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, product = observation
        l2_w, l2_loss = epistemic if epistemic else (1.0, 0.0)
        
        # Law is the semantic delta
        delta = product.features - reactant.features # 5-dim
        
        # Pad to 18, then append 2 epistemic
        vec = np.pad(delta, (0, 13))
        vec = np.concatenate([vec, [l2_w, l2_loss]])
        return [vec]
