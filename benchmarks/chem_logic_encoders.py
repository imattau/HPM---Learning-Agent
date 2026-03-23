"""Encoders for the Chem-Logic molecular benchmark (SP12)."""

import numpy as np

class ChemLogicL1Encoder:
    """L1: SMILES Distribution.
    Encodes the raw character frequency of the SMILES string.
    """
    feature_dim = 32
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, product = observation
        # Character-based frequency vector
        text = reactant.smiles + product.smiles
        vec = np.zeros(self.feature_dim)
        for char in text:
            idx = ord(char) % self.feature_dim
            vec[idx] += 1.0
        total = vec.sum()
        if total > 0: vec /= total
        return [vec]

class ChemLogicL2Encoder:
    """L2: Structural Anatomy.
    Identifies functional groups and threads L1 epistemic state.
    """
    feature_dim = 16
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, product = observation
        l1_w, l1_loss = epistemic if epistemic else (1.0, 0.0)
        
        # Combine multi-hot features of reactant and product
        combined = np.concatenate([reactant.features, product.features]) # 7+7 = 14
        # Pad to 14, then append 2 epistemic
        vec = np.concatenate([combined, [l1_w, l1_loss]])
        return [vec]

class ChemLogicL3Encoder:
    """L3: Relational Law (The Reaction).
    Represents the transformation delta between reactant and product.
    """
    feature_dim = 20
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        reactant, product = observation
        l2_w, l2_loss = epistemic if epistemic else (1.0, 0.0)
        
        # The 'Law' is the delta in functional groups
        delta = product.features - reactant.features # 7-dim
        
        # Pad to 18, then append 2 epistemic
        padded = np.pad(delta, (0, 11))
        vec = np.concatenate([padded, [l2_w, l2_loss]])
        return [vec]
