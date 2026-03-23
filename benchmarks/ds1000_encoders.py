"""Encoders for the DS-1000 5-level stack benchmark."""

import numpy as np

class DS1000L1Encoder:
    """L1: Syntax and Token Boilerplate.
    Simulates parsing the raw text/code by producing a 32-dim syntax vector.
    """
    feature_dim = 32
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        input_data, output_data = observation
        # Mock syntactic signature: hash-like projection from data values
        # In a real system, this would encode the raw code tokens / API string
        np.random.seed(int(np.sum(input_data) * 100) % 10000)
        syntax_vec = np.random.uniform(-1, 1, size=self.feature_dim)
        return [syntax_vec]


class DS1000L2Encoder:
    """L2: Structural Anatomy.
    Identifies specific objects (DataFrames vs Tensors vs missing values).
    Takes a simulated 8-dim structure feature vector and threads L1 epistemic state.
    """
    feature_dim = 16
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        input_data, output_data = observation
        l1_w, l1_loss = epistemic if epistemic else (1.0, 0.0)
        
        # Combine input and output structures (8+8 = 16) and blend in epistemic state
        # For simplicity in feature dimension matching, we will embed the 8-dim into 14, and append 2 for epistemic.
        embedded_struct = np.pad(input_data, (0, 6)) + np.pad(output_data, (0, 6)) * 0.5
        vec = np.concatenate([embedded_struct, [l1_w, l1_loss]])
        return [vec]


class DS1000L3Encoder:
    """L3: Relational Law.
    Represents the mathematical or logical transformation required.
    """
    feature_dim = 20
    max_steps_per_obs = 1

    def encode(self, observation: tuple, epistemic: tuple | None = None) -> list[np.ndarray]:
        input_data, output_data = observation
        l2_w, l2_loss = epistemic if epistemic else (1.0, 0.0)
        
        # Relational law is primarily the delta (transformation)
        delta = output_data - input_data
        
        # Pad delta (8-dim) to 18-dim, then append epistemic (2-dim)
        padded_delta = np.pad(delta, (0, 10))
        vec = np.concatenate([padded_delta, [l2_w, l2_loss]])
        return [vec]
