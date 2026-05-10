import numpy as np
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.environments.base_env import BaseEnvironment

class NoisyEnvironment(BaseEnvironment):
    """
    Wrapper that adds stochastic noise to an underlying environment.
    Randomly replaces observations with a certain probability.
    """
    def __init__(self, base_env: BaseEnvironment, noise_level: float = 0.2, all_objects: List[Cell] = None):
        self.base_env = base_env
        self.noise_level = noise_level
        self.all_objects = all_objects or []

    def generate_episode(self, length: int) -> List[Cell]:
        base_seq = self.base_env.generate_episode(length)
        if not self.all_objects:
            return base_seq
            
        noisy_seq = []
        for cell in base_seq:
            if np.random.rand() < self.noise_level:
                # Replace with random object from the pool
                noisy_seq.append(np.random.choice(self.all_objects))
            else:
                noisy_seq.append(cell)
        return noisy_seq
