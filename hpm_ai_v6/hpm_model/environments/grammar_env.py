import numpy as np
from typing import List
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.environments.base_env import BaseEnvironment

class GrammarEnvironment(BaseEnvironment):
    """
    Implements a hidden phase grammar environment.
    Generates cycles of transitions: A->B, B->C, C->A.
    """
    def __init__(self, objects: List[Cell]):
        if len(objects) < 3:
            raise ValueError("GrammarEnvironment requires at least 3 objects.")
        self.objects = objects

    def generate_episode(self, length: int) -> List[Cell]:
        seq = []
        cur_idx = 0
        for _ in range(length):
            seq.append(self.objects[cur_idx])
            cur_idx = (cur_idx + 1) % 3
        return seq
