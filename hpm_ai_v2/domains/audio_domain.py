"""
AudioDomainConfig — Configuration for Librosa-based audio transformations.
"""
from __future__ import annotations
import numpy as np
from typing import List
from hpm_ai_v2.domains.base import DomainConfig
from hfn.hfn import HFN

class AudioDomainConfig(DomainConfig):
    """
    Manifold structure for audio domain.
    """
    def __init__(self, sample_rate: int = 22050, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        concepts = [
            "PITCH_UP_2", "PITCH_DOWN_2", "VOLUME_UP", "VOLUME_DOWN",
            "TIME_STRETCH_2X", "REVERB", "LOW_PASS", "HIGH_PASS"
        ]
        # S_DIM=20, DIM=len(concepts), m_dim = S_DIM + DIM + S_DIM
        super().__init__(concepts, s_dim=20)

def get_audio_primitive_nodes(config: AudioDomainConfig) -> List[HFN]:
    """
    Instantiate one HFN node per primitive concept defined in the config.
    """
    nodes = []
    for i, concept in enumerate(config.concepts):
        mu = np.zeros(config.m_dim)
        # concept one-hot in middle slice
        mu[config.S_DIM + i] = 1.0
        node = HFN(
            mu=mu,
            sigma=np.ones(config.m_dim),
            id=f"audio_op_{concept}",
            use_diag=True
        )
        nodes.append(node)
    return nodes
