"""
ImageDomainConfig — Configuration for PIL-based image transformations.
"""
from __future__ import annotations
import numpy as np
from typing import List
from hpm_ai_v2.domains.base import DomainConfig
from hfn.hfn import HFN

class ImageDomainConfig(DomainConfig):
    """
    Manifold structure for image domain.
    """
    def __init__(self, image_size: int = 32):
        self.image_size = image_size
        self.pixels = image_size * image_size
        concepts = [
            "ROTATE_90", "ROTATE_180", "ROTATE_270",
            "FLIP_H", "FLIP_V",
            "BLUR", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN",
            "EDGE_DETECT", "CONTRAST_UP"
        ]
        # S_DIM=20, DIM=len(concepts), m_dim = S_DIM + DIM + S_DIM
        super().__init__(concepts, s_dim=20)

def get_primitive_nodes(config: ImageDomainConfig) -> List[HFN]:
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
            id=f"img_op_{concept}",
            use_diag=True
        )
        nodes.append(node)
    return nodes
