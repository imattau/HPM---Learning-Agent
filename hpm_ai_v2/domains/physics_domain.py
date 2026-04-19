"""
PhysicsDomainConfig — manifold configuration for classical mechanics.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional
from hpm_ai_v2.domains.base import DomainConfig

class PhysicsDomainConfig(DomainConfig):
    """Manifold structure for physics (kinematics, dynamics)."""
    def __init__(self, concepts: Optional[List[str]] = None, s_dim: int = 20):
        if concepts is None:
            concepts = [
                "VAR_MASS", "VAR_ACCEL", "VAR_FORCE", "VAR_VELOCITY", "VAR_TIME", "VAR_DIST",
                "OP_NEWTON_2", "OP_VELOCITY", "OP_ACCEL", "OP_GRAVITY",
                "UNIT_KG", "UNIT_MS2", "UNIT_N", "UNIT_M", "UNIT_S",
                "TRANS_CALC_FORCE", "TRANS_CALC_VELOCITY", "TRANS_CALC_ACCEL",
                "PHYS_LAW", "PHYS_PROBLEM"
            ]
        super().__init__(concepts, s_dim=s_dim)

    @property
    def domain_type(self) -> str:
        return "physics"
