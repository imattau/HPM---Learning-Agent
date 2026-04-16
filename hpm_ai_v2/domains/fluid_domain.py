"""
FluidDomainConfig — Configuration for the fluid dynamics domain (SP95).
"""
from __future__ import annotations
from hpm_ai_v2.domains.base import DomainConfig

FLUID_CONCEPTS = [
    "VAR_Q",           # Load Q
    "VAR_RHO",         # Load rho
    "VAR_L",           # Load L
    "VAR_THETA",       # Load theta
    "OP_MUL_Q",        # res = res * Q
    "OP_MUL_RHO",      # res = res * rho
    "OP_SQUARE",       # res = res**2
    "OP_SIN",          # res = sin(res)
    "OP_SIGN",         # res = sign(res)
]

class FluidDomainConfig(DomainConfig):
    """
    Configuration for the fluid dynamics domain.
    Used for the Inverse Sprinkler experiment (SP95).
    """
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=FLUID_CONCEPTS, s_dim=s_dim)
