"""
FluidDomainConfig — Configuration for the fluid dynamics domain (SP95).
"""
from __future__ import annotations
from hpm_ai_v2.domains.base import DomainConfig

FLUID_CONCEPTS = [
    "MOMENTUM_FLUX",   # ρ * Q**2 / A
    "TORQUE",          # F * L
    "SIGN",            # sign(x)
    "MULTIPLY",        # a * b
    "SINE",            # sin(θ)
    "COSINE",          # cos(θ)
]

class FluidDomainConfig(DomainConfig):
    """
    Configuration for the fluid dynamics domain.
    Used for the Inverse Sprinkler experiment (SP95).
    """
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=FLUID_CONCEPTS, s_dim=s_dim)
