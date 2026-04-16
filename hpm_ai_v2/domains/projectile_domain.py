"""
ProjectileDomainConfig — Configuration for the projectile motion domain (SP96).
"""
from __future__ import annotations
from hpm_ai_v2.domains.base import DomainConfig

PROJECTILE_CONCEPTS = [
    "VAR_THETA",       # Angle (rad)
    "VAR_V0",          # Initial velocity (m/s)
    "VAR_T",           # Time (s)
    "VAR_G",           # Gravity (9.8 m/s^2)
    "VAR_Z1",          # Spurious 1 (Correlated with Y)
    "VAR_Z2",          # Spurious 2 (Correlated with T)
    "VAR_Z3",          # Spurious 3 (Random)
    "OP_MUL",          # b = pop(); a = pop(); push(a * b)
    "OP_ADD",          # b = pop(); a = pop(); push(a + b)
    "OP_SUB",          # b = pop(); a = pop(); push(a - b)
    "OP_NEG",          # a = pop(); push(-a)
    "OP_SIN",          # res = sin(res)
    "OP_SQUARE",       # res = res**2
    "OP_CONST_05",     # push(0.5)
]

class ProjectileDomainConfig(DomainConfig):
    """
    Configuration for the projectile motion domain.
    Used for the SP96 experiment.
    """
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=PROJECTILE_CONCEPTS, s_dim=s_dim)
        self.concept_idx = {c: i for i, c in enumerate(self.concepts)}
