from hpm_ai_v2.domains.base import DomainConfig

MATH_PHYSICS_CONCEPTS = [
    "RETURN",
    "CONST_1",
    "VAR_INP",
    "OP_ADD",
    "OP_MUL2",
    "OP_SUB",
    "OP_SQUARE",
    "OP_SQRT",
    "OP_DIV2",
    "LIST_INIT",
    "FOR_LOOP",
    "ITEM_ACCESS",
    "LIST_APPEND",
    "COND_IS_EVEN",
    "COND_IS_POSITIVE",
    "BLOCK_ELSE",
    "BLOCK_END",
]

class MathPhysicsDomainConfig(DomainConfig):
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=MATH_PHYSICS_CONCEPTS, s_dim=s_dim)
