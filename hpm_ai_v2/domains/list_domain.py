from hpm_ai_v2.domains.base import DomainConfig

LIST_CONCEPTS = [
    "RETURN",
    "CONST_1",
    "VAR_INP",
    "OP_ADD",
    "OP_MUL2",
    "OP_SUB",
    "LIST_INIT",
    "FOR_LOOP",
    "ITEM_ACCESS",
    "LIST_APPEND",
    "COND_IS_EVEN",
    "COND_IS_POSITIVE",
    "BLOCK_ELSE",
    "BLOCK_END",
    "MAP_START",
    "MAP_END",
]

class ListDomainConfig(DomainConfig):
    def __init__(self, s_dim: int = 20):
        super().__init__(concepts=LIST_CONCEPTS, s_dim=s_dim)
