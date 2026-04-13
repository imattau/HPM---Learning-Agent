"""
State vector constants shared across hpm_ai_v2.

Ported from:
- experiment_unified_perception_action.py (CONCEPTS, S_DIM, DIM)
- experiment_generative_forward_model.py (STRUCT_DIMS)
"""
import numpy as np

CONCEPTS = [
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
]
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}

# Empirical state vector dimension
S_DIM = 20
# Action / concept vector dimension
DIM = len(CONCEPTS)
# Structural (code-structure) dimensions — predictable by forward model
# dims 0 (valid) + 10-16 (code structure flags)
STRUCT_DIMS = [0] + list(range(10, 17))
# Legacy slice form (used by some renderers)
STRUCTURE_DIMS = slice(10, 17)
