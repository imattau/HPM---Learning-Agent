"""
Code loader for the Python code experiment.

Fixed vocabulary of ~70 tokens (Python keywords, operators, punctuation, builtins).
Generates synthetic Python code snippet observations with masked context windows.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed vocabulary (~70 tokens)
# ---------------------------------------------------------------------------

VOCAB: list[str] = [
    # Keywords (28)
    "if", "for", "while", "def", "class", "return", "import", "from",
    "with", "try", "except", "else", "elif", "pass", "break", "continue",
    "lambda", "yield", "and", "or", "not", "in", "is", "as", "raise",
    "assert", "del", "global",
    # Operators (18)
    "=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//",
    "%", "**", "&", "|", "^", "~",
    # Punctuation (9)
    "(", ")", ":", ",", ".", "[", "]", "{", "}",
    # Builtins (15)
    "print", "len", "range", "type", "input", "open", "map", "filter",
    "zip", "enumerate", "int", "str", "float", "bool", "list",
]

assert len(VOCAB) == 70, f"Vocab size {len(VOCAB)} != 70"
assert len(set(VOCAB)) == 70, "Vocabulary has duplicate entries"

VOCAB_INDEX: dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)
D: int = VOCAB_SIZE  # 70


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_token(token: str) -> np.ndarray:
    """One-hot encode a token; unknown tokens get zero vector."""
    vec = np.zeros(VOCAB_SIZE, dtype=np.float64)
    idx = VOCAB_INDEX.get(token)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def compose_context_node(
    left2: str,
    left1: str,
    right1: str,
    right2: str,
) -> np.ndarray:
    """
    Compose 4-slot context window into a single D=70 vector via slot-weighted recombination.

    Slot weights encode proximity to the masked position:
        left2=0.20, left1=0.35, right1=0.35, right2=0.10  (sum=1.0)

    Unknown tokens (not in VOCAB) contribute zero weight.
    Returns shape (D,) = (70,) float64 vector.
    """
    return (
        0.20 * _encode_token(left2)
        + 0.35 * _encode_token(left1)
        + 0.35 * _encode_token(right1)
        + 0.10 * _encode_token(right2)
    )


# ---------------------------------------------------------------------------
# Snippet generation
# ---------------------------------------------------------------------------

# Category word sets
_KEYWORDS_CONTROL = ["if", "for", "while", "elif", "else", "break", "continue", "pass"]
_KEYWORDS_DEF = ["def", "class", "return", "lambda", "yield"]
_KEYWORDS_IMPORT = ["import", "from", "as", "with", "try", "except"]
_KEYWORDS_LOGIC = ["and", "or", "not", "in", "is", "assert", "raise", "del", "global"]
_OPERATORS_CMP = ["==", "!=", "<", ">", "<=", ">="]
_OPERATORS_ARITH = ["+", "-", "*", "/", "//", "%", "**"]
_OPERATORS_BIT = ["&", "|", "^", "~"]
_OPERATORS_ASSIGN = ["="]
_PUNCT_OPEN = ["(", "[", "{"]
_PUNCT_CLOSE = [")", "]", "}"]
_PUNCT_OTHER = [":", ",", ".", ]
_BUILTINS_IO = ["print", "input", "open"]
_BUILTINS_TYPE = ["int", "str", "float", "bool", "list"]
_BUILTINS_ITER = ["range", "len", "type", "map", "filter", "zip", "enumerate"]


def _obs(left2: str, left1: str, right1: str, right2: str,
         true_token: str, category: str) -> tuple[np.ndarray, str, str]:
    vec = compose_context_node(left2, left1, right1, right2)
    return vec, true_token, category


def generate_code_snippets(
    seed: int = 42,
) -> list[tuple[np.ndarray, str, str]]:
    """
    Generate synthetic Python code snippet observations.

    Returns list of (context_vector, true_token, category).
    Categories: control_flow, functions, data, builtins.
    Labels are for evaluation only — never fed to the Observer.
    """
    observations: list[tuple[np.ndarray, str, str]] = []

    # --- control_flow category ---
    # Template: "[MASK:if/for/while] COND : BODY"
    for kw in ["if", "for", "while"]:
        for cmp in _OPERATORS_CMP:
            observations.append(_obs("=", "(", cmp, ":", kw, "control_flow"))
        for logic in ["and", "or", "not"]:
            observations.append(_obs("=", "(", logic, ":", kw, "control_flow"))

    # Template: "for VAR in [MASK:range/enumerate/zip] ( ..."
    for builtin in ["range", "enumerate", "zip"]:
        for _ in range(10):
            observations.append(_obs("for", "in", "(", ")", builtin, "control_flow"))

    # Template: "[MASK:break/continue/pass] after condition"
    for kw in ["break", "continue", "pass"]:
        for _ in range(15):
            observations.append(_obs("if", ":", kw, "else", kw, "control_flow"))

    # Template: "elif [MASK:else] condition"
    for _ in range(20):
        observations.append(_obs("elif", ":", "pass", "else", "else", "control_flow"))
    for _ in range(20):
        observations.append(_obs("if", ":", "pass", "else", "elif", "control_flow"))

    # --- functions category ---
    # Template: "[MASK:def/class] NAME ( ) :"
    for kw in ["def", "class"]:
        for _ in range(30):
            observations.append(_obs("(", ")", ":", "(", kw, "functions"))

    # Template: "[MASK:return/yield/lambda] VALUE"
    for kw in ["return", "yield"]:
        for op in _OPERATORS_ARITH[:4]:
            observations.append(_obs("def", ":", kw, "(", kw, "functions"))
    for _ in range(20):
        observations.append(_obs("def", ":", "lambda", ":", "lambda", "functions"))

    # Template: "with [MASK:try/except/as]"
    for kw in ["try", "except", "as", "with"]:
        for _ in range(15):
            observations.append(_obs("(", ")", kw, ":", kw, "functions"))

    # Template: "[MASK:import/from] MODULE"
    for kw in ["import", "from"]:
        for _ in range(20):
            observations.append(_obs("(", ")", kw, "as", kw, "functions"))

    # --- data category ---
    # Template: "VAR [MASK:=] VALUE"
    for _ in range(50):
        observations.append(_obs("(", ")", "=", "(", "=", "data"))

    # Template: "VAR [MASK:==,!=,<,>,<=,>=] VALUE"
    for op in _OPERATORS_CMP:
        for _ in range(15):
            observations.append(_obs("(", ")", op, "(", op, "data"))

    # Template: "A [MASK:+,-,*,/] B"
    for op in _OPERATORS_ARITH:
        for _ in range(15):
            observations.append(_obs("(", "(", op, ")", op, "data"))

    # Template: "[MASK:[,{,(] items ]"
    for punct in ["[", "{", "("]:
        for _ in range(20):
            observations.append(_obs("=", ")", punct, ",", punct, "data"))

    # Template: "LIST [MASK:.] METHOD"
    for _ in range(20):
        observations.append(_obs(")", ")", ".", "(", ".", "data"))

    # Template: "VAR [MASK:in] ITER"
    for _ in range(25):
        observations.append(_obs("for", "(", "in", "range", "in", "data"))

    # --- builtins category ---
    # Template: "[MASK:print/input/open] ( )"
    for builtin in _BUILTINS_IO:
        for _ in range(25):
            observations.append(_obs("=", "(", builtin, "(", builtin, "builtins"))

    # Template: "[MASK:int/str/float/bool/list] ( VAR )"
    for builtin in _BUILTINS_TYPE:
        for _ in range(20):
            observations.append(_obs("=", "(", builtin, "(", builtin, "builtins"))

    # Template: "[MASK:range/len/type/map/filter/zip/enumerate] ( )"
    for builtin in _BUILTINS_ITER:
        for _ in range(15):
            observations.append(_obs("for", "in", builtin, "(", builtin, "builtins"))

    # Template: "len ( [MASK:list] )"
    for _ in range(20):
        observations.append(_obs("len", "(", "list", ")", "list", "builtins"))

    # Shuffle and return
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(observations))
    observations = [observations[i] for i in idx]
    return observations
