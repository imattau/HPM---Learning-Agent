"""
Code world model — HPM-aligned world model for Python code experiment.

Five sub-trees, all nodes at D=70:
  Atomic token nodes (70): one per vocabulary token, mu = one-hot
  Syntax sub-tree: keywords + operators + punctuation hierarchy
  Type sub-tree: int, str, float, bool, list, dict, tuple, None
  Builtin sub-tree: print, len, range, type, input, open, map, filter, zip, enumerate
  Pattern sub-tree: for_loop, if_condition, function_def, assignment (recombined pairs)
  Sentence priors (~15): short synthetic code snippet mus

All composed node mus are equal-weight recombinations of child mus unless noted.
"""
from __future__ import annotations

import numpy as np

from hfn import HFN, Forest
from hpm_fractal_node.code.code_loader import D, VOCAB_INDEX, VOCAB_SIZE, VOCAB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(token: str) -> np.ndarray:
    """One-hot vector at D=70 for a vocabulary token."""
    vec = np.zeros(D, dtype=np.float64)
    idx = VOCAB_INDEX.get(token)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _recombine(*mus: np.ndarray) -> np.ndarray:
    """Equal-weight recombination of mus. Returns float64."""
    stacked = np.stack(mus, axis=0)
    return np.mean(stacked, axis=0).astype(np.float64)


_SIGMA = np.eye(D, dtype=np.float64)


def _node(node_id: str, mu: np.ndarray) -> HFN:
    return HFN(mu=mu, sigma=_SIGMA.copy(), id=node_id)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_code_world_model() -> tuple[Forest, list[HFN]]:
    """
    Build the code world model with five sub-trees.

    Returns
    -------
    forest : Forest
        Contains all prior nodes.
    prior_nodes : list[HFN]
        All prior nodes (pass their ids as protected_ids to Observer).
    """
    forest = Forest(D=D, forest_id="code_python")
    prior_nodes: list[HFN] = []

    def add(node: HFN) -> HFN:
        forest.register(node)
        prior_nodes.append(node)
        return node

    # ==================================================================
    # ATOMIC TOKEN NODES (70 nodes): word_<token>
    # ==================================================================
    token_nodes: dict[str, HFN] = {}
    for token in VOCAB:
        nid = f"word_{token}"
        n = _node(nid, _one_hot(token))
        token_nodes[token] = n
        add(n)

    def wmu(token: str) -> np.ndarray:
        """Get atomic token node mu."""
        return token_nodes[token].mu

    # ==================================================================
    # SYNTAX SUB-TREE
    # ==================================================================

    # --- Control keyword leaves ---
    syn_control = add(_node("syn_control", _recombine(
        wmu("if"), wmu("elif"), wmu("else"), wmu("for"), wmu("while"),
        wmu("break"), wmu("continue"), wmu("pass"),
    )))

    # --- Definition keyword leaves ---
    syn_def_kw = add(_node("syn_def_kw", _recombine(
        wmu("def"), wmu("class"), wmu("return"), wmu("lambda"), wmu("yield"),
    )))

    # --- Import/context keyword leaves ---
    syn_import_kw = add(_node("syn_import_kw", _recombine(
        wmu("import"), wmu("from"), wmu("as"), wmu("with"),
        wmu("try"), wmu("except"),
    )))

    # --- Logic keyword leaves ---
    syn_logic_kw = add(_node("syn_logic_kw", _recombine(
        wmu("and"), wmu("or"), wmu("not"), wmu("in"), wmu("is"),
        wmu("assert"), wmu("raise"), wmu("del"), wmu("global"),
    )))

    # --- Keyword parent ---
    syn_keywords = add(_node("syn_keywords", _recombine(
        syn_control.mu, syn_def_kw.mu, syn_import_kw.mu, syn_logic_kw.mu,
    )))
    syn_keywords.add_child(syn_control)
    syn_keywords.add_child(syn_def_kw)
    syn_keywords.add_child(syn_import_kw)
    syn_keywords.add_child(syn_logic_kw)

    # --- Comparison operators ---
    syn_cmp_ops = add(_node("syn_cmp_ops", _recombine(
        wmu("=="), wmu("!="), wmu("<"), wmu(">"), wmu("<="), wmu(">="),
    )))

    # --- Arithmetic operators ---
    syn_arith_ops = add(_node("syn_arith_ops", _recombine(
        wmu("+"), wmu("-"), wmu("*"), wmu("/"), wmu("//"), wmu("%"), wmu("**"),
    )))

    # --- Bitwise operators ---
    syn_bit_ops = add(_node("syn_bit_ops", _recombine(
        wmu("&"), wmu("|"), wmu("^"), wmu("~"),
    )))

    # --- Assignment operator ---
    syn_assign_op = add(_node("syn_assign_op", _recombine(wmu("="),)))

    # --- Operator parent ---
    syn_operators = add(_node("syn_operators", _recombine(
        syn_cmp_ops.mu, syn_arith_ops.mu, syn_bit_ops.mu, syn_assign_op.mu,
    )))
    syn_operators.add_child(syn_cmp_ops)
    syn_operators.add_child(syn_arith_ops)
    syn_operators.add_child(syn_bit_ops)
    syn_operators.add_child(syn_assign_op)

    # --- Punctuation nodes ---
    syn_open_punct = add(_node("syn_open_punct", _recombine(
        wmu("("), wmu("["), wmu("{"),
    )))
    syn_close_punct = add(_node("syn_close_punct", _recombine(
        wmu(")"), wmu("]"), wmu("}"),
    )))
    syn_sep_punct = add(_node("syn_sep_punct", _recombine(
        wmu(":"), wmu(","), wmu("."),
    )))

    syn_punctuation = add(_node("syn_punctuation", _recombine(
        syn_open_punct.mu, syn_close_punct.mu, syn_sep_punct.mu,
    )))
    syn_punctuation.add_child(syn_open_punct)
    syn_punctuation.add_child(syn_close_punct)
    syn_punctuation.add_child(syn_sep_punct)

    # --- Syntax root ---
    syn_root = add(_node("syn_root", _recombine(
        syn_keywords.mu, syn_operators.mu, syn_punctuation.mu,
    )))
    syn_root.add_child(syn_keywords)
    syn_root.add_child(syn_operators)
    syn_root.add_child(syn_punctuation)

    # ==================================================================
    # TYPE SUB-TREE
    # ==================================================================

    type_int = add(_node("type_int", _recombine(wmu("int"), wmu("="), wmu("("), wmu(")"))))
    type_str = add(_node("type_str", _recombine(wmu("str"), wmu("="), wmu("("), wmu(")"))))
    type_float = add(_node("type_float", _recombine(wmu("float"), wmu("="), wmu("("), wmu(")"))))
    type_bool = add(_node("type_bool", _recombine(wmu("bool"), wmu("="), wmu("("), wmu(")"))))

    type_numeric = add(_node("type_numeric", _recombine(type_int.mu, type_float.mu)))
    type_numeric.add_child(type_int)
    type_numeric.add_child(type_float)

    type_primitive = add(_node("type_primitive", _recombine(
        type_numeric.mu, type_str.mu, type_bool.mu,
    )))
    type_primitive.add_child(type_numeric)
    type_primitive.add_child(type_str)
    type_primitive.add_child(type_bool)

    type_list = add(_node("type_list", _recombine(wmu("list"), wmu("["), wmu("]"), wmu("for"))))

    # dict, tuple, None — available in Python but not in VOCAB, so we approximate
    # with closest tokens in vocabulary
    type_container = add(_node("type_container", _recombine(
        type_list.mu, syn_open_punct.mu, syn_close_punct.mu,
    )))
    type_container.add_child(type_list)

    type_root = add(_node("type_root", _recombine(type_primitive.mu, type_container.mu)))
    type_root.add_child(type_primitive)
    type_root.add_child(type_container)

    # ==================================================================
    # BUILTIN SUB-TREE
    # ==================================================================

    bi_print = add(_node("bi_print", _recombine(wmu("print"), wmu("("), wmu(")"))))
    bi_input = add(_node("bi_input", _recombine(wmu("input"), wmu("("), wmu(")"))))
    bi_open = add(_node("bi_open", _recombine(wmu("open"), wmu("("), wmu("with"))))

    bi_io = add(_node("bi_io", _recombine(bi_print.mu, bi_input.mu, bi_open.mu)))
    bi_io.add_child(bi_print)
    bi_io.add_child(bi_input)
    bi_io.add_child(bi_open)

    bi_len = add(_node("bi_len", _recombine(wmu("len"), wmu("("), wmu(")"))))
    bi_range = add(_node("bi_range", _recombine(wmu("range"), wmu("("), wmu("for"))))
    bi_type = add(_node("bi_type", _recombine(wmu("type"), wmu("("), wmu(")"))))
    bi_map = add(_node("bi_map", _recombine(wmu("map"), wmu("("), wmu("lambda"))))
    bi_filter = add(_node("bi_filter", _recombine(wmu("filter"), wmu("("), wmu("lambda"))))
    bi_zip = add(_node("bi_zip", _recombine(wmu("zip"), wmu("("), wmu("for"))))
    bi_enumerate = add(_node("bi_enumerate", _recombine(wmu("enumerate"), wmu("("), wmu("for"))))

    bi_iter = add(_node("bi_iter", _recombine(
        bi_len.mu, bi_range.mu, bi_type.mu,
        bi_map.mu, bi_filter.mu, bi_zip.mu, bi_enumerate.mu,
    )))
    for child in [bi_len, bi_range, bi_type, bi_map, bi_filter, bi_zip, bi_enumerate]:
        bi_iter.add_child(child)

    bi_root = add(_node("bi_root", _recombine(bi_io.mu, bi_iter.mu)))
    bi_root.add_child(bi_io)
    bi_root.add_child(bi_iter)

    # ==================================================================
    # PATTERN SUB-TREE
    # ==================================================================

    pat_for_loop = add(_node("pat_for_loop", _recombine(
        wmu("for"), wmu("in"), wmu("range"), wmu(":"),
    )))
    pat_if_condition = add(_node("pat_if_condition", _recombine(
        wmu("if"), wmu("=="), wmu(":"), wmu("else"),
    )))
    pat_function_def = add(_node("pat_function_def", _recombine(
        wmu("def"), wmu("("), wmu(")"), wmu(":"), wmu("return"),
    )))
    pat_assignment = add(_node("pat_assignment", _recombine(
        wmu("="), wmu("("), wmu(")"), wmu(":"),
    )))

    # Recombined pairs
    pat_for_if = add(_node("pat_for_if", _recombine(pat_for_loop.mu, pat_if_condition.mu)))
    pat_for_if.add_child(pat_for_loop)
    pat_for_if.add_child(pat_if_condition)

    pat_def_assign = add(_node("pat_def_assign", _recombine(pat_function_def.mu, pat_assignment.mu)))
    pat_def_assign.add_child(pat_function_def)
    pat_def_assign.add_child(pat_assignment)

    pat_root = add(_node("pat_root", _recombine(pat_for_if.mu, pat_def_assign.mu)))
    pat_root.add_child(pat_for_if)
    pat_root.add_child(pat_def_assign)

    # ==================================================================
    # SENTENCE PRIORS (~15 nodes from short synthetic code snippets)
    # mu = (1/N) * sum(one_hot(t) for t in snippet_tokens)
    # ==================================================================

    def snippet_mu(*tokens: str) -> np.ndarray:
        """Equal-weight average of atomic one-hot mus for snippet tokens."""
        vecs = [wmu(t) for t in tokens if t in token_nodes]
        if not vecs:
            return np.zeros(D, dtype=np.float64)
        return np.mean(np.stack(vecs), axis=0).astype(np.float64)

    snippet_defs = [
        ("snip_for_range",      ("for", "in", "range", "(", ")", ":")),
        ("snip_if_equal",       ("if", "==", ":", "else", ":")),
        ("snip_def_return",     ("def", "(", ")", ":", "return")),
        ("snip_assign",         ("=", "(", ")")),
        ("snip_print_str",      ("print", "(", "str", "(", ")", ")")),
        ("snip_len_list",       ("len", "(", "list", ")", "==")),
        ("snip_import_as",      ("import", "as")),
        ("snip_try_except",     ("try", ":", "except", ":")),
        ("snip_lambda_map",     ("lambda", ":", "map", "(", "lambda", ")")),
        ("snip_while_break",    ("while", ":", "if", ":", "break")),
        ("snip_class_def",      ("class", "(", ")", ":", "def", "(")),
        ("snip_list_comp",      ("for", "in", "if", "]")),
        ("snip_enumerate_for",  ("for", "in", "enumerate", "(", ")", ":")),
        ("snip_assert_eq",      ("assert", "==", ",")),
        ("snip_with_open",      ("with", "open", "(", ")", "as", ":")),
    ]

    for nid, tokens in snippet_defs:
        n = _node(nid, snippet_mu(*tokens))
        add(n)

    return forest, prior_nodes
