"""
NLP world model — 38 prior HFN nodes for child language experiment.

Three layers:
  Word priors   (25): one per target word, mu = avg of 5 canonical sentence encodings
  Category priors (8): avg of direct word-child mus
  Relational priors (5): avg of category-child mus

All nodes are protected — the Observer cannot absorb or remove them.
"""
from __future__ import annotations

import numpy as np

from hfn import HFN, Forest
from hpm_fractal_node.nlp.nlp_loader import D, encode_context_window


# ---------------------------------------------------------------------------
# Canonical sentences for word prior mu construction
# (left2, left1, right1, right2) — the masked word is NOT in the tuple
# ---------------------------------------------------------------------------

_CANONICAL_SENTENCES: dict[str, list[tuple[str, str, str, str]]] = {
    "dog":       [("<start>", "the",    "barked",  "<end>"),
                  ("<start>", "the",    "ran",     "<end>"),
                  ("a",       "dog",    "chased",  "the"),
                  ("the",     "big",    "sat",     "<end>"),
                  ("<start>", "my",     "played",  "<end>")],
    "cat":       [("<start>", "the",    "meowed",  "<end>"),
                  ("<start>", "the",    "ran",     "<end>"),
                  ("a",       "cat",    "chased",  "the"),
                  ("the",     "small",  "sat",     "<end>"),
                  ("<start>", "my",     "played",  "<end>")],
    "bird":      [("<start>", "the",    "chirped", "<end>"),
                  ("<start>", "the",    "ran",     "<end>"),
                  ("a",       "bird",   "chased",  "the"),
                  ("the",     "little", "sat",     "<end>"),
                  ("<start>", "my",     "played",  "<end>")],
    "mum":       [("<start>", "my",     "ate",     "the"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "gave",    "me"),
                  ("<start>", "my",     "took",    "the")],
    "dad":       [("<start>", "my",     "ate",     "the"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "gave",    "me"),
                  ("<start>", "my",     "took",    "the")],
    "grandma":   [("<start>", "my",     "ate",     "the"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "her",    "gave",    "me"),
                  ("<start>", "her",    "took",    "the")],
    "grandpa":   [("<start>", "my",     "ate",     "the"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "his",    "gave",    "me"),
                  ("<start>", "his",    "took",    "the")],
    "brother":   [("<start>", "my",     "ran",     "to"),
                  ("<start>", "my",     "played",  "with"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "took",    "the"),
                  ("<start>", "my",     "walked",  "to")],
    "sister":    [("<start>", "my",     "ran",     "to"),
                  ("<start>", "my",     "played",  "with"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "took",    "the"),
                  ("<start>", "my",     "walked",  "to")],
    "teacher":   [("<start>", "my",     "helped",  "me"),
                  ("<start>", "the",    "walked",  "to"),
                  ("<start>", "the",    "gave",    "me"),
                  ("<start>", "the",    "ran",     "to"),
                  ("<start>", "my",     "played",  "with")],
    "doctor":    [("<start>", "the",    "helped",  "me"),
                  ("<start>", "the",    "walked",  "to"),
                  ("<start>", "the",    "gave",    "me"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "the",    "took",    "the")],
    "friend":    [("<start>", "my",     "played",  "with"),
                  ("<start>", "my",     "ran",     "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "gave",    "me")],
    "classmate": [("<start>", "my",     "played",  "with"),
                  ("<start>", "my",     "ran",     "to"),
                  ("<start>", "my",     "helped",  "me"),
                  ("<start>", "my",     "walked",  "to"),
                  ("<start>", "my",     "took",    "the")],
    "boy":       [("<start>", "the",    "ran",     "to"),
                  ("<start>", "the",    "played",  "with"),
                  ("<start>", "the",    "walked",  "to"),
                  ("<start>", "the",    "helped",  "the"),
                  ("<start>", "the",    "threw",   "the")],
    "girl":      [("<start>", "the",    "ran",     "to"),
                  ("<start>", "the",    "played",  "with"),
                  ("<start>", "the",    "walked",  "to"),
                  ("<start>", "the",    "helped",  "the"),
                  ("<start>", "the",    "threw",   "the")],
    "baby":      [("<start>", "the",    "played",  "with"),
                  ("<start>", "the",    "walked",  "to"),
                  ("<start>", "my",     "played",  "with"),
                  ("<start>", "the",    "helped",  "the"),
                  ("<start>", "the",    "took",    "the")],
    "apple":     [("<start>", "the",    "is",      "good"),
                  ("<start>", "an",     "is",      "on"),
                  ("the",     "big",    "is",      "good"),
                  ("the",     "red",    "is",      "good"),
                  ("<start>", "the",    "was",     "good")],
    "bread":     [("<start>", "the",    "is",      "good"),
                  ("<start>", "the",    "is",      "on"),
                  ("the",     "big",    "is",      "good"),
                  ("the",     "old",    "is",      "good"),
                  ("<start>", "the",    "was",     "good")],
    "milk":      [("<start>", "the",    "is",      "good"),
                  ("<start>", "the",    "is",      "on"),
                  ("the",     "big",    "is",      "good"),
                  ("the",     "old",    "is",      "good"),
                  ("<start>", "the",    "was",     "good")],
    "ball":      [("<start>", "a",      "is",      "on"),
                  ("<start>", "the",    "is",      "here"),
                  ("the",     "big",    "is",      "here"),
                  ("the",     "red",    "is",      "here"),
                  ("<start>", "the",    "was",     "<end>")],
    "book":      [("<start>", "a",      "is",      "on"),
                  ("<start>", "the",    "is",      "here"),
                  ("the",     "big",    "is",      "here"),
                  ("the",     "old",    "is",      "here"),
                  ("<start>", "the",    "was",     "<end>")],
    "toy":       [("<start>", "a",      "is",      "on"),
                  ("<start>", "the",    "is",      "here"),
                  ("the",     "little", "is",      "here"),
                  ("the",     "old",    "is",      "here"),
                  ("<start>", "the",    "was",     "<end>")],
    "park":      [("we",      "went",   "to",      "the"),
                  ("<start>", "ran",    "to",      "the"),
                  ("<start>", "walked", "to",      "the"),
                  ("at",      "the",    "<end>",   "<end>"),
                  ("the",     "big",    "is",      "near")],
    "home":      [("we",      "went",   "to",      "the"),
                  ("<start>", "ran",    "to",      "the"),
                  ("<start>", "walked", "to",      "the"),
                  ("at",      "the",    "<end>",   "<end>"),
                  ("the",     "old",    "is",      "near")],
    "school":    [("we",      "went",   "to",      "the"),
                  ("<start>", "ran",    "to",      "the"),
                  ("<start>", "walked", "to",      "the"),
                  ("at",      "the",    "<end>",   "<end>"),
                  ("the",     "big",    "is",      "near")],
}


def _word_mu(word: str) -> np.ndarray:
    """Compute mu for a word prior as average of its canonical sentence encodings."""
    vecs = [
        encode_context_window(left1, right1, left2=left2, right2=right2)
        for (left2, left1, right1, right2) in _CANONICAL_SENTENCES[word]
    ]
    return np.mean(vecs, axis=0).astype(np.float32)


def _avg_mus(mus: list[np.ndarray]) -> np.ndarray:
    return np.mean(mus, axis=0).astype(np.float32)


def build_nlp_world_model(forest_cls=None, **tiered_kwargs) -> tuple[Forest, set[str]]:
    """
    Build the NLP world model with 38 prior HFN nodes.

    Returns
    -------
    forest : Forest
        Contains all prior nodes.
    prior_ids : set[str]
        All node IDs to pass as protected_ids to Observer.
    """
    from hfn.forest import Forest as _Forest
    if forest_cls is None:
        forest_cls = _Forest
    kwargs = tiered_kwargs if forest_cls is not _Forest else {}
    forest = forest_cls(D=D, forest_id="nlp_child_language", **kwargs)
    prior_ids: set[str] = set()

    def add(node: HFN) -> None:
        forest.register(node)
        prior_ids.add(node.id)

    sigma = np.eye(D) * 1.0

    # ------------------------------------------------------------------
    # Layer 1 — Word priors (25 nodes)
    # ------------------------------------------------------------------
    word_mus: dict[str, np.ndarray] = {}
    for word in _CANONICAL_SENTENCES:
        mu = _word_mu(word)
        word_mus[word] = mu
        add(HFN(mu=mu, sigma=sigma, id=f"prior_{word}"))

    # ------------------------------------------------------------------
    # Layer 2 — Category priors (8 nodes)
    # ------------------------------------------------------------------
    category_mus: dict[str, np.ndarray] = {}

    def cat_mu(words: list[str]) -> np.ndarray:
        return _avg_mus([word_mus[w] for w in words])

    cat_defs = {
        "prior_animal":       ["dog", "cat", "bird"],
        "prior_adult":        ["teacher", "doctor", "friend", "classmate"],
        "prior_child_person": ["boy", "girl", "baby"],
        "prior_family":       ["mum", "dad", "grandma", "grandpa", "brother", "sister"],
        "prior_food":         ["apple", "bread", "milk"],
        "prior_object":       ["ball", "book", "toy"],
        "prior_place":        ["park", "home", "school"],
        "prior_person":       ["mum", "dad", "boy", "girl", "teacher", "doctor",
                               "friend", "classmate", "baby"],
    }

    for cid, words in cat_defs.items():
        mu = cat_mu(words)
        category_mus[cid] = mu
        add(HFN(mu=mu, sigma=sigma, id=cid))

    # ------------------------------------------------------------------
    # Layer 3 — Relational priors (5 nodes)
    # ------------------------------------------------------------------
    def rel_mu(cat_ids: list[str]) -> np.ndarray:
        return _avg_mus([category_mus[c] for c in cat_ids])

    relational_defs = {
        "prior_agent_action":   ["prior_animal", "prior_person"],
        "prior_action_target":  ["prior_food", "prior_object"],
        "prior_thing_place":    ["prior_place"],
        "prior_descriptor_thing": ["prior_animal", "prior_food", "prior_object"],
        "prior_social_relation":  ["prior_family"],
    }

    for rid, cat_ids in relational_defs.items():
        mu = rel_mu(cat_ids)
        add(HFN(mu=mu, sigma=sigma, id=rid))

    return forest, prior_ids
