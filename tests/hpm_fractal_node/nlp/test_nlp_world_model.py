"""Tests for nlp_world_model: four sub-tree world model at D=107."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hfn import Forest
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model
from hpm_fractal_node.nlp.nlp_loader import D, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build():
    return build_nlp_world_model()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_build_nlp_world_model_returns_tuple():
    forest, prior_ids = _build()
    assert isinstance(forest, Forest)
    assert isinstance(prior_ids, set)


def test_all_priors_in_registry():
    forest, prior_ids = _build()
    for pid in prior_ids:
        assert forest.get(pid) is not None, f"Missing node: {pid}"


def test_all_nodes_d107():
    """Every registered node must have mu.shape == (107,)."""
    forest, prior_ids = _build()
    for pid in prior_ids:
        node = forest.get(pid)
        assert node.mu.shape == (D,), f"{pid} has mu shape {node.mu.shape}"


def test_all_nodes_sigma_d107():
    """Every node sigma must be (107, 107)."""
    forest, prior_ids = _build()
    for pid in prior_ids:
        node = forest.get(pid)
        assert node.sigma.shape == (D, D), f"{pid} has sigma shape {node.sigma.shape}"


# ---------------------------------------------------------------------------
# Atomic word nodes (107)
# ---------------------------------------------------------------------------

def test_atomic_word_nodes_present():
    forest, prior_ids = _build()
    for word_id in ["word_dog", "word_cat", "word_bird", "word_the", "word_park"]:
        assert forest.get(word_id) is not None, f"Missing atomic node: {word_id}"


def test_atomic_word_nodes_count():
    """There should be exactly 107 atomic word nodes."""
    forest, prior_ids = _build()
    word_nodes = [pid for pid in prior_ids if pid.startswith("word_")]
    assert len(word_nodes) == VOCAB_SIZE


def test_atomic_word_node_mu_is_one_hot():
    """Atomic word node mu must be a one-hot vector (exactly one 1.0, rest 0.0)."""
    forest, prior_ids = _build()
    node = forest.get("word_dog")
    assert node.mu.sum() == pytest.approx(1.0)
    assert (node.mu == 1.0).sum() == 1
    assert node.mu.dtype == np.float64


def test_atomic_word_node_mu_dtype():
    forest, prior_ids = _build()
    node = forest.get("word_the")
    assert node.mu.dtype == np.float64


# ---------------------------------------------------------------------------
# Objects sub-tree
# ---------------------------------------------------------------------------

def test_objects_leaf_nodes_present():
    forest, prior_ids = _build()
    for nid in ["obj_dog", "obj_cat", "obj_bird", "obj_mum", "obj_dad",
                "obj_apple", "obj_ball", "obj_park"]:
        assert forest.get(nid) is not None, f"Missing object leaf: {nid}"


def test_objects_parent_nodes_present():
    forest, prior_ids = _build()
    for nid in ["obj_animal", "obj_person", "obj_family", "obj_adult",
                "obj_child", "obj_animate", "obj_food", "obj_object",
                "obj_place", "obj_inanimate", "obj_noun"]:
        assert forest.get(nid) is not None, f"Missing object parent: {nid}"


def test_objects_animal_has_children():
    """obj_animal should have dog, cat, bird as children."""
    forest, prior_ids = _build()
    animal = forest.get("obj_animal")
    child_ids = {c.id for c in animal._children}
    assert "obj_dog" in child_ids
    assert "obj_cat" in child_ids
    assert "obj_bird" in child_ids


def test_objects_noun_root_has_animate_inanimate():
    forest, prior_ids = _build()
    noun = forest.get("obj_noun")
    child_ids = {c.id for c in noun._children}
    assert "obj_animate" in child_ids
    assert "obj_inanimate" in child_ids


def test_objects_leaf_mu_nonzero():
    """Object leaf mus must be non-trivial (not all zeros)."""
    forest, prior_ids = _build()
    node = forest.get("obj_dog")
    assert node.mu.sum() > 0


def test_objects_leaf_mu_sums_to_one():
    """Object leaf mu should be a weighted recombination summing to 1.0."""
    forest, prior_ids = _build()
    for nid in ["obj_dog", "obj_cat", "obj_park", "obj_apple"]:
        node = forest.get(nid)
        assert abs(node.mu.sum() - 1.0) < 1e-9, f"{nid} mu sum = {node.mu.sum()}"


# ---------------------------------------------------------------------------
# Grammar sub-tree
# ---------------------------------------------------------------------------

def test_grammar_leaf_nodes_present():
    forest, prior_ids = _build()
    for nid in ["gram_determiner", "gram_preposition", "gram_descriptor"]:
        assert forest.get(nid) is not None, f"Missing grammar leaf: {nid}"


def test_grammar_parent_nodes_present():
    forest, prior_ids = _build()
    for nid in ["gram_word_class", "gram_noun_phrase", "gram_verb_phrase",
                "gram_prep_phrase", "gram_phrase_structure",
                "gram_agent_action", "gram_action_patient",
                "gram_motion_to_place", "gram_sentence_pattern",
                "gram_root"]:
        assert forest.get(nid) is not None, f"Missing grammar parent: {nid}"


def test_grammar_word_class_children():
    forest, prior_ids = _build()
    wc = forest.get("gram_word_class")
    child_ids = {c.id for c in wc._children}
    assert "gram_determiner" in child_ids
    assert "gram_preposition" in child_ids
    assert "gram_descriptor" in child_ids


def test_grammar_root_children():
    forest, prior_ids = _build()
    root = forest.get("gram_root")
    child_ids = {c.id for c in root._children}
    assert "gram_word_class" in child_ids
    assert "gram_phrase_structure" in child_ids
    assert "gram_sentence_pattern" in child_ids


# ---------------------------------------------------------------------------
# Capabilities sub-tree
# ---------------------------------------------------------------------------

def test_capabilities_leaf_nodes_present():
    forest, prior_ids = _build()
    for nid in ["cap_dog_barks", "cap_dog_fetches", "cap_cat_meows",
                "cap_cat_chases", "cap_bird_chirps"]:
        assert forest.get(nid) is not None, f"Missing capability leaf: {nid}"


def test_capabilities_parent_nodes_present():
    forest, prior_ids = _build()
    for nid in ["cap_dog", "cap_cat", "cap_bird", "cap_animal",
                "cap_general_person", "cap_family", "cap_child",
                "cap_person", "cap_root"]:
        assert forest.get(nid) is not None, f"Missing capability parent: {nid}"


def test_capabilities_dog_children():
    forest, prior_ids = _build()
    cap_dog = forest.get("cap_dog")
    child_ids = {c.id for c in cap_dog._children}
    assert "cap_dog_barks" in child_ids
    assert "cap_dog_fetches" in child_ids


def test_capabilities_animal_children():
    forest, prior_ids = _build()
    cap_animal = forest.get("cap_animal")
    child_ids = {c.id for c in cap_animal._children}
    assert "cap_dog" in child_ids
    assert "cap_cat" in child_ids
    assert "cap_bird" in child_ids


def test_capabilities_root_children():
    forest, prior_ids = _build()
    root = forest.get("cap_root")
    child_ids = {c.id for c in root._children}
    assert "cap_animal" in child_ids
    assert "cap_person" in child_ids


# ---------------------------------------------------------------------------
# Sentence priors
# ---------------------------------------------------------------------------

def test_sentence_priors_present():
    forest, prior_ids = _build()
    sent_nodes = [pid for pid in prior_ids if pid.startswith("sent_")]
    assert len(sent_nodes) >= 15, f"Only {len(sent_nodes)} sentence priors, expected >= 15"


def test_sentence_prior_mu_sums_to_one():
    """Sentence prior mu = (1/N)*sum(one_hot(w)), sums to 1.0."""
    forest, prior_ids = _build()
    sent_nodes = [pid for pid in prior_ids if pid.startswith("sent_")]
    for nid in sent_nodes[:5]:
        node = forest.get(nid)
        assert abs(node.mu.sum() - 1.0) < 1e-9, f"{nid} mu sum = {node.mu.sum()}"


def test_sentence_prior_mu_dtype():
    forest, prior_ids = _build()
    sent_nodes = [pid for pid in prior_ids if pid.startswith("sent_")]
    node = forest.get(sent_nodes[0])
    assert node.mu.dtype == np.float64


# ---------------------------------------------------------------------------
# Total node count
# ---------------------------------------------------------------------------

def test_total_node_count():
    """Forest must contain at least 107 (atomic) + 25 (objects) + 13 (grammar)
    + 19 (capabilities) + 15 (sentences) = 179 nodes."""
    forest, prior_ids = _build()
    assert len(prior_ids) >= 179, f"Only {len(prior_ids)} nodes registered"
