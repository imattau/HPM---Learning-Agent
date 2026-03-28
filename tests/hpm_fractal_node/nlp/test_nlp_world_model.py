"""Tests for nlp_world_model: 38 prior nodes in Forest."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from hfn import Forest
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model


def test_build_nlp_world_model_returns_tuple():
    forest, prior_ids = build_nlp_world_model()
    assert isinstance(forest, Forest)
    assert isinstance(prior_ids, set)


def test_world_model_has_38_priors():
    forest, prior_ids = build_nlp_world_model()
    assert len(prior_ids) == 38


def test_word_prior_ids_present():
    forest, prior_ids = build_nlp_world_model()
    assert "prior_dog" in prior_ids
    assert "prior_mum" in prior_ids
    assert "prior_apple" in prior_ids


def test_category_prior_ids_present():
    forest, prior_ids = build_nlp_world_model()
    assert "prior_animal" in prior_ids
    assert "prior_person" in prior_ids
    assert "prior_family" in prior_ids


def test_relational_prior_ids_present():
    forest, prior_ids = build_nlp_world_model()
    assert "prior_agent_action" in prior_ids
    assert "prior_action_target" in prior_ids
    assert "prior_thing_place" in prior_ids


def test_prior_mu_shape():
    forest, prior_ids = build_nlp_world_model()
    node = forest._registry["prior_dog"]
    assert node.mu.shape == (428,)


def test_all_priors_in_registry():
    forest, prior_ids = build_nlp_world_model()
    for pid in prior_ids:
        assert pid in forest._registry


def test_prior_ids_count_equals_registry():
    forest, prior_ids = build_nlp_world_model()
    assert len(prior_ids) == len(forest._registry)
