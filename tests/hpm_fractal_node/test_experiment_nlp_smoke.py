"""Smoke tests for experiment_nlp.py integration with D=107 refactor."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from hpm_fractal_node.nlp.nlp_loader import D, generate_sentences
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model


def test_d_is_107():
    """D must be 107 in the refactored loader."""
    assert D == 107


def test_observation_vector_shape():
    """Observations returned by generate_sentences must be shape (107,)."""
    data = generate_sentences(seed=42)
    vec, _, _ = data[0]
    assert vec.shape == (107,)
    assert vec.dtype == np.float64


def test_world_model_d_matches_loader():
    """World model forest D must match loader D."""
    forest, _ = build_nlp_world_model()
    assert forest._D == D


def test_observer_can_observe_d107_vector():
    """Observer accepts D=107 input without error."""
    from hfn import Observer, calibrate_tau
    from hfn.tiered_forest import TieredForest
    import tempfile

    data = generate_sentences(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        forest, prior_ids = build_nlp_world_model(
            forest_cls=TieredForest,
            cold_dir=Path(tmpdir),
            max_hot=100,
        )
        forest.set_protected(prior_ids)
        tau = calibrate_tau(D, sigma_scale=1.0, margin=5.0)
        obs = Observer(forest, tau=tau, protected_ids=prior_ids)
        # Observe 10 samples without error
        for vec, _, _ in data[:10]:
            x = vec.astype(np.float64)
            result = obs.observe(x)
            forest._on_observe()
        assert True  # no exception raised


def test_no_encode_context_window_in_loader():
    """encode_context_window must not exist in the refactored loader."""
    import hpm_fractal_node.nlp.nlp_loader as loader_module
    assert not hasattr(loader_module, "encode_context_window"), (
        "encode_context_window should be removed in the refactor"
    )


def test_compose_context_node_importable():
    """compose_context_node must be importable from nlp_loader."""
    from hpm_fractal_node.nlp.nlp_loader import compose_context_node
    assert callable(compose_context_node)


def test_no_registry_in_experiment():
    """experiment_nlp.py must not use forest._registry directly."""
    exp_path = Path(__file__).parents[2] / "hpm_fractal_node" / "experiments" / "experiment_nlp.py"
    source = exp_path.read_text()
    assert "_registry" not in source, (
        "experiment_nlp.py still uses forest._registry — replace with forest.get() / forest.active_nodes()"
    )
