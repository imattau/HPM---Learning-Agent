import pytest

from hpm_fractal_node.experiments.experiment_multifractal_learning import (
    ExperimentConfig,
    run_experiment,
)


@pytest.fixture(scope="module")
def summary():
    return run_experiment(ExperimentConfig(dense_inputs=8, sparse_inputs=6))


def test_dense_region_prefers_compression(summary):
    assert summary.dense_phase.compressions > summary.dense_phase.creations


def test_sparse_region_prefers_creation(summary):
    assert summary.sparse_phase.creations > summary.sparse_phase.compressions


def test_both_mechanisms_occur(summary):
    assert summary.created_leaf_count > 0
    assert summary.compressed_node_count > 0


def test_density_signal_differs_by_region(summary):
    assert summary.dense_phase.mean_density_ratio > summary.sparse_phase.mean_density_ratio


def test_not_stagnant(summary):
    assert not summary.stagnation
