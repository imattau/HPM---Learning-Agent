"""
Integration smoke test: LaplacePattern agents run end-to-end through the ARC pipeline.

Uses synthetic tasks (no network dependency) to validate the full stack:
LaplacePattern → Agent → InMemoryStore → PatternField → MetaPatternRule → ensemble_score
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from benchmarks.multi_agent_arc import (
    make_arc_orchestrator, ensemble_score, encode_pair, FEATURE_DIM, N_DISTRACTORS
)
from hpm.patterns.gaussian import GaussianPattern
from hpm.patterns.laplace import LaplacePattern


def _make_synthetic_task(rng, n_train=3):
    """Generate a synthetic ARC-shaped task with random grids."""
    def rand_grid(size=5):
        return rng.integers(0, 9, size=(size, size)).tolist()

    return {
        "train": [{"input": rand_grid(), "output": rand_grid()} for _ in range(n_train)],
        "test": [{"input": rand_grid(), "output": rand_grid()}],
    }


def _make_synthetic_all_tasks(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [_make_synthetic_task(rng) for _ in range(n)]


@pytest.fixture(scope="module")
def all_tasks():
    return _make_synthetic_all_tasks(n=20)


def test_laplace_arc_no_crash(all_tasks):
    """Laplace agents run evaluate_task without raising exceptions."""
    from benchmarks.multi_agent_arc import evaluate_task
    correct, rank = evaluate_task(all_tasks[0], all_tasks, 0,
                                  pattern_types=["laplace", "laplace"])
    assert isinstance(correct, bool)
    assert 1 <= rank <= N_DISTRACTORS + 1


def test_laplace_mean_rank_in_bounds(all_tasks):
    """Run 10 tasks and confirm mean_rank is in [1, 5].

    Note: the upper bound (5.0) is the maximum possible rank with N_DISTRACTORS=4,
    so this is a pipeline smoke test — it validates no crash and correct rank range,
    not performance.
    """
    from benchmarks.multi_agent_arc import evaluate_task
    ranks = []
    for i in range(10):
        _, rank = evaluate_task(all_tasks[i], all_tasks, i,
                                pattern_types=["laplace", "laplace"])
        ranks.append(rank)
    mean_rank = np.mean(ranks)
    assert 1.0 <= mean_rank <= 5.0


def test_laplace_and_gaussian_produce_different_nll():
    """LaplacePattern and GaussianPattern produce different NLL for the same input."""
    rng = np.random.default_rng(42)
    mu = rng.standard_normal(FEATURE_DIM)
    x = rng.standard_normal(FEATURE_DIM)

    gauss = GaussianPattern(mu, np.eye(FEATURE_DIM))
    laplace = LaplacePattern(mu, np.ones(FEATURE_DIM))

    nll_g = gauss.log_prob(x)
    nll_l = laplace.log_prob(x)
    assert nll_g != nll_l, "Gaussian and Laplace should produce different NLL values"


def test_gaussian_orchestrator_still_works(all_tasks):
    """Confirm the default Gaussian orchestrator is unaffected by this change."""
    from benchmarks.multi_agent_arc import evaluate_task
    correct, rank = evaluate_task(all_tasks[0], all_tasks, 0)
    assert 1 <= rank <= N_DISTRACTORS + 1
