"""
HFN — HPM Fractal Node library.

A domain-agnostic implementation of the Hierarchical Pattern Modelling (HPM)
framework. Patterns are encoded as Gaussian distributions in a directed acyclic
graph; an Observer learns which patterns explain observations and discovers new
ones through residual surprise and co-occurrence compression.

Public API
----------
HFN          — a single node: Gaussian identity + DAG body
Forest       — a collection of active HFN nodes (the world model)
Observer     — holds all dynamic learning state; call obs.observe(x) to learn
AsyncHFNController — async orchestration around the single-writer HFN core
calibrate_tau — compute a sensible tau threshold for a given D and sigma scale

Quick start
-----------
    import numpy as np
    from hfn import HFN, Forest, Observer, AsyncHFNController, calibrate_tau

    D = 9  # e.g. flattened 3x3 grid
    forest = Forest(D=D, forest_id="my_domain")

    # Add a prior
    prior = HFN(mu=np.full(D, 0.5), sigma=np.eye(D) * 2.0, id="prior_uniform")
    forest.register(prior)

    tau = calibrate_tau(D, sigma_scale=2.0, margin=1.0)
    obs = Observer(forest, tau=tau, protected_ids={"prior_uniform"})

    obs.observe(np.random.rand(D))

    # Optional: wrap the core with an async controller for replay/prefetch/state export
    controller = AsyncHFNController(forest, obs)
"""

from hfn.hfn import HFN, Edge
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator, ActivationTracker
from hfn.recombination import Recombination
from hfn.reasoning import CognitiveSolver
from hfn.query import Query
from hfn.converter import Converter
from hfn.fractal import (
    box_counting_dimension, population_dimension, dimension_profile,
    self_similarity_score, hausdorff_distance,
    correlation_dimension, information_dimension, intrinsic_dimensionality,
    persistence_scores, RecurrenceTracker, lacunarity, multifractal_spectrum,
)
from hfn.hfn_controller import AsyncHFNController

import numpy as np


def calibrate_tau(D: int, sigma_scale: float = 1.0, margin: float = 1.0) -> float:
    """
    Compute a sensible tau (surprise threshold) for a D-dimensional Gaussian
    with diagonal covariance sigma = I * sigma_scale.

    baseline = D/2 * log(2π * sigma_scale)
      — the log-prob of the mean itself under this distribution.
      Any observation within one sigma of the mean scores <= baseline.

    tau = baseline + margin
      — margin controls how much deviation is tolerated before an observation
        is considered novel. Each unit of margin ≈ 2 cells differing by 1 sigma.

    Typical margins:
      0.5 — tight: only very close matches accepted
      1.0 — moderate: standard starting point
      2.5 — loose: good for broad structural priors at large D
      5.0 — very loose: used for 10x10 grids where patterns vary widely
    """
    baseline = D / 2.0 * np.log(2.0 * np.pi * sigma_scale)
    return baseline + margin


__all__ = [
    "HFN", "Edge", "Forest", "Observer", "AsyncHFNController", "Evaluator", "ActivationTracker",
    "Recombination", "CognitiveSolver", "calibrate_tau",
    "Query", "Converter",
    "box_counting_dimension", "population_dimension", "dimension_profile",
    "self_similarity_score", "hausdorff_distance",
    "correlation_dimension", "information_dimension", "intrinsic_dimensionality",
    "persistence_scores", "RecurrenceTracker", "lacunarity", "multifractal_spectrum",
]
