# HFN Core Library

The **Hierarchical Fractal Node (HFN)** library is a domain-agnostic implementation of the Hierarchical Pattern Modelling (HPM) framework.

## Architecture

HFN encodes patterns as Gaussian distributions within a directed acyclic graph (DAG). The library provides the tools to create, update, and evaluate these patterns through observation and discovery.

### Key Components

- **`HFN`**: The fundamental unit. Represents a single pattern with a Gaussian identity (`mu`, `sigma`) and a DAG body (edges to parent/child nodes).
- **`Forest`**: A collection of active HFN nodes representing the agent's world model. Supports registration, retrieval, and persistence.
- **`Observer`**: The primary learning engine. It processes observations, updates pattern weights, and triggers the discovery of new nodes via residual surprise and co-occurrence compression.
- **`Evaluator`**: Determines pattern utility. The default implementation uses epistemic metrics (accuracy, complexity, coherence).
- **`PatternDensityTracker`**: Tracks the "stickiness" of patterns based on structural connectivity, reinforcement history, and field amplification (recency-weighted usage).
- **`AffectiveEvaluator`**: An advanced evaluator that manages emotional state (arousal, valence). It implements HPM §9.3 (anxiety-driven persistence) and §9.4 (curiosity-driven exploration).

## API Overview

### Initialization

```python
import numpy as np
from hfn import HFN, Forest, Observer, calibrate_tau

D = 30  # Manifold dimensionality
forest = Forest(D=D)

# Surprising threshold calibration
tau = calibrate_tau(D, sigma_scale=1.0, margin=1.0)
obs = Observer(forest, tau=tau)
```

### Learning from Observations

```python
# Observe a new data point
result = obs.observe(np.random.rand(D))

if result.is_surprising:
    print(f"Novelty detected! Surprise: {result.residual_surprise:.2f}")
```

### Pattern Density Tracking (SP66)

```python
from hfn import attach_density_tracker

# Attach tracker to an observer
tracker = attach_density_tracker(obs)

# Update density for a node
tracker.update_field_amplification("node_id", time.time())

# Check if a node should be pruned based on density vs epistemic loss
if tracker.should_prune("node_id"):
    forest.deregister("node_id")
```

### Affective Modulation (SP66)

```python
from hfn import AffectiveEvaluator, AffectiveState

# Use affective evaluator in observer
aff_eval = AffectiveEvaluator()
obs_aff = Observer(forest, evaluator=aff_eval)

# Update state from task outcome
aff_eval.update_from_outcome("task_id", success=True, surprise=0.1)

# Check affective bonus
bonus = aff_eval.get_affective_bonus("pattern_id")
```

## Advanced Features

### Pluggable Probabilistic Models

HFN nodes support pluggable probabilistic models. By default, nodes use a single-diagonal-Gaussian model (`FlatGaussianModel`), but they can be initialized with any class implementing the `ProbabilisticModel` interface.

```python
from hfn.hfn import HFN
from hfn.probabilistic_models import ProbabilisticModel

class CustomModel(ProbabilisticModel):
    def log_prob(self, x): ...
    def overlap(self, other): ...
    def description_length(self): ...

# Initialize HFN with custom model
node = HFN(mu=mu, sigma=sigma, prob_model=CustomModel(...))
```

This allows the HFN substrate to support non-Gaussian identities (e.g., GMMs, hierarchical latents) while maintaining structural uniformity.

- **Geometric Retrieval**: Efficiently find candidate patterns using the `GeometricRetriever`.
- **Fractal Metrics**: Measure the complexity and self-similarity of the forest using tools like `box_counting_dimension` and `multifractal_spectrum`.
- **Query/Converter Pipeline**: Map raw data to vectors and handle information gaps via the `Query` and `Converter` interfaces.
