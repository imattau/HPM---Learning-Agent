# Probabilistic Model Parameter Learning Plan

## Objective
Extend the `ProbabilisticModel` interface to support online parameter learning (e.g., updating means and variances based on observations). Integrate this into the core `Observer` so that nodes dynamically adapt their geometry (not just their weights) when they successfully explain incoming data.

## Motivation
Currently, the `Observer` only updates node weights, scores, and hit/miss counts. Node parameters (`mu` and `sigma`) remain static unless updated by specialized plasticity rules. For advanced probabilistic models like Gaussian Mixture Models or Hierarchical Latent Models to be fully functional, they must be capable of adapting their internal parameters (e.g., component means and mixture weights) online based on evidence.

## Implementation Steps

### 1. Extend `ProbabilisticModel` API
**File:** `hfn/probabilistic_models.py`
- Add an abstract method `update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1) -> None` to `ProbabilisticModel`.
- Implement `update` in `FlatGaussianModel` using an exponential moving average (EMA) on `mu` (and optionally `sigma`).
- Refactor the existing `update` method in `GaussianMixtureModel` to match the new signature. It will perform responsibility-weighted EMA updates on component means and mixing weights.

### 2. Delegate `update` in `HFN`
**File:** `hfn/hfn.py`
- Add an `update(self, x: np.ndarray, weight: float = 1.0, learning_rate: float = 0.1)` method to `HFN`.
- This method will delegate directly to `self.prob_model.update(x, weight, learning_rate)`.

### 3. Integrate Parameter Learning in `Observer`
**File:** `hfn/observer.py`
- In `Observer._update_weights`, identify the loop over active nodes.
- For nodes that are in `effective_explaining_ids` (nodes that helped explain the observation), invoke `node.update(x, weight=float(s.mu[0]), learning_rate=self.alpha_gain)` immediately after updating the node's explaining weight.
- This ensures that parameter learning is driven by positive evidence and is proportional to the node's current reliability (weight) and the observer's learning rate.

### 4. Update the GMM Experiment
**File:** `hpm_ai_v2/experiments/experiment_probabilistic_models.py`
- Remove the manual `mu` updates (`flat_node.mu = ...`) and `prob_model.update(...)` calls from the experiment loop.
- The `Observer.observe(x)` call will now handle parameter learning natively for both the Flat Gaussian and the GMM scenarios, provided the nodes explain the observation (which they will, given the setup).

## Verification
- Run the full test suite (`pytest`) to verify no regressions in weight updating or observer logic.
- Run `PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_probabilistic_models.py` to ensure that both the `FlatGaussianModel` and `GaussianMixtureModel` adapt correctly to the data streams via the newly integrated Observer parameter updates.