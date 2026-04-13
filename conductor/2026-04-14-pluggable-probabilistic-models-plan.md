# Pluggable Probabilistic Models Refactor

## Objective
Refactor the HFN core to make probabilistic models **pluggable** without breaking existing code. This will allow future subclasses or custom models (e.g., `HierarchicalLatentModel`, `GaussianMixtureModel`) to be attached to any `HFN` node without changing existing code that uses the node’s geometry directly.

## Implementation Steps

### 1. Create `hfn/probabilistic_models.py`
- Define an abstract `ProbabilisticModel` class (or Protocol) with methods `log_prob`, `overlap`, `description_length`.
- Move the current Gaussian implementation into a `FlatGaussianModel` that implements this interface.
- Ensure `FlatGaussianModel` properly handles both full covariance and diagonal matrices based on `use_diag`.

### 2. Modify `HFN` class in `hfn/hfn.py`
- Import `ProbabilisticModel` and `FlatGaussianModel` from `hfn.probabilistic_models`.
- Add an optional `prob_model: Optional[ProbabilisticModel] = None` parameter to `HFN.__init__`.
- If `prob_model` is not provided, initialize a default `FlatGaussianModel` using the `mu`, `sigma`, and `use_diag` parameters.
- Delegate the `log_prob`, `overlap`, and `description_length` methods of `HFN` to the underlying `self.prob_model`.
- Keep existing fields (`mu`, `sigma`, `use_diag`, `_sigma_diag`, `_log_det_cached`) intact for backward compatibility, although the caching logic for the default Gaussian will be handled by `FlatGaussianModel`.

### 3. Ensure Backward Compatibility
- Existing code creating `HFN` nodes without specifying `prob_model` will default to `FlatGaussianModel`.
- Direct access to `node.mu` and `node.sigma` will still work.
- Existing tests and experiments should pass without modifications.

## Verification
- Run the full test suite (`pytest`) to ensure no existing functionality is broken.
- Specifically verify tests related to `HFN` creation, `log_prob`, `overlap`, and `description_length`.
