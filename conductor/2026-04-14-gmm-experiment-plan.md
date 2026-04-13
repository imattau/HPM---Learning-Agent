# GMM vs Flat Gaussian Experiment Plan

## Objective
Validate the new pluggable `ProbabilisticModel` architecture in the HFN core by implementing a `GaussianMixtureModel` (GMM) and comparing its performance against the default `FlatGaussianModel` on a multi-modal dataset.

## Motivation
The recent refactor made probabilistic models pluggable. To prove the utility of this architectural upgrade, we will show that an `HFN` node equipped with a GMM can effectively learn a multi-modal concept (e.g., "parity" - even vs. odd numbers) that a single flat Gaussian cannot represent accurately. This demonstrates that we can increase the expressivity of nodes without altering the underlying DAG structure or the observer's core logic.

## Implementation Steps

### 1. Implement `GaussianMixtureModel`
**File:** `hfn/probabilistic_models.py`
- Add a new class `GaussianMixtureModel` inheriting from `ProbabilisticModel`.
- Implement `__init__` to accept a list of components (e.g., `FlatGaussianModel` instances) and their mixing weights.
- Implement `log_prob` using `np.logaddexp` across the weighted component probabilities.
- Implement `overlap` (e.g., max overlap of components or a weighted sum).
- Implement `description_length` as the sum of component description lengths plus the weights.
- Add an `update` method that performs a simple Hebbian competitive learning step (winner-take-all or soft responsibility updates) to adjust the means and weights based on observations.

### 2. Create the Experiment Script
**File:** `hpm_ai_v2/experiments/experiment_probabilistic_models.py`
- Setup a 1-D domain with two distinct modes (e.g., even numbers `[2, 4, 6, 8, 10]` and odd numbers `[1, 3, 5, 7, 9]`).
- Instantiate two HFN nodes:
  - Node A: Uses the default `FlatGaussianModel`.
  - Node B: Uses the new `GaussianMixtureModel` (K=2).
- Train both nodes on the dataset.
  - Manually update the mean of the `FlatGaussianModel` via a running average.
  - Use the `update` method on the `GaussianMixtureModel` to adapt its components.
- Test both nodes on held-out data from both modes (e.g., `[12, 14]` and `[11, 13]`).
- Compare the residual surprise (`-log_prob`) of both models.

## Verification & Success Criteria
- **Execution:** Run `PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_probabilistic_models.py` without errors.
- **Performance:** The GMM should exhibit significantly lower average residual surprise on the test set compared to the Flat Gaussian.
- **Differentiation:** The two components of the GMM should specialize, with one centering near the even numbers and the other near the odd numbers.