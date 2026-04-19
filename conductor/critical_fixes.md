# Plan to fix critical issues

## 1. `hfn.fractal` module import error

- Read `hfn/fractal.py` to identify any potential syntax errors.
- If no syntax errors are found, modify `hfn/evaluator.py` and `hfn/observer.py` to use local imports for the functions from `hfn.fractal`.

## 2. `observer_state_to_vec` references non-existent attributes

- Read `hpm_ai_v2/hpm/meta_observer.py`.
- Replace `observer._weights` with `observer.state_store.weights_dict()`.
- Replace `observer._scores` with a loop over `observer.state_store.active_nodes()` to get the scores.

## 3. Stale μ-index in `TieredForest`

- Implement `update_mu_index` in `hfn/tiered_forest.py` that updates the `_mu_index` with the new `mu` of a node.
- Read `hfn/observer.py`.
- Find all places where `node.mu` is updated.
- Add a call to `self.forest.update_mu_index(node)` after each update.
