# SP66 Core Modules Integration Plan

## 1. Objective
Integrate the core HFN modules developed for SP66 (`density.py` and `affective.py`) into the `hfn/` directory, making them available as standard library components. Include their respective unit tests to ensure stability and correctness.

## 2. Implementation Steps
- [ ] Create `hfn/density.py` with the `PatternDensityTracker` implementation.
- [ ] Create `hfn/affective.py` with the `AffectiveState` and `AffectiveEvaluator` implementations.
- [ ] Update `hfn/observer.py` to include `self.density_tracker = None` in `__init__`.
- [ ] Create `tests/test_density.py`.
- [ ] Create `tests/test_affective.py`.

## 3. Verification & Testing
- Run `pytest tests/test_density.py` to verify density tracking logic.
- Run `pytest tests/test_affective.py` to verify affective state logic.
- Ensure no regressions by running the full test suite `pytest tests/` (excluding slow benchmarks).
