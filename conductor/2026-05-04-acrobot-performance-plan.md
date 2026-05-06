# Acrobot Performance Tuning Plan

## Objective
Optimize the HPM v5 core to speed up execution of high-dimensional, long-horizon benchmarks like Acrobot. The current implementation evaluates pattern distance repeatedly, which is a bottleneck when episode lengths and pattern counts grow.

## Identified Bottlenecks
1. **Repeated Canonicalization in Pattern Matching:**
   `PatternStore.match` and `PatternStore.top_k` call `Pattern.distance` for every pattern in the store. Inside `Pattern.distance`, both the `candidate` and the `pattern.template` are canonicalized (which involves calculating all rotations of the sequence and finding the minimum) on *every single distance check*. This results in $O(|patterns| \times N^2)$ operations per engine step, where $N$ is the sequence length.

2. **Redundant Calculations in Post-processing:**
   The `AcrobotForecastPostprocessor` recalculates `math.cos(theta1)` and `math.cos(theta2)` during every step to build its `q_state_key`, even though `AcrobotStateAdapter` already calculates these exact values and stores them in the state vector. Additionally, multiple `dict.get` calls with defaults are made in `postprocess` for values that are guaranteed to exist.

## Proposed Solutions

### 1. Pre-calculate Candidate Canonicalization
- **Change:** Modify `PatternStore.match` and `PatternStore.top_k` to calculate `canon_candidate = canonicalize_sequence(candidate)` *once* before iterating over patterns.
- **Change:** Update the signature of `Pattern.distance` to accept an optional `canon_candidate` keyword argument to bypass redundant calculation.

### 2. Cache Pattern Canonicalization
- **Change:** Since a pattern's `template` changes infrequently (only on `update()`), we should cache its canonical form.
- **Change:** Modify `Pattern.distance` to compute `canonical_template` once, cache it on the pattern object (e.g., as `self._cached_canon_template`), and invalidate the cache if `self.template` or the `canonicalization_mode` changes.

### 3. Optimize Pipeline Adapters
- **Change:** Update `AcrobotForecastPostprocessor._policy_state_key` to extract `cos1`, `cos2`, `v1`, and `v2` directly from the `context["flattened_state"]` instead of re-calculating trig functions from the raw observation.
- **Change:** Streamline `_magnitude_bucket` to use native float comparisons directly.

## Scope of Changes
- `hpm_ai_v5/core/pattern.py`: Update `Pattern.distance` signature and implementation.
- `hpm_ai_v5/core/store.py`: Update `PatternStore.match` and `PatternStore.top_k` to pre-calculate `canon_candidate` and pass it to `distance()`.

## Validation
- Re-run `test_acrobot.py` and existing tests to ensure no regressions in pattern matching logic.
- Run a short `--smoke` test of the benchmark to verify the speedup empirically.
