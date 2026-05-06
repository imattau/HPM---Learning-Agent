# Regime Shift Adaptation (RSA) Benchmark Plan

## Objective
Evaluate the HPM core's ability to detect sudden changes in the environment dynamics (regime shifts) and adapt its policy online without catastrophic forgetting.

## 1. Agent-Layer Components (`hpm_ai_v5/planning/rsa.py`)

### `RegimeShiftAdapterPipeline`
*   **Pipeline Setup**: Uses the standard continuous control adapters (`RunningNormaliserAdapter`, `DiscretiserAdapter`) combined with the `ChangepointAdapter`.
*   **Changepoint Signal**: The `ChangepointAdapter` will process a stream of scalar values to detect shifts. We will feed it the *prediction error* (e.g., $1.0 - \text{confidence}$) at each step, or a moving average of step rewards. If a shift is detected (`packet.context["regime_changed"] == True`), it signals the agent to adapt.

### `RSABenchmark`
*   **Environment**: Uses `CartpoleEnv`.
*   **Regime Shift Logic**: Manually alters the environment config (e.g., mass) at specific episode boundaries without explicitly notifying the core.
*   **Agent Policy for Adaptation**: 
    When the `ChangepointAdapter` flags `regime_changed = True`:
    1.  **Exploration**: Temporarily increase the exploration rate ($\epsilon$) to gather data in the new regime.
    2.  **Utility Reset**: Decay the utilities of the current working patterns or switch the active context signature in the `PatternManager` to load a fresh/different set of patterns.
    3.  **Pattern Isolation**: Use the `PatternManager` to archive the pre-shift patterns under the old context signature.

## 2. Benchmark Tasks
### Sudden Mass Change (Cartpole)
*   **Phase 1 (Normal)**: Episodes 1-50. Mass = 0.1, Length = 0.5. Baseline learning.
*   **Phase 2 (Heavy)**: Episodes 51-60. Shift mass to 0.5. The agent must detect the drop in performance/confidence, adapt, and learn the new dynamics.
*   **Phase 3 (Normal Return)**: Episodes 61-80. Shift mass back to 0.1. The agent must detect the shift again and rapidly recover its initial performance by recalling patterns from the `PatternManager`.

## 3. Evaluation Protocol & Metrics
*   **Detection Delay**: Number of episodes between the true shift and the `ChangepointAdapter` firing. Target $\le 3$.
*   **Heavy Regime Performance**: Score > 300 within 10 episodes after the shift.
*   **Recovery Performance**: Score > 450 within a few episodes after returning to the Normal regime (retention test).

## 4. Execution Harness (`hpm_ai_v5/experiments/run_rsa_benchmark.py`)
Executes the benchmark and logs the detection points, exploration dynamics, and episode-by-episode scores.