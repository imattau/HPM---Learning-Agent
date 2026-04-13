# Full Integration of HFN Core into Agent Layer

## Objective
Transform the existing `hpm_ai_v2` agent layer to fully utilize all capabilities of the underlying HFN core architecture. This refactor bridges the gap between the theoretical capabilities of the HFN nodes (pluggable models, structural retrieval, recombination, affective evaluation, density tracking) and their practical application in the agent's problem-solving loop.

## Motivation
While the core `hfn/` library supports advanced probabilistic and dynamic features, the `BaseHFNAgent` and its mixins currently use a minimal subset (e.g., flat Gaussians, simple geometric retrieval). By exposing and utilizing these advanced features, the agent becomes a full-stack HPM AI capable of multi-modal learning, active curiosity-driven exploration, and resilient pattern absorption, aligning fully with the theoretical framework of Hierarchical Pattern Modelling.

## Implementation Steps

### 1. Enable Mixture Models for Macros
**Target:** `hpm_ai_v2/agents/mixins/l2_macro.py` (and relevant `BaseHFNAgent` logic if needed)
- **Action:** Update `register_macro` and `register_code_macro` to accept an optional `k_components: int = 1` parameter.
- **Logic:**
  - If `k_components > 1`, instantiate a `GaussianMixtureModel` (GMM) with `k_components` components. Initialize the means by adding small random noise to the computed solution state. Set initial weights uniformly.
  - Pass this `prob_model` when creating the `HFN` node.
  - If `k_components == 1`, continue using the default `FlatGaussianModel` behavior.

### 2. Switch to Hybrid Retrieval
**Target:** `hpm_ai_v2/agents/base_agent.py` (`BaseHFNAgent.__init__`)
- **Action:** Introduce a `retriever_type: str = "geometric"` parameter.
- **Logic:**
  - If `"hybrid"`, instantiate `HybridRetriever(self.forest)`.
  - If `"structural"`, instantiate `StructuralRetriever(self.forest)`.
  - Otherwise (default), use `GeometricRetriever(self.forest)`.
  - Ensure the selected retriever is passed to the `Observer`.

### 3. Integrate Recombination as a Strategy
**Target:** `hpm_ai_v2/agents/mixins/recombination.py`
- **Action:** Implement `_try_recombine(self, inputs, outputs)`.
- **Logic:**
  - Iterate over pairs of existing macros in `self.patterns`.
  - Use `self.recombine_patterns` to create a new composite node.
  - Render the node to code and execute it against the inputs.
  - If it produces the correct outputs, register the new pattern and return `[new_node]` (or `code, depth` depending on the agent's solve loop expectations).
- **Integration:** Ensure agents like `SocialAnalogicalAgent` explicitly call `self.add_strategy("recombine", self._try_recombine)`.

### 4. Affective Curiosity for Active Learning
**Target:** `hpm_ai_v2/agents/base_agent.py` or a specific experiment utility.
- **Action:** Add a method `select_next_task(self, task_pool, learnability_dict)`.
- **Logic:**
  - Use `self.observer.affective_evaluator.curiosity_exploration_probability(learnability)` to compute the probability of selecting each task based on its current learnability estimate.
  - Normalize the probabilities and sample the next task.
  - Note: Requires the agent to have `use_affective_evaluator=True` enabled.

### 5. Meta-Controller Ranking of All Strategies
**Target:** `hpm_ai_v2/utils/meta_controller.py`
- **Action:** Extend `MetaStrategyController.DEFAULT_ORDER`.
- **Logic:**
  - Update `DEFAULT_ORDER = ["exact", "decompose", "imagine", "bfs", "analogy", "social", "recombine", "compose"]`.
  - This ensures the meta-controller knows about all possible strategies and can rank them appropriately once stats are gathered.

### 6. Density-Modulated Absorption
**Target:** `hfn/observer.py`
- **Action:** Modify `Observer._check_absorption(self, result: ObserverResult, new_node: HFN)`.
- **Logic:**
  - After computing `effective_miss_threshold`, check if `self.density_tracker` is active.
  - Retrieve the density for the existing node: `density = self.density_tracker.get_total_density(node.id)`.
  - Apply a scaling factor: `density_factor = 1.0 + density`.
  - Adjust the threshold: `effective_miss_threshold = int(round(effective_miss_threshold * density_factor))`.
  - This makes dense ("sticky") nodes harder to absorb/overwrite.

## Backward Compatibility
- All new parameters (`k_components`, `retriever_type`) will have defaults that preserve existing behavior.
- Density modulation will only occur if `use_density_tracker=True` is explicitly passed during agent initialization.
- The extended meta-controller strategy list safely ignores strategies that are not registered by a specific agent instance.

## Verification
- Run existing experiments (`experiment_ms_sl.py`, `experiment_math_physics.py`) to ensure baseline behavior is unaffected.
- Create or update an experiment script to explicitly test `k_components=2`, `retriever_type="hybrid"`, and active learning task selection.
