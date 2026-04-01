# SP22: The Agnostic Decoder — Design Specification

## 1. Overview and Rationale

The **Agnostic Decoder** experiment (SP22) formalises the generative half of the HPM AI. Where the Observer (Bottom-Up) compresses concrete observations into abstract representations (high $\Sigma$), the Decoder (Top-Down) collapses abstract goals into concrete actions (low $\Sigma$).

Crucially, this Decoder must reside purely in `hfn/` and contain **zero domain-specific logic**. It knows nothing of language, math, or physics. It operates solely on:
1.  **Geometric Variance ($\Sigma$)**: To distinguish abstract concepts from concrete tokens.
2.  **Topological Edges**: To resolve constraints when "instantiating" an abstract node.

## 2. The Universal Decoder Algorithm

The `hfn.decoder.Decoder` class takes three arguments:
*   `goal_node`: The abstract HFN to collapse.
*   `target_forest`: The HFN manifold containing the allowable output leaves.
*   `sigma_threshold`: The maximum variance allowable for a node to be considered "concrete" (e.g., $10^{-3}$).

### The Recursive "Variance Collapse" Loop:

For any given node $N$:
1.  **Concrete Check**: If $N.\Sigma \le \sigma_{threshold}$ AND $N \in target\_forest$, return $[N]$.
2.  **Explicit Expansion**: If $N.\Sigma > \sigma_{threshold}$ AND $N$ has explicit `children()`, recursively decode each child in order. Return the concatenated list of results.
3.  **Implicit Resolution (The "Slot" Fill)**: If $N.\Sigma > \sigma_{threshold}$ but $N$ has *no children* (it is a dangling abstract concept), the Decoder must "instantiate" it.
    *   **Retrieve**: Query $target\_forest$ for the top $k$ nodes closest to $N.\mu$.
    *   **Score**: For each candidate $C$, evaluate how well it satisfies $N$'s topological edges (e.g., if $N$ has an `IS_A` edge to Node $X$, does $C$ also have a path to $X$?).
    *   **Select**: Choose the candidate $C^*$ that maximizes the joint `log_prob` and topological fit. Return $[C^*]$.

## 3. The Test Domain: "Abstract Block Stacking"

To prove the Decoder is domain-agnostic, we will NOT use language. We will use a synthetic 1D block-stacking task. This ensures we don't accidentally leak NLP concepts into the core `hfn` code.

### The Manifolds:
*   `Target Forest` (The "Hand"): 1D points representing specific coordinates (e.g., $X=1.0$, $X=2.0$). These have $\Sigma \approx 0$.
*   `Object Forest` (The "Blocks"): Nodes representing "Red Block", "Blue Block".
*   `Rule Forest` (The "Concepts"): Abstract nodes like "Stack", "Next To".

### The Priors:
*   `Node_Red`: Concrete location $X=2.0$.
*   `Node_Blue`: Concrete location $X=5.0$.
*   `Node_NextTo`: Abstract concept. $\mu = 1.0$ (represents a delta offset), high $\Sigma$.

## 4. The Experiments (The "Collapse" Scenarios)

We will test the Decoder with three progressively harder generative goals.

### Test 1: Explicit Expansion (The Script)
*   **Goal**: An HFN `Goal_Sequence` with high $\Sigma$, containing two explicit children: `Node_Red` and `Node_Blue`.
*   **Expected Behavior**: Decoder sees high $\Sigma$, expands to children. Both children have $\Sigma=0$ and are in the Target Forest.
*   **Expected Output**: `[X=2.0, X=5.0]`.

### Test 2: Implicit Resolution (The Search)
*   **Goal**: An abstract HFN `Goal_Find` with no children, but an edge `HAS_COLOR -> RED`. Its $\mu$ is roughly near $2.0$.
*   **Expected Behavior**: Decoder sees high $\Sigma$, no children. It queries the Target Forest using $\mu$, retrieves candidates, and selects the one that satisfies the `RED` edge constraint.
*   **Expected Output**: `[X=2.0]`.

### Test 3: Relational Synthesis (The "Grammar")
*   **Goal**: An abstract HFN `Goal_Stack` with two children: `Node_Red` and `Node_NextTo`.
*   **Expected Behavior**: 
    1.  Decodes `Node_Red` $\rightarrow$ `X=2.0`.
    2.  Decodes `Node_NextTo` $\rightarrow$ It's abstract. It needs a concrete action. The Decoder queries the Target Forest for a location relative to the previous context ($X=2.0 + 1.0 = 3.0$).
*   **Expected Output**: `[X=2.0, X=3.0]`.

## 5. Evaluation Metrics

1.  **Agnosticism**: Does the `hfn.decoder.py` file contain any domain-specific strings, rules, or logic? (Must be purely geometric/topological).
2.  **Convergence**: Does the algorithm successfully terminate at concrete Target nodes for all three test cases?
3.  **Constraint Adherence**: In Test 2, does it correctly reject nearby blocks that do not match the required color edge?

## 6. Implementation Roadmap

1.  **Core HFN Decoder**: Implement `hfn/decoder.py` according to the Variance Collapse algorithm.
2.  **Experiment Setup**: Create `hpm_fractal_node/experiments/experiment_agnostic_decoder.py`.
3.  **Domain Construction**: Build the synthetic 1D block forests and register the priors.
4.  **Execution**: Run the three generative tests and validate the output coordinates.
