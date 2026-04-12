# SP60: Cumulative Abstraction via Multi-Polygraph HFNs (Design)

## 1. Objective
To elevate the HPM framework to a **Self-Improving Symbolic Learner** using its native **Multi-Polygraph HFN** architecture. This experiment will demonstrate **Cumulative Abstraction**: the agent will synthesize a complex operator chain, encapsulate it as a structural Polygraph node (a "macro"), store it in the HFN manifold, and reuse this structural abstraction to drastically accelerate future reasoning tasks.

## 2. The Paradigm Shift: Multi-Polygraph Encapsulation
In SP59, the manifold was flat (a 90D vector). While search was guided, composed chains were ephemeral. In SP60, we leverage the full fractal graph capability of HFNs.
*   **The Polygraph Node**: When a useful composition (e.g., `Add_1 ∘ Mul_2`) is discovered, it is not just flattened into a new parameter vector. Instead, the agent creates a **Multi-Polygraph HFN Node**.
*   **Structural Composition**: The new macro-node has explicit DAG edges (`inputs=[Add_1_node, Mul_2_node]`, `relation_type="macro"`). Its geometric $\mu$ vector summarizes the *effective* input/output delta, while its graph structure preserves the *causal mechanism*.
*   **Fractal Retrieval**: The Retriever can now return this Polygraph node just like any primitive. When the `CognitiveSolver` or `BeamSearch` evaluates it, it can dynamically unpack its subgraph to execute the composition.

## 3. Key Upgrades from SP59

### A. Feedback Loop (Storing Structural Polygraphs)
*   **Mechanism**: Once `ManifoldGuidedBeamSearch` identifies a winning operator chain, it triggers a Co-occurrence Compression event.
*   **Graph Injection**: A new `HFN` node is instantiated with `inputs` pointing to the constituent primitives. This Polygraph node is registered in the `Forest`, making the composition permanently retrievable.

### B. Accelerated Secondary Induction (The Test of Learning)
To prove the system has learned a structural abstraction, it must face a complex task that builds upon the first.
*   If the agent learns $Polygraph_{A} = Mul\_2 \circ Add\_1$, a subsequent task requiring $Mod\_10 \circ Polygraph_{A}$ should be solvable in **Depth 2** search, whereas a naive agent (without the Polygraph node) would require **Depth 3**.

### C. True Mahalanobis Distance Retrieval
Retrieval will use the HFN's $\Sigma$ (covariance) matrix via the `log_prob` method. This allows the system to naturally ignore noisy or irrelevant dimensions based on learned variance, naturally guiding the search toward the effective deltas of the Polygraphs.

## 4. Experimental Curriculum

### Phase 1: Primitive Pre-training
Train the agent on base primitives: `Add_1`, `Mul_2`, `Mod_10`. These are registered as root HFN nodes.

### Phase 2: Task A (The Catalyst for Polygraph Formation)
*   **Task**: $x \to (x + 1) \times 2$ (Sequence: `1 -> 4 -> 10 -> 22...`)
*   **Action**: The agent uses guided search to find the chain `Mul_2 ∘ Add_1`.
*   **The Leap**: The agent compresses this chain into a new Multi-Polygraph HFN node (e.g., `macro(Add_1+Mul_2)`) and registers it in the Forest.

### Phase 3: Task B (Cumulative Transfer)
*   **Task**: $x \to ((x + 1) \times 2) \pmod{10}$ (Sequence: `1 -> 4 -> 0 -> 2 -> 6...`)
*   **Action**: The agent searches for a solution.
*   **Verification**: The agent must retrieve and use the newly stored Polygraph node, solving the task at Depth 2, rather than rebuilding the full Depth 3 chain from scratch.

## 5. Success Metrics and Baselines

| Condition | Task B Search Depth | Task B Nodes Evaluated | Result |
| :--- | :--- | :--- | :--- |
| **Static Manifold (SP59)** | 3 | High (Combinatorial expansion) | **Baseline** |
| **Multi-Polygraph HPM (SP60)** | 2 | Low (Guided by Polygraph prior) | **SUCCESS** |

## 6. Conclusion
If the agent successfully shortens its search depth on Task B by reusing the Multi-Polygraph node from Task A, it provides definitive proof that the HFN is a **Cumulative Abstraction Engine**. It demonstrates that structural knowledge can be encapsulated into subgraphs, geometrically retrieved, and functionally executed, unifying neural manifold search with classic symbolic chunking.
