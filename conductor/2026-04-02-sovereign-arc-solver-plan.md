# Implementation Plan: Sovereign ARC Solver (SP27)

## 1. Goal
Implement a full cognitive loop for ARC reasoning: observe training examples, deduce generative rules via Stereo Vision, and synthesize the test output grid using the Agnostic Decoder.

## 2. Architecture

### 2.1 The Perceptual Cluster
We reuse the established Perceptual Cluster from SP18 (Sovereign ARC):
- **Spatial Specialist**: 100D grid delta ($\Delta$) observer.
- **Symbolic Specialist**: 30D numerical invariant observer.
- **Explorer**: 150D generalist.

### 2.2 The Generative Decoder
- **Spatial Decoder**: A new worker process containing a `Target Forest` of all seen and potential 100D grid deltas. It uses `hfn.Decoder` to collapse abstract geometric rules into concrete pixel shifts.

## 3. Smoke Test (System Verification)

Before running full ARC tasks, the solver must pass these three system integrity checks:

1.  **Cluster Communication**: Governor broadcasts a 150D test vector; all 4 processes (Spatial, Symbolic, Explorer, Decoder) must return a valid heartbeat message.
2.  **Topological Retrieval**: Given a `Rule_Node` (e.g., identity), the Spatial Decoder must successfully resolve it to a zero-delta `100D` leaf.
3.  **Active Curiosity Loop**: Governor dispatches a Goal with a deliberately missing leaf; the Decoder must correctly emit a `ResolutionRequest`, and the Governor must successfully trigger the Explorer to fulfill it.

## 4. The Generative Loop (The Algorithm)

### Phase 1: Rule Induction (Bottom-Up)
1. Governor loads an ARC task.
2. For each training example `(In, Out)`:
   - Extract multi-modal feature vector `v`.
   - Broadcast `v` to Perceptual Cluster.
   - Collect "Explanation Winners" (the specific HFN nodes that the specialists claim best describe the transformation).
3. **Stereo Vision Aggregation**: The Governor looks for a persistent combination of explanation nodes across *all* training examples. 
   - *Example*: If Training Example 1, 2, and 3 all return `[Spatial: Rotation_90]` and `[Symbolic: Unique_Color_Preserved]`, this combination becomes the **Generative Rule ($R$)**.

### Phase 2: Goal Formulation
1. Extract features for the Test Input.
2. Create an abstract **Goal HFN ($G$)**:
   - `G.mu` = The spatial features of the Test Input.
   - `G.sigma` = High variance ($10.0$) because the output is unknown.
   - Add topological edges to enforce the Generative Rule $R$: `G.add_edge(G, Rule_Node, "MUST_SATISFY")`.

### Phase 3: Variance Collapse (Top-Down)
1. Governor dispatches $G$ to the **Spatial Decoder**.
2. Decoder attempts to find a concrete 100D delta ($\Sigma \approx 0$) in its Target Forest that maximizes the topological fit with the required Rule edges.
3. If no candidate is found, Decoder issues a `ResolutionRequest` (Demand-Driven Learning - SP24).
4. Explorer scans historical context (training examples), creates the missing node, and Decoder retries.

### Phase 4: Output Synthesis
1. Decoder returns the concrete 100D $\Delta$ vector.
2. Governor reconstructs the 10x10 output grid: `Output_Grid = Input_Grid + \Delta`.

## 4. Draft Code Structure (`experiment_sovereign_arc_solver.py`)

```python
"""
SP27: Sovereign ARC Solver.
Closes the cognitive loop: Observe -> Deduce Rule -> Decode Output.
"""
# ... imports and setups ...

def run_experiment():
    print("SP27: Sovereign ARC Solver\n")
    
    # 1. Initialize 4-Process Cluster
    # Perceptual: Spatial_Spec, Symbolic_Spec, Explorer
    # Generative: Spatial_Decoder
    
    tasks = load_sovereign_tasks()
    
    for task in tasks:
        # --- Phase 1: Decentralized Observation ---
        train_explanations = []
        for ex in task["train"]:
            # Broadcast to Perceptual Cluster
            # Collect winners (e.g., [Node_Rot90, Node_CountPreserved])
            train_explanations.append(winners)
            
        # Deduce Rule (Intersection of all train explanations)
        rule_nodes = find_intersection(train_explanations)
        
        # --- Phase 2: Goal Formulation ---
        test_ex = task["test"][0]
        goal = HFN(mu=test_ex["input_features"], sigma=np.ones(100)*10.0, id="goal_test")
        for rn in rule_nodes:
            goal.add_edge(goal, rn, "MUST_SATISFY")
            
        # --- Phase 3: Sovereign Decoding ---
        # Dispatch to Spatial Decoder
        # Implement Fail-Learn-Retry curiosity loop if request stalls
        
        # --- Phase 4: Output Construction ---
        delta = decoded_leaf.mu
        reconstructed_grid = reconstruct_grid(test_ex["input_grid"], delta)
        score_accuracy(reconstructed_grid, test_ex["output_grid"])
```

## 5. Review Against Specification

- **Generative Workflow**: *Pass*. The plan explicitly outlines the transition from bottom-up observation (Phase 1) to top-down decoding (Phase 3).
- **Decentralized Architecture**: *Pass*. It utilizes the 3 perceptual specialists + 1 dedicated generative decoder.
- **Goal Formulation**: *Pass*. The Governor correctly constructs an abstract high-variance HFN and applies topological constraints based on the deduced rule.
- **Anti-Hallucination/Curiosity**: *Pass*. Phase 3 includes the `ResolutionRequest` handling to trigger learning if the exact required delta isn't in the Target Forest.
- **Output Synthesis**: *Pass*. Reconstructs the grid via `Input + Delta`.
