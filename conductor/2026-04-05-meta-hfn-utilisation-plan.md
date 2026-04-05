# Objective
Design and implement "Experiment 8: Meta-HFN Utilisation" to validate that the system's self-representation (the `meta_forest`) accelerates adaptation and improves structural self-organization without explicit supervision, particularly when faced with a sudden environmental shift.

# Background & Motivation
The HPM framework asserts that learning is structural self-organization under pressure. A critical new capability is the `meta_forest`, which represents the state of the primary `forest` (tracking weights, scores, hit/miss counts) as HFN nodes themselves. By observing its own state, the system should be able to guide absorption, creation, and weight dynamics more effectively than a system without this meta-awareness. We need to prove that this self-representation actually provides a tangible benefit: faster adaptation to concept drift.

# Scope & Impact
- **New File:** `hpm_fractal_node/experiments/experiment_meta_hfn.py`
- **Metrics Tracked:** 
  - Adaptation speed (epochs to recover low surprise after a shift)
  - Node states (weight, miss counts) over time
  - Structural efficiency (number of redundant nodes pruned or absorbed)
- **Impact:** Demonstrates the value of Layer 2/3 (Dynamics and Gatekeepers) self-representation, proving that a system that "knows what it knows" learns faster and more efficiently.

# Proposed Solution
1. **Curriculum Design:**
   - **Phase A (Stable):** A sequence of structured observations (e.g., Pattern Set A).
   - **Phase B (Sudden Shift):** A completely different sequence of observations (e.g., Pattern Set B), introduced midway through the experiment.
2. **A/B Testing Setup:**
   - **Agent 1 (With Meta-Awareness):** Uses the full `Observer` dynamics, relying on the `meta_forest` for weight updates, pruning, and absorption (guided by miss counts and scores).
   - **Agent 2 (Ablated Meta/Baseline):** We will ablate or severely restrict the meta-driven dynamics (e.g., disable absorption, disable weight decay, or freeze the `meta_forest` updates) to simulate a system without active self-representation.
3. **Execution:**
   - Run both agents through Phase A until their world models stabilize (surprise drops).
   - Introduce Phase B.
   - Measure how quickly each agent's residual surprise drops back to zero.
   - Track the size and composition of their respective forests.
4. **Expected Outcome:** 
   - Both agents learn Phase A.
   - Upon the shift to Phase B, both spike in surprise.
   - **Agent 1 (Meta)** should recover significantly faster because its `meta_forest` will quickly identify that the Phase A nodes are now "missing" (accumulating miss counts), causing their weights to drop and triggering absorption/pruning. This clears the hypothesis space for Phase B patterns.
   - **Agent 2 (Baseline)** will suffer from "shadowing" or interference from obsolete Phase A nodes, recovering slower and maintaining a bloated, inefficient structure.

# Alternatives Considered
- **ARC Tasks vs Synthetic Data:** Synthetic data (similar to Exp 7) is preferable for this targeted test because we can strictly control the "concept drift" without introducing the reasoning noise of full ARC grids.
- **How to Ablate Meta:** Completely disabling the `meta_forest` would break the `Observer`. Instead, we will ablate the *policies* that rely on it (e.g., setting `absorption_miss_threshold` to infinity, disabling weight updates, or disabling pruning).

# Implementation Steps
- [ ] **Step 1:** Create `hpm_fractal_node/experiments/experiment_meta_hfn.py`.
- [ ] **Step 2:** Implement the two-phase synthetic curriculum generator (Phase A -> Phase B shift).
- [ ] **Step 3:** Setup the A/B testing loop (Full Observer vs Ablated Observer).
- [ ] **Step 4:** Track `residual_surprise` and `forest_size` epoch by epoch for both agents.
- [ ] **Step 5:** Generate a comparative report highlighting adaptation speed (Epochs to Recovery) and structural efficiency (Final Forest Size).

# Verification & Testing
- Run `python hpm_fractal_node/experiments/experiment_meta_hfn.py`.
- Verify that Agent 1 recovers from the Phase B shift in fewer epochs than Agent 2.
- Verify that Agent 1's final forest size is smaller/more compressed than Agent 2's, proving that structure improves without explicit supervision.
