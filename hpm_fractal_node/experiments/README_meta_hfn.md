# Experiment 8: Meta-HFN Utilisation (Adaptation & Substrate Efficiency)

**Script:** `experiment_meta_hfn.py`

## Objective
To test if self-representation (the `meta_forest`) accelerates adaptation to concept drift and improves structural self-organization without explicit supervision. The experiment tests learning under pressure to see if an agent that manages its own state can outperform a "dumb" accumulator.

## Setup
- **Curriculum:** A sequence of 3 structured patterns in Phase A, followed by a sudden shift to 3 completely different patterns in Phase B.
- **Environment:** Two agents competing under "Resource Pressure" (a strict `budget=2` for expansion).
- **Configurations:** 
  - **Agent 1 (Meta-Active):** Full HFN dynamics, tracking hit/miss counts, weight decay, and active pruning/absorption of obsolete nodes.
  - **Agent 2 (Ablated):** Accumulator only (disabled weight updates, disabled pruning, disabled absorption).

## Results & Analysis
The experiment yielded a resounding success, proving the critical role of self-representation.

1. **Phase A (Stabilization):** Both agents quickly matched the Phase A patterns. However, the Meta-Active agent continually refined its structure, while the Ablated agent grew indefinitely.
2. **Phase B (The Shift):** Upon the introduction of completely new patterns, both agents spiked in surprise.
3. **Recovery & Structural Efficiency:**
   - **Ablated Agent:** Ended with **211 nodes**. With no mechanism to recognize overlap or obsolescence, it blindly accumulated Phase A and Phase B nodes, becoming slow and bloated.
   - **Meta-Active Agent:** Ended with **97 nodes**. It pruned redundant Phase A nodes and aggressively absorbed overlapping structures. During the shift, its size grew to accommodate Phase B, then shrank again as it organized the new knowledge.
4. **Complexity-Error Product:** A combined metric of efficiency (`Avg Surprise * Final Size`). The Meta-Active agent scored **~723**, while the Ablated agent scored **~1574**. The Meta-Active agent was over twice as efficient.

### Key Takeaway
Self-representation (the `meta_forest`) enables **Dynamic Structural Pruning**. The agent maintains a lean, efficient world model that adapts to change by actively discarding old knowledge that no longer explains the present. The system is structurally sovereign: it manages its own growth and decay based on environmental pressure.
