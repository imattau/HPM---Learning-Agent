# SP35: Experiment 11 — True Cross-Domain Transfer (Structural Equivalence)

## 1. Overview and Rationale
The **True Cross-Domain Transfer** experiment evaluates the HFN system's ability to recognize structural equivalence across different domains *without* external alignment tricks (like the Procrustes alignment used in SP10). 

If the HFN framework genuinely learns "structure" rather than just fitting "embedding geometry," it should be able to reuse nodes learned in Domain A (e.g., Math) to solve structurally identical problems in Domain B (e.g., Symbolic Transformations), provided the system can discover the mapping through overlap and co-occurrence.

## 2. Setup & Execution
- **Curriculum:** 
  - **Domain A (Math):** Sequences of numbers following a structural rule (e.g., `+1` -> `[1, 2, 3]`, `+2` -> `[2, 4, 6]`).
  - **Domain B (Symbolic):** Sequences of abstract symbols following the *exact same* structural rule, but embedded in a completely different region of the manifold (e.g., `Shift_Right` -> `[A, B, C]`, `Shift_Rightx2` -> `[A, C, E]`).
- **The Challenge:** Domain A and Domain B exist in orthogonal subspaces of the 100D manifold (e.g., Math uses dims 0-50, Symbolic uses dims 50-100). No Euclidean distance overlaps between the raw inputs.
- **The Mechanism (Cross-Domain Anchoring):** We will provide a small set of "Rosetta Stone" examples where the structural transformation is applied simultaneously to both domains. The `Observer` must use these to build a high-level `Composite Node` (a "Concept") that links the Math transformation rule to the Symbolic transformation rule via co-occurrence.
- **Testing:** We then probe Domain B. The system must retrieve the shared high-level concept, map it down to Domain A's rule to leverage the deep structural history, and then project the prediction back to Domain B.

## 3. Evaluation Metrics
1. **Node Reuse Rate:** What percentage of nodes activated during Domain B testing were originally created during Domain A training? (Success requires high reuse).
2. **New Node Creation:** Does the system create a completely redundant parallel hierarchy for Domain B, or does it rely on Domain A's structure? (Success requires minimal new creation).
3. **Transfer Accuracy:** Can the system correctly predict Domain B sequences using the shared structural rules?

## 4. Why This Matters
*Is HFN learning structure, or just embedding geometry?*
If the system builds a completely new forest for Domain B, it is only learning surface-level geometry. If it builds a "Bridge Node" and reuses the deep structure of Domain A, it is achieving **True Abstraction**—the holy grail of Artificial General Intelligence (AGI). It proves the system can separate the *rule* from the *data it operates on*.

## 5. Implementation Roadmap
1. **Orthogonal Domains:** Create synthetic data generators for Domain A (Dims 0-50) and Domain B (Dims 50-100).
2. **Rosetta Curriculum:** Create a 3-stage training phase:
   - Train Domain A heavily.
   - Present a few "Rosetta" examples (joint A+B observations) to trigger co-occurrence compression.
   - Probe Domain B.
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_true_cross_domain.py` to run the curriculum, tracking node ancestry and reuse rates across domains.
