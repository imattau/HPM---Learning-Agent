# Experiment 15: Hierarchical Abstraction Discovery

**Script:** `experiment_hierarchical_abstraction.py`

## Objective
To validate the core claim of the Hierarchical Pattern Modelling (HPM) framework: that learning involves discovering structure across multiple levels of abstraction. This experiment tests whether the system autonomously builds a multi-layered Directed Acyclic Graph (DAG) and reuses lower-level concepts (compositional generalization) instead of memorizing patterns in flat clusters.

## Setup
- **Curriculum (Nested Structure):** 
  - A synthetic dataset simulating linguistic composition:
    - **L1 (Letters):** 10 base concepts.
    - **L2 (Words):** 10 composite concepts, each being an average of two "letters".
    - **L3 (Sentences):** 10 composite concepts, each being an average of two "words".
- **Mechanism:** The `Observer` is fed the curriculum sequentially. We rely on its native **Co-occurrence Compression** dynamics to recognize recurring combinations and collapse them into parent nodes. 
- **Analysis:** After processing, we analyze the topology of the resulting HFN `Forest`, checking for maximum depth and node reuse.

## Results & Analysis
The experiment was a **Success**, demonstrating autonomous hierarchical construction.

1. **Layer Emergence:** The system correctly structured the 30 concepts into a tree with a **Maximum DAG Depth of 3 layers** (Letters $\rightarrow$ Words $\rightarrow$ Sentences).
2. **Compositional Generalization:** Rather than memorizing 30 distinct flat clusters, the system connected the layers via parent-child edges. 
3. **Node Reuse:** The average reuse rate was **1.35 parents per node**, with **12 highly reused nodes**. This proves that the system successfully reused the shared L1 (Letter) and L2 (Word) nodes to explain multiple different L3 (Sentence) observations, exactly matching the structure of the synthetic generator.

### Key Takeaway
The HFN system does not just cluster; it **composes**. By autonomously discovering hierarchies and reusing sub-components, the system achieves exponential efficiency in representation, satisfying the fundamental premise of the HPM theory.
