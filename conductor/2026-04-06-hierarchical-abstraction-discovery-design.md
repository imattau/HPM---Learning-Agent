# SP39: Experiment 15 — Hierarchical Abstraction Discovery (Core HPM Claim)

## 1. Overview and Rationale
The **Hierarchical Abstraction Discovery** experiment is the ultimate validation of the core claim of the Hierarchical Pattern Modelling (HPM) framework: learning is the progressive discovery of structure across *multiple levels of abstraction*. 

We need to prove that the HFN system does not merely form "flat clusters" (like K-means or GMMs). When presented with data that contains an inherent nested structure (e.g., letters → words → sentences), the system should autonomously construct a multi-layered Directed Acyclic Graph (DAG), reusing lower-level nodes to compose higher-level concepts.

## 2. Setup & Execution
- **Curriculum Design (Nested Structure):**
  - We will generate a synthetic dataset with a strict, known hierarchy.
  - **L1 (Letters):** Basic atomic patterns in the manifold.
  - **L2 (Words):** Specific spatial or temporal co-occurrences of L1 patterns (e.g., Pattern A + Pattern B = Concept X).
  - **L3 (Sentences):** Co-occurrences of L2 concepts (e.g., Concept X + Concept Y = Concept Z).
- **The Learning Phase:**
  - The `Observer` is fed a continuous stream of these complex L3 observations.
  - It is heavily reliant on its **Co-occurrence Compression** dynamics. As it sees the same components repeatedly, it should compress them into single nodes.
- **The Structural Analysis Phase:**
  - After training, we will pause and inspect the topological structure of the `Forest`.
  - We will trace the parent-child edges of the resulting graph.

## 3. Evaluation Metrics
1. **Layer Emergence (Depth):** What is the maximum depth of the resulting DAG? (Success requires Depth $\ge$ 3).
2. **Node Reuse Rate:** Are the lower-level nodes (Letters/Words) duplicated for every new sentence, or are they shared and reused across multiple parent nodes? 
3. **Compression Ratio by Level:** How efficiently does the hierarchy represent the data compared to a flat memorization of all sequences?

## 4. Why This Matters
*Does hierarchy emerge, or is it just flat clustering?*
If the system just creates a flat list of 100 "Sentence" nodes, it has failed. It is just memorizing. If it creates 10 "Letter" nodes, 20 "Word" nodes that point to the letters, and 100 "Sentence" nodes that point to the words, it has achieved **Compositional Generalization**. This is the mechanism that allows biological intelligence to learn exponentially faster than standard neural networks: when a new word is learned, it doesn't need to relearn the letters.

## 5. Implementation Roadmap
1. **Hierarchical Generator:** Implement a data generator that creates compositional vectors (e.g., using specific slices of the manifold for different "slots" in a sequence).
2. **DAG Analyzer:** Write a utility to traverse the HFN `Forest`, calculating maximum depth, average children per parent, and average parents per child (reuse).
3. **Experiment Script:** Create `hpm_fractal_node/experiments/experiment_hierarchical_abstraction.py` to run the curriculum and output the structural topology report.
