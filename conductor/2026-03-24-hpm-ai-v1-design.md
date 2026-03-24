# HPM AI v2: Abstract Relational Intelligence & Universal Logic
**Design Specification (v1.2 - The MMR & Cross-Modal Bridge)**

## 1. Overview
The HPM AI v2 addresses the "Ontological Gap" between code syntax and relational logic. It transitions from AST-bound mutations to **Middle-Manifold Representation (MMR)**, allowing the system to model logic as language-independent graphs. It introduces **Metacognitive Exploration** to escape local minima and establishes the **Cross-Modal Logic Bridge (SP19)** to prove that discovered laws are universal invariants.

## 2. Core Architecture

### 2.1 Middle-Manifold Representation (MMR)
- **Module**: `hpm_ai_v1.transpiler.mmr`
- **Function**: Decouples "Truth" from "Syntax".
- **Logic**: 
    - `to_relational_graph(ast_node)`: Converts Python syntax into an abstract graph of relational primitives.
    - `from_relational_graph(graph)`: Synthesizes valid Python AST from the abstract graph.
- **Benefits**: Substrate independence; the agent modifies the *semantic intent* rather than the *syntactic tokens*.

### 2.2 Metacognitive Exploration (Bloat Window)
- **Module**: `hpm_ai_v1.core.l5_compiler`
- **Logic**: Implements a stagnation-triggered "Bloat Window".
- **Rule**: If stagnation > 3 generations, L5 permits a node count increase of <20% if the **Novelty Score** (Surprise $S$) is high. This allows "tunnelling" through complex implementations to reach a new algorithmic peak (e.g., discovering vectorization).

### 2.3 Active Prior Injection (Prior-Guided Recombination)
- **Module**: `hpm_ai_v1.main` / `hpm_ai_v1.core.mutator`
- **Source**: `PyPISubstrate` (Memoization, Vectorization, Symmetry patterns).
- **Process**: Mined snippets are encoded into L3 Relational Tokens and injected as **Blueprints** into the agent's candidate pool during the `RecombinationStep`.

### 2.4 The Rosetta Refactor (SP19 - Cross-Modal Bridge)
- **Objective**: Prove Functional Universalisation.
- **Process**:
    1. Agent refactors its own code (e.g., an L5 symmetry check).
    2. The discovered "Symmetry Law" is saved to the `Librarian`.
    3. An ARC Agent (visual domain) pulls that law to solve a grid puzzle.
- **Validation**: Success confirms the AI is discovering **Universal Invariants**, not just code optimizations.

## 3. Directory Structure: `hpm_ai_v1/` (Refined)
- `core/`: Now includes the SuccessionController and Bloat-aware L5 Judge.
- `transpiler/`: Now includes the MMR graph engine.
- `substrates/`: Knowledge and Codebase ingestion with active prior injection.
- `sandbox/`: AST-Native Refactoring environment.
- `store/`: Concurrent pattern storage with lineage and parentage.
