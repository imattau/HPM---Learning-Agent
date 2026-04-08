# SP53: Experiment 29 — Robust Intent Inference and AST Synthesis

## 1. Objective
To redesign the Intent-Driven Reasoning pipeline to address critical architectural flaws (Oracle Leakage, Lossy State, Brittle Rendering) identified in Experiment 28. The system must synthesize programs from **partial intent constraints** using an **expanded semantic state space** and an **AST-based renderer**, proving true structural comprehension without hardcoded fallbacks.

## 2. Background & Motivation
Feedback on Experiment 28 revealed that the "Semantic Bridge" was heavily scaffolded. The `IntentOracle` collapsed intent inference into exact output reconstruction (Oracle Leakage), the 9D state vector was too lossy to represent complex distributions or program structures, and the string-based `CodeRenderer` was brittle. Furthermore, falling back to pre-constructed chunks undermined the claim of autonomous planning.

This experiment tackles these issues by:
1.  **Constraint-Based Oracle**: The LLM Oracle no longer provides the exact expected output. Instead, it provides *semantic constraints* (e.g., target type, structural hints like "requires loop", and statistical properties).
2.  **Expanded State Space**: Moving from 9D to a richer continuous state space that includes distributional features (e.g., mean) and structural markers (AST depth, branch count).
3.  **AST-Based Rendering**: Replacing string concatenation with Python's `ast` module to guarantee syntactically valid code structures and true scope management.
4.  **Multi-Objective Planning**: The DFS planner will score candidates based on semantic distance *and* structural consistency, with absolutely **zero hardcoded fallbacks**.

## 3. Setup & Environment

### Domain: Constraint-to-AST Space
- **State Representation**: 14D extended semantic vector:
    - *Execution Features*: `[Returned, TypeSignature (List/Scalar), ListLen, MeanVal, FirstVal]`
    - *Structural Markers*: `[ASTDepth, LoopCount, BranchCount, StatementCount, CallCount]`
    - *Execution State*: `[IteratorActive, ConditionActive, ListInit, ErrorFlag]`
- **AST Renderer**: 
    - Primitives (e.g., `OP_ADD`) map to AST nodes (e.g., `ast.AugAssign`).
    - Composition nodes (`FOR_LOOP`) map to AST blocks (`ast.For`).

### The Constraint Oracle
Instead of mapping `(Prompt, Input) -> Output`, the Oracle maps `Prompt -> Goal Constraints Vector`.
*Example*: "Double all numbers" -> Goal state specifies: `TypeSignature = List`, `LoopCount = 1`, `MeanVal = InputMean * 2`, `ASTDepth = 2`.

## 4. Architectural Enhancements

### 1. Expanded Semantic State & Oracles
The `StateOracle` is upgraded to compute statistical properties (mean, length) and structural properties from the current AST. The `IntentOracle` outputs a partial constraint vector that the planner aims to satisfy. 

### 2. AST Node Folding (`ASTRenderer`)
The HFN nodes no longer represent strings; they represent AST transformations.
- `FOR_LOOP` applies an `ast.For` wrapper.
- `COND_IS_EVEN` applies an `ast.If`.
- `RETURN` applies an `ast.Return`.
This completely decouples the planner from string indentation heuristics, ensuring 100% syntactically valid code (via `ast.unparse()`).

### 3. Multi-Objective DFS Planner
The planner minimizes a composite loss:
`Loss = w1 * ||SemanticDelta|| + w2 * ASTComplexity + w3 * (1 / PriorWeight)`
No fallback to pre-built chunks is allowed. If the planner times out, the attempt is marked as a failure, forcing the system to rely purely on generative search and retrieval.

## 5. Evaluation Metrics
1.  **Zero-Shot Constraint Satisfaction**: The ability to synthesize an AST that meets the target semantic constraints without knowing the exact output beforehand.
2.  **Structural Robustness**: 100% of generated programs must compile to valid Python ASTs (zero `SyntaxError` or `IndentationError`).
3.  **Autonomous Planning Yield**: Success rate of the DFS planner solving the constraints strictly via retrieval and composition, without fallbacks.

## 6. Implementation Steps
- [ ] **Step 1**: Expand the semantic state dimensions (`S_DIM = 14`) to include structural and distributional features.
- [ ] **Step 2**: Implement `ASTRenderer` using Python's `ast` module to map HFN concepts to AST nodes and generate source code via `ast.unparse()`.
- [ ] **Step 3**: Implement `ConstraintOracle` to map intents to partial goal vectors instead of exact outputs.
- [ ] **Step 4**: Update the DFS planner to use multi-objective scoring and strictly remove all hardcoded fallbacks.
- [ ] **Step 5**: Execute the intent curriculum and evaluate AST compilation rates and semantic goal satisfaction.