# SP54: Experiment 30 — Execution-Guided Synthesis & Empirical Priors

## 1. Objective
To redesign the synthesis pipeline to absolutely eliminate Oracle Leakage and symbolic state simulation. The agent must discover program structures purely through **Execution Feedback** (running candidate ASTs on real inputs) and must learn the semantic prior values (`mu`) online rather than relying on hand-coded approximations.

## 2. Background & Motivation
Feedback on Experiment 29 correctly pointed out that while the AST renderer was robust and the Oracle output was masked, the planner still operated on "symbolic simulation" (e.g., `next_state[3] += 1.0`). This is a critical flaw: it assumes the agent *already knows* exactly what each concept does, essentially pre-solving the intent mapping. 

To achieve true Artificial General Intelligence, the agent cannot be fed the answers in its priors. This experiment transitions the agent to a **Beam Search Planner guided by an Empirical State Oracle**.

## 3. Architectural Enhancements

### 1. The Empirical State Oracle
Instead of a 14D vector representing symbolic constraints, the `EmpiricalOracle` takes actual Python inputs and outputs and computes an empirical feature vector (e.g., `[IsValid, IsList, Length, Mean, Min, Max, IsInt]`). 
- The Goal State is precisely the feature vector of the `expected_outputs`.
- The Current State is precisely the feature vector produced by executing the current AST candidate.

### 2. Execution-in-the-Loop Planning
The planner uses Beam Search over the AST structures. 
- At each depth $d$, it composes the current AST with a new primitive.
- It renders the code and calls the `PythonExecutor` to run it against the test inputs.
- The resulting outputs are passed to the `EmpiricalOracle` to get the new state.
- Distance is calculated physically: `|| Empirical(Output) - Empirical(ExpectedOutput) ||`.

### 3. Online Learning of Semantic Priors
All primitives (`OP_ADD`, `FOR_LOOP`, etc.) are initialized with zero semantic delta (`mu[S_DIM+DIM:] = 0`). 
When the planner evaluates a composition, it observes the *actual* state change: `delta = new_empirical_state - old_empirical_state`.
It then calls `observer.observe(delta)`, autonomously updating the Gaussian prior of that specific concept. The agent learns what `OP_ADD` does by watching what it does to the data.

## 4. Evaluation
1. **No Handcoded Priors**: The agent starts with zero semantic knowledge of its concepts.
2. **No Oracle Constraints**: The agent only receives `(input, expected_output)` pairs.
3. **Execution Verification**: A solution is only accepted if its actual executed output perfectly matches the expected output.

## 5. Implementation Steps
- [ ] **Step 1**: Implement `EmpiricalOracle` to compute statistical features from Python objects.
- [ ] **Step 2**: Update `PythonExecutor` with threading timeouts to handle infinite loops during search.
- [ ] **Step 3**: Implement `ExecutionGuidedAgent` that initializes blank priors and updates them via `observer.observe()` during search.
- [ ] **Step 4**: Implement the execution-guided Beam Search planner.
- [ ] **Step 5**: Run the experiment on a concrete curriculum and verify the AST output.