# Plan: SP‑Math1 – MathAgent – Symbolic Mathematics & Equation Solving

**Objective:** Integrate a **MathAgent** that can parse, represent, and solve symbolic mathematical expressions using HFN trees. The agent will learn transformation rules (e.g., the power rule for differentiation) from a few examples by composing deterministic mathematical primitives.

**Strategic Intent:** Demonstrate that symbolic logic and formal rule-following can be natively emergent in HPM through macro composition over deterministic functional primitives (using `sympy` as the underlying engine).

## Changes

### 1. Math Domain Configuration & Oracle (`hpm_ai_v2/domains/math_domain.py` & `hpm_ai_v2/utils/math_oracle.py`)
- [ ] Create `MathDomainConfig(DomainConfig)`:
    - [ ] Define concepts for operators (`ADD`, `MUL`, `POW`, etc.), functions (`SIN`, `LOG`), variables (`X`, `Y`), and transformations (`DERIVATIVE`, `SOLVE`).
    - [ ] Implement `encode_expression(expr_str_or_sympy)`: Generates an HFN state vector (`mu`) based on structural features (term count, tree depth, and a structural hash).
- [ ] Create `MathOracle(Oracle)`:
    - [ ] Uses `sympy` to verify equality between generated expressions and expected results.
    - [ ] Computes structural state vectors for verification and retrieval.

### 2. Math Agent & Primitives (`hpm_ai_v2/agents/math_agent.py`)
- [ ] Create `MathAgent(BaseHFNAgent)`:
    - [ ] Implement L1 Primitives: `BUILD_VAR`, `BUILD_CONST`, `APPLY_OP`, `DIFFERENTIATE`, `INTEGRATE`, `SIMPLIFY`, `SOLVE`.
    - [ ] Each primitive is wrapped as a callable that takes HFN nodes (trees) and returns a new HFN node (result tree).
    - [ ] Use `self.add_strategy("_try_bfs", ...)` to enable rule discovery.

### 3. Math Renderer (`hpm_ai_v2/domains/math_renderer.py`)
- [ ] Create `MathRenderer(Renderer)`:
    - [ ] Recursively traverses an HFN tree with `relation_type="math_expr"`.
    - [ ] Converts the tree structure back into a standard mathematical string or LaTeX (e.g., `x**2 + sin(x)`).

### 4. Agent Integrations
- [ ] Update `ReaderAgent` (`hpm_ai_v2/agents/reader_agent.py`):
    - [ ] Add `extract_math_expressions(text)`: Heuristic or primitive-based extraction of math strings from text, converting them to HFN trees via `MathAgent`.
- [ ] Update `WriterAgent` (`hpm_ai_v2/agents/writer_agent.py`):
    - [ ] Update `answer_natural`: If the answer phrase is a `math_expr` node, use `MathRenderer` and include the formatted expression in the response.

### 5. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_math1_power_rule.py`)
- [ ] **Phase 1 - Discovery:**
    - Provide training example: Input `x**3`, Target `3*x**2`.
    - Agent runs BFS (depth 3-4) to find the macro: `DIFFERENTIATE(POW(X, 3))`.
    - *Correction:* The macro should ideally represent the rule `d/dx x^n = n*x^(n-1)`, but for SP-Math1, we focus on the agent finding the path to the correct symbolic transformation.
- [ ] **Phase 2 - Generalization:**
    - Test on `x**5`.
    - Verify the agent applies the learned macro/strategy to produce `5*x**4`.
- [ ] **Phase 3 - Persistence:**
    - Verify that the `MathDomainConfig` is saved to the forest and re-loaded correctly on next run.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_math1_power_rule.py`.
- [ ] Confirm 100% accuracy on polynomial differentiation after 1 example.
- [ ] Confirm HFN trees for math expressions are correctly structured and rendered.
