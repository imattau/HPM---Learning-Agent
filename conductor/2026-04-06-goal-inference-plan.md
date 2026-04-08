# Implementation Plan: Goal Inference (The Semantic Bridge)

## Objective
Implement Experiment 28 to validate the transition from numerical `goal_state` vectors to **Intent-Driven Reasoning** using a simulated Semantic Oracle pipeline.

## Clarification on `QueryLLM` vs `LLMOracle`
While `QueryLLM` exists in the codebase (`hpm_fractal_node/nlp/query_llm.py`), it is specifically designed for NLP vocabulary expansion via Ollama (returning synonyms/related words). 
For this algorithmic experiment, we require a different interface: mapping an *Intent* string and an *Input* to an *Expected Output*. To avoid network fragility in the CI/CD pipeline, we will build a mocked `IntentOracle` directly in the experiment file that mimics the exact signature an LLM API would use.

## Implementation Steps

### Step 1: Implement the Semantic Oracles
- **Task**: Create `IntentOracle` that takes `(prompt: str, inp: Any)` and returns `expected_out: Any`. It will use a simple heuristic/dictionary mock to simulate LLM logic execution.
- **Task**: Reuse and adapt `StateOracle` from SP51 to take `(inp, expected_out)` and compute the 9D `goal_state`.
- **Conformity Check**: Verify the oracles correctly translate prompts like "Return the number 1" into the correct 9D vector.

### Step 2: Implement `IntentDrivenAgent`
- **Task**: Create an agent that encapsulates the `TieredForest`, `Observer`, and the new Oracle pipeline.
- **Task**: Add an `execute_intent(prompt, inp)` method that:
    1. Calls `IntentOracle` to get the expected output.
    2. Calls `StateOracle` to generate the 9D semantic target.
    3. Calls the standard DFS `plan()` to synthesize the program graph.
    4. Executes the generated code to verify it matches the Oracle's expected output.
- **Conformity Check**: Ensure the agent successfully loads the persistent knowledge base from SP49/SP50.

### Step 3: Run Curriculum and Evaluate
- **Task**: Run the agent through a suite of 5 natural language prompts ranging from simple constants to complex conditional maps.
- **Task**: Assert that the synthesized programs execute successfully and achieve the intended goals.
- **Conformity Check**: Confirm that higher-order tasks (like mapping) utilize the cached library functions (`TEMPLATE_MAP`, `increment`) rather than planning from scratch.

### Step 4: Final Documentation
- **Task**: Create `hpm_fractal_node/experiments/README_goal_inference.md`.
- **Task**: Update project-wide READMEs.
