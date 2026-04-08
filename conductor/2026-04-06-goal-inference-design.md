# SP52: Experiment 28 — Goal Inference (The Semantic Bridge)

## 1. Objective
To validate the transition of the HFN architecture from a purely numerical synthesizer (requiring explicit 9D `goal_state` vectors) to an **Intent-Driven Reasoning Agent**. The system must bridge the gap between abstract natural language instructions (e.g., "Add one to all even numbers") and its rigorous, latent planning space using a specialized "Semantic Oracle" architecture.

## 2. Background & Motivation
In all previous experiments, the agent's target was a hard-coded 9D semantic state vector provided by a JSON curriculum file. While Experiment 27 (Curiosity) allowed the agent to autonomously map *its own* generated outputs to states, a true Artificial General Intelligence (AGI) must interact with human users. It must understand symbols (language) and translate them into its internal vectors.

As detailed in `hfn-future-directions.md`, Large Language Models (LLMs) are excellent at vast, flat, semantic reasoning, but terrible at rigorous, verifiable structural generation. HFNs are the opposite. This experiment proves the **LLM Oracle Integration**: The LLM acts purely as a semantic translator (Intent $\rightarrow$ Expected Output), while the HFN retains absolute sovereignty over the code synthesis (Expected Output $\rightarrow$ 9D State $\rightarrow$ Program Graph).

## 3. Setup & Environment

### Domain: Intent-to-Program Space
- **State Representation**: 9D semantic vector `[Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot]`.
- **The Knowledge Base**: The agent will boot up using the persistent `TieredForest` accumulated during the `curriculum`, `refactoring`, and `curiosity` experiments.

### The Semantic Bridge Architecture
Instead of forcing an LLM to understand our proprietary 9D vector math, we will leverage the `StateOracle` developed in Experiment 27.
1.  **The Prompt**: User provides a natural language intent (e.g., `"Double the numbers"`) and a sample input (`[1, 2, 3]`).
2.  **The LLM Oracle**: A lightweight, sandboxed semantic engine evaluates the intent against the input and determines the **Expected Output** (`[2, 4, 6]`). *(Note: For the scope of this specific algorithmic experiment, we will use a simulated/heuristic LLM Oracle to avoid external API dependencies, but the architectural boundary will be strictly identical to an API call).*
3.  **The State Oracle**: Converts the `(Input, Expected Output)` pair into the precise 9D `goal_state` vector.
4.  **The HFN Planner**: The agent retrieves the necessary templates, functions, and primitives from its knowledge base to synthesize a program that satisfies the vector.

## 4. The Intent Curriculum
The agent will be tested on a series of natural language prompts:
1.  `"Return the number 1"` (Tests primitive retrieval).
2.  `"Increment the input value"` (Tests basic operator synthesis).
3.  `"Map: add one to every item in the list"` (Tests loop construction).
4.  `"Filter: keep only the even numbers"` (Tests conditional logic).
5.  `"Complex: Double the even numbers, otherwise keep them the same"` (Tests non-linear procedural abstraction).

## 5. Evaluation Metrics
1.  **Semantic Translation Accuracy**: The ability of the pipeline to correctly convert the natural language string into a stable 9D target vector.
2.  **Zero-Shot Synthesis**: The ability of the HFN planner to synthesize the correct code for the translated vector *without* any manual curriculum definitions.
3.  **Knowledge Reuse**: The planner must leverage the `TieredForest` cold storage, utilizing the `TEMPLATE_MAP` and procedural chunks learned in previous sessions to solve the natural language tasks in $O(1)$ or $O(\log N)$ steps.

## 6. Failure Modes to Watch
- **Semantic Ambiguity**: The natural language prompt is underspecified (e.g., "Process the list"), leading to a faulty expected output from the Oracle. (Solution: Strict, clear prompts for the validation phase).
- **Latent Space Mismatch**: The `StateOracle` might generate a 9D vector that the HFN planner cannot mathematically reach because of missing prior rules. (Solution: Ensure the knowledge base from Exp 23-27 is properly loaded).

## 7. Implementation Steps
- [ ] **Step 1**: Implement the `LLMOracle` class (using a simulated dictionary mapping for the experiment to decouple from network API fragility, while preserving the exact input/output signatures of an LLM call).
- [ ] **Step 2**: Integrate the `StateOracle` from Exp 27 to form the complete `IntentToVector` pipeline.
- [ ] **Step 3**: Create the `IntentDrivenAgent` that marries the pipeline with the existing DFS `plan()` logic and `TieredForest` loading.
- [ ] **Step 4**: Run the agent against the 5 natural language tasks and verify the synthesized code executes correctly.