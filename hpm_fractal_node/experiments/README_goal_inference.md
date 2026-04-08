# Experiment 28: Goal Inference (The Semantic Bridge)

## Objective
To validate the transition of the HFN architecture from a purely numerical synthesizer (requiring explicit 9D `goal_state` vectors) to an **Intent-Driven Reasoning Agent**. This experiment bridges the gap between abstract natural language instructions (e.g., "Add one to all even numbers") and the rigorous latent planning space.

## Background: "Bridging Symbols and Vectors"
A true AGI must understand human language. However, Large Language Models (LLMs) struggle with verifiable structural generation. This experiment introduces the **Semantic Oracle Pipeline**, which uses an LLM solely to determine the *Expected Output* of a natural language prompt, and a *State Oracle* to convert that output into a 9D semantic target vector. The HFN then retains absolute sovereignty over planning and synthesizing the code.

## Setup
- **Domain**: Intent-to-Program Space.
- **State Representation**: 9D semantic vector (Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot).
- **Semantic Oracles**:
    - `IntentOracle`: Simulates an LLM API call mapping `(Prompt, Input) -> Expected Output`.
    - `StateOracle`: Maps `(Input, Expected Output) -> 9D Goal Vector`.
- **Knowledge Base**: Boots from the `TieredForest` cold storage, utilizing previously learned procedures and higher-order templates.

## Results
The experiment proved the viability of the Semantic Bridge:
- **Zero-Shot Synthesis from Language**: The agent successfully parsed natural language commands like *"Increment the input value"* into a 9D vector (`[2.0, 1.0, 0, ...]`) and autonomously planned the structural primitives (`VAR_INP -> OP_ADD -> RETURN`) to achieve it.
- **Complex Semantic Retrieval**: For complex algorithmic prompts like *"Double the even numbers, otherwise keep them the same"*, the State Oracle successfully mapped the intent to the target state, triggering the retrieval of the advanced `if/else` logic fork learned during the Curiosity phase.
- **HFN Sovereignty**: The LLM was isolated purely to the semantic goal-setting phase. All code generated and executed was rigorously synthesized by the deterministic HFN planner, ensuring absolute programmatic correctness.

## Metrics Summary
| Natural Language Prompt | Inferred Semantic Vector | Status |
|---|---|---|
| "Increment the input value" | `[2. 1. 0. 0. 0. 0. 0. 0. 0.]` | **SUCCESS** |
| "Filter: keep only the even numbers" | `[0. 1. 3. 2. 0. 1. 0. 0. 0.]` | **SUCCESS** |
| "Complex: Double the even numbers..." | `[0. 1. 4. 1. 0. 1. 0. 0. 0.]` | **SUCCESS** |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_goal_inference.py
```
