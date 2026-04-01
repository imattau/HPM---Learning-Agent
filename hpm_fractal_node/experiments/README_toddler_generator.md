# Toddler Sentence Generator (SP21)

This experiment evaluates the HPM AI's capacity for **Top-Down Synthesis**. Unlike previous experiments that focused on bottom-up observation and clustering, SP21 tests if the system can use its distributed world models to *generate* semantically and grammatically correct simple sentences from high-level mental goals.

## Architecture: The Structural Octet

The system implements 8 distinct `TieredForest` domains that represent the cognitive building blocks of language:

1.  **Token Forest**: Maps to raw word strings ("mum", "eat", "apple").
2.  **Morphology Forest**: Handles tense and plurality (e.g., mapping "eat" + PAST to "ate").
3.  **Identity Forest**: Manages persistent entities (e.g., `Entity_Mum`, `Entity_Apple`).
4.  **Categorical Forest**: Hierarchical taxonomy (`[Person]`, `[Food]`, `[Animal]`).
5.  **Affordance Forest**: Action capabilities (e.g., `[Person]` can `[Eat]`).
6.  **Syntax Forest**: Grammatical slots (`[Subject]`, `[Verb]`, `[Object]`).
7.  **Relational Forest**: Cross-domain bindings between entities and tokens.
8.  **Narrative Forest**: Event templates (`[Agent] -> [Action] -> [Patient]`).

## Top-Down Synthesis Algorithm

Generation is performed through a constraint-resolution loop:
1.  **Retrieve Template**: Follows the `Narrative_Event` HFN to find required slots.
2.  **Validate Affordances**: Checks the `Affordance Forest` to ensure the proposed Agent can actually perform the Action.
3.  **Apply Syntax**: Determines if determiners (e.g., "the") are required based on Category.
4.  **Apply Morphology**: Modifies the base Action token based on the Goal tense (PAST/PRESENT).
5.  **Resolve Tokens**: Traverses the `Relational` path to extract final word strings.

## Key Insights

- **Semantic Guarding**: The experiment proves that HFN world models can prevent "hallucination" or nonsensical output. The system correctly identifies that "Apple ate mum" is an **Affordance Violation** and blocks the generation.
- **Cross-Forest Coordination**: Proves that a single "thought" can be realized by coordinating 8 independent, sovereign cognitive processes.
- **Functional Building Blocks**: Demonstrates that HFNs are not just for pattern recognition; they are functional data structures capable of supporting rule-bound reasoning.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_toddler_generator.py
```
