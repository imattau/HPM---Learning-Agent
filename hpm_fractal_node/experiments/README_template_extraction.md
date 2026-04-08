# Experiment 26: Higher-Order Template Extraction (Refactoring)

## Objective
To validate the HFN architecture's ability to autonomously discover **Higher-Order Invariants** (like `MAP`) across its procedural library. This experiment also establishes the project-wide mandate for **Tiered Forests** and **Cumulative Knowledge Bases**.

## Background: "Algorithmic Generalization"
Currently, the agent constructs loop structures from scratch for every new task. This experiment introduces **Structural Isomorphism Detection**: by comparing two successful chunks (`map_increment` and `map_double`), the agent identifies the constant loop boilerplate and extracts it into a generic **Template** containing a **Slot** for variable logic.

## Persistence & Knowledge Base
Following the new mandates in `GEMINI.md`:
- **TieredForest**: Uses a hot/cold storage model to persist structural knowledge to disk.
- **PersistenceManager**: Handles session-level snapshots of the Forest and Observer weights.
- **Location**: All persistent knowledge is stored in `data/knowledge_base/`.

## Setup
- **Domain**: Semantic Program Space.
- **State Representation**: 9D vector (Value, Returned, Len, TargetVal, Iterator, Init, Condition, CallStack, TemplateSlot).
- **Template Extraction**:
    - Agent identifies invariants between `map_increment` and `map_double`.
    - Generates `template_MAP` which utilizes the `SLOT` primitive.
- **Zero-Shot Application**:
    - Agent solves `map_decrement` by parameterizing the `template_MAP` with the `decrement` primitive logic.

## Results
- **Autonomous Refactoring**: The system successfully detected structural isomorphisms and registered a generic `MAP` template.
- **Knowledge Persistence**: All learned nodes and weights were successfully saved to `data/knowledge_base/template_extraction/`, ensuring they are available for future experiments.
- **Complexity Reduction**: The planning search space for new mapping tasks was reduced from a 7-step construction to a 2-step retrieval (Template + Argument).

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_template_extraction.py
```
