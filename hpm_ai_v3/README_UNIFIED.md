# HPM AI v3: Unified Agnostic Discovery Architecture

This directory implements a unified, agnostic discovery architecture for HPM AI v3. It moves away from hardcoded domain knowledge and heuristic sequence logic toward an emergent learning model driven by HPM's core population dynamics and evaluator feedback.

## Core Components

### 1. Agnostic Discovery Agent (`agents/base_discovery.py`)
A base `ABC` class that defines the discovery lifecycle:
- **`PatternPopulation`**: Maintains a set of `UnifiedOrchestrator` patterns that compete and replicate.
- **Evaluator Integration**: All learning flows through `EvaluatorManager`. Orchestrator patterns are rewarded based on task performance (`Epistemic` and `Curiosity`).
- **Discovery Loop**: Iteratively selects tools and generates arguments based on learned policies and current context.

### 2. Python Substrate (`tools/python_substrate.py`)
Replaces domain-specific tools with a minimal, agnostic interface to the Python ecosystem:
- **`list_modules`**: Returns available modules (e.g., `numpy`, `sympy`, `spacy`).
- **`list_functions`**: Dynamically introspects a module to find callable operations.
- **`python_call`**: Executes a function with arbitrary arguments, allowing the agent to discover its utility through trial and reward.

### 3. Task-Specific Implementations
Subclasses implement the task-specific environment and evaluation while inheriting the agnostic discovery logic:
- **`MathDiscoveryAgent`** (`agents/active_discovery_agent.py`): Discovering hidden formulas via active environment querying (`evaluate_at`).
- **`PhysicsWordAgent`** (`agents/physics_word_agent.py`): Solving physics word problems by exploring NLP and physics libraries.

## HPM Alignment
- **Substrate Independence**: The agent is not pre-programmed with physics or math knowledge. It "discovers" `numpy.polyfit` or `spacy.tokenize` purely through evaluator feedback.
- **Hierarchical Learning**: Meta-orchestrator patterns learn to sequence low-level library calls, forming a hierarchy of discovery strategies.
- **Evaluator-Driven**: Replicator dynamics automatically prune poor strategies and amplify successful ones based on `Accuracy - Complexity + Coherence`.

## Running the Benchmark
To verify the architecture:
```bash
PYTHONPATH=. python3 hpm_ai_v3/task8/run_active_discovery.py
```
This runs the formula discovery benchmark using the unified `MathDiscoveryAgent`.
