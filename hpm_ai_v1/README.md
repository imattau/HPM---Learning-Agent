# HPM AI — Recursive Self-Refactoring Engine

The `hpm_ai_v1` package implements a recursive, self-evolving engine built on the Hierarchical Pattern Modelling (HPM) framework. It transitions the system from **Stochastic Patching** (guessing code changes) to **Manifold Alignment** (modelling the relational logic of the code).

## Core Philosophy: The "Axiom Forge"

In the HPM theory (§3.5), intelligence is the discovery of relational invariants that persist across different substrates. HPM AI treats **Source Code** as its primary substrate. By analyzing its own implementation, mining external patterns from Math and PyPI, and performing structural synthesis, the AI aims to discover increasingly elegant and efficient "Laws of Computation."

## Architecture

### 1. Middle-Manifold Representation (MMR)
HPM AI v2 decouples "Truth" from "Syntax" using an abstract representation layer.
- **Transpilation**: Converts Python Abstract Syntax Trees (AST) into relational graphs.
- **Semantic Invention**: The agent modifies the *logic graph* rather than the text string, ensuring that every generation is syntactically valid and structurally grounded.

### 2. Relational Synthesis (Generative Recombination)
Instead of hardcoded mutations, the system uses a **Generative Recombination Head**.
- **Crossover**: Merges the target function's MMR graph with successful "Donor Patterns" (Relational Laws) mined from other agents or external substrates.
- **Diversity**: Allows the framework to "forge" novel logic by combining disparate functional building blocks.

### 3. Metacognitive Gating (The L5 Compiler)
The **L5 Meta-Monitor** acts as the final judge of code quality, enforcing the **HPM Elegance Principle** (Minimal Description Length).
- **Pareto Check**: Rejects any mutation that increases node count unless it reduces execution time by >15%.
- **Bloat Window**: Detects evolutionary stagnation and temporarily allows complexity increases (<20%) if the Novelty Score (Surprise $S$) is high, enabling the AI to "tunnel" through local optima to reach superior algorithmic structures.

### 4. Cross-Modal Logic Bridge (SP19)
The "Rosetta Refactor" proves that the AI is discovering **Universal Invariants**, not just Python-specific tricks.
- **Transfer**: A symmetry law discovered while refactoring code is exported as an HPM vector.
- **Validation**: An ARC agent pulls that same vector to solve a visual grid puzzle, proving the law's domain-agnostic nature.

## Directory Structure

| Module | Function |
|---|---|
| `core/` | SuccessionController and the Bloat-aware L5 Judge. |
| `transpiler/` | MMR graph engine and AST-to-Relational mapping. |
| `substrates/` | `LocalCodeSubstrate` for codebase ingestion. |
| `sandbox/` | Safe, isolated environment for verifying new generations. |
| `store/` | Concurrent SQLite persistence with lineage/parentage tracking. |

## Evidence of Self-Improvement

In initial recursive runs, HPM AI demonstrated the following Pareto gains:
- **Elegance**: Collapsed a 317-node evaluator into an 80-node version (**~75% reduction**).
- **Performance**: Achieved a **3.1% speedup** while simultaneously reducing complexity.
- **Robustness**: 100% test compliance across the 800+ core HPM test suite.

## Usage

To start the recursive self-refactoring loop on a target file:

```bash
# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:.

# Run the Succession Controller
python hpm_ai_v1/main.py --target_file <path_to_file>
```

*(Note: The L5 Compiler requires all existing tests to pass before it will accept a new generation as a baseline.)*

## Future Objectives (HPM AI v3)
- **Multi-Language Transpilation**: Moving from Python MMR to C++/Rust targets.
- **Autonomous Lab**: Integrated sandbox for hardware-aware performance profiling.
- **Open-Ended Prior Mining**: Continuous background harvesting of the entire PyPI ecosystem to build a "Global Relational Library."
