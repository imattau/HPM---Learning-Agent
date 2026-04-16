# [SP94] One-Shot Reaction Prediction — Ester Hydrolysis

## Objective
Demonstrate HPM's one-shot learning capability in the domain of chemical informatics by discovering a reaction transformation from a single molecular fingerprint pair and generalizing it to novel reactants.

## Chemical Domain
- **Reaction**: Ester Hydrolysis (Methyl Acetate + $H_2O$ → Acetic Acid + Methanol).
- **Representation**: 8-bit molecular fingerprints representing key functional groups (methyl, ethyl, ester bond, carboxyl, hydroxyl, etc.).
- **Primitives**: `SET_BIT_i` and `CLEAR_BIT_i` operations that simulate bond breaking and formation at the bit level.

## Experimental Phases

### 1. One-Shot Discovery (Methyl Acetate)
- **Goal**: Learn the bit-level transformation for ester hydrolysis from a single $(input, output)$ pair.
- **Process**: The agent uses BFS search to find a sequence of `SET_BIT` and `CLEAR_BIT` operations that transforms the methyl acetate fingerprint into the products (acetic acid + methanol).
- **Outcome**: Successfully discovered and registered a reusable **`hydrolysis`** macro.

### 2. Generalization (Ethyl Acetate)
- **Goal**: Apply the learned `hydrolysis` macro to a novel molecule: ethyl acetate.
- **Process**: The agent attempts to solve the task for ethyl acetate.
- **Outcome**: The agent successfully reused the `hydrolysis` macro at Depth 1 (Zero-Shot), proving that HPM has captured the **functional group transformation** independently of the specific hydrocarbon chain.

## Key HPM Mechanisms

### Bit-Level Abstraction
By representing molecules as bit fingerprints, HPM can treat chemical reactions as discrete bit-flipping programs. This allows it to leverage its powerful program synthesis engine to "discover" chemistry from data.

### One-Shot Structural Transfer
The core breakthrough of SP94 is the ability to extract a **multi-step procedural macro** from a single observation and successfully apply it to a new input. This mirrors how human chemists learn a reaction mechanism once and then apply it to an entire class of molecules.

## Running the Experiment
```bash
PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp94_reaction_prediction.py
```
