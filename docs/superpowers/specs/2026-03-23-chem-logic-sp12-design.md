# Sub-project 12: Chem-Logic — The Molecular Hidden Law Benchmark

## Goal
Design and implement a "Hidden Law" benchmark where agents must infer chemical reaction rules (e.g., Oxidation, Esterification) from reactant-product pairs. This benchmark tests the grounding of abstract transformations in a specialized molecular substrate (simulating RDKit).

## Architecture

### 1. Molecular Substrate (`benchmarks/chem_logic_sim.py`)
Since RDKit is a heavy dependency, we implement a `ChemLogicSimulator` that:
- Defines a set of "Molecules" using mock SMILES strings.
- Implements `get_functional_groups(smiles)`: Returns a vector of presence/absence for groups like Hydroxyl (-OH), Carboxyl (-COOH), Aldehyde (-CHO), etc.
- Implements `apply_reaction(reactant, reaction_type)`: Generates the correct product.
- Implements `is_valid(molecule)`: Simulates a valence/stability checker (L5 feedback).

### 2. HPM 5-Level Mapping
- **L1 (Syntax)**: SMILES token distribution. Encodes the raw character sequence of the molecule.
- **L2 (Structural Anatomy)**: Functional Group vector. Identifies the "organs" of the molecule.
- **L3 (Relational Law)**: The transformation delta (e.g., `-OH` becomes `=O`). This is the "Hidden Law".
- **L4 (Generative Head)**: Predicting the L2 features of the product given the reactant.
- **L5 (Strategy/Monitor)**: Monitoring for "Chemical Surprise". If L4 predicts a product that violates simulated valence rules, L5 lowers strategic confidence.

### 3. Reaction Rules
The benchmark will include rules such as:
- **Oxidation**: Alcohol -> Aldehyde / Ketone.
- **Reduction**: Ketone -> Alcohol.
- **Esterification**: Alcohol + Acid -> Ester.
- **Hydrolysis**: Ester -> Alcohol + Acid.

## Success Criteria
- The agent identifies the correct product candidate by matching the hidden L3 transformation.
- `l4l5_full` demonstrates robustness by rejecting "invalid" molecular predictions (simulated L5 valence check).
- Integrated into the standard `pytest` suite with a `--smoke` flag.
