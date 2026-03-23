# Sub-project 13: Chem-Logic II — Ambiguity & Competitive Reasoning

## Goal
Transform the Chem-Logic benchmark from a deterministic pattern matcher into a stress test for high-level scientific reasoning. By introducing regioselectivity, competitive inhibition, and latent environmental variables, we force the HPM stack to handle ambiguity, prioritize conflicting rules, and use L5 Surprise to discover unobserved variables.

## New Architectural Challenges

### 1. Competitive Inhibition (Priority Ranking)
- **Mechanism**: Reactants with multiple reactive sites (e.g., -OH and -NH2) and limited reagents.
- **HPM Logic**: L3 must learn not just "rules" but a **Priority Law**. If the agent applies the -OH rule when the -NH2 rule was higher priority, the resulting NLL spike forces L5 to re-evaluate the ranking.
- **Simulator Update**: Implement `react_priority(molecule, reagent)` using nucleophilicity scales.

### 2. Latent Environmental Variables (The "Hidden" Context)
- **Mechanism**: Reaction outcomes change based on unobserved `pH` or `Temperature`.
- **HPM Logic**: Two identical L1/L2 inputs produce different L3 deltas. This creates **Maximum Surprise** ($S > 0.8$) in L5.
- **Success Condition**: L5 must trigger an "Environment Search" (simulated) or shift to a multi-modal pattern distribution to account for the divergence.

### 3. Regioselectivity (Neighborhood Effects)
- **Mechanism**: Positional directs (ortho/meta/para) in aromatic substitution.
- **HPM Logic**: L2 must expand its "Sensory Field" to include the electronic influence of neighbors. We will update the L2 Encoder to use RDKit's `GetDistanceMatrix` or `GasteigerCharges` to encode the "reactivity landscape".

### 4. Stereoselectivity (3D Geometry)
- **Mechanism**: Cis/Trans or R/S isomerism.
- **HPM Logic**: Move L1 from flat SMILES to 3D-aware features (e.g., using RDKit's 3D coordinate generation and encoding distances).

## Success Criteria
- **L4/L5 Accuracy**: `l4l5_full` should outperform `l4_only` by correctly rejecting "intuitive but low-priority" outcomes.
- **Surprise Response**: L5 correctly identifies latent variable divergence (pH shift) by flagging high surprise ($S$) when L3 analytical matching fails.
- **Ranking Stability**: The agent stabilizes a "Nucleophilicity Rank" in its pattern store that predicts which group will react first.

## Phase 1 Implementation: Competition & Latency
We will start by implementing **Competitive Inhibition** and **Latent pH** as these provide the most immediate stress-test for the HPM evaluators.
