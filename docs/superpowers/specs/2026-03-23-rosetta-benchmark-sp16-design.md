# Sub-project 16: Rosetta — Shared Understanding Across Divergent Math

## Goal
Design and implement a benchmark where two HPM agents with fundamentally different sensory substrates (L1/L2) must "explain" a concept (a **Square**) to each other. This tests the HPM framework's ability to achieve **Relational Shared Understanding** through Procrustes alignment of L3 Laws.

## Architecture

### 1. Divergent Substrates
- **Domain A (Euclidean)**: 
  - L2 Features: `[n_sides, length_1, length_2, length_3, length_4, angle_sum]`.
  - Concept "Square": `[4, 1, 1, 1, 1, 360]`.
- **Domain B (Coordinate)**:
  - L2 Features: `[x1, y1, x2, y2, x3, y3, x4, y4]`.
  - Concept "Square": `[0, 0, 1, 0, 1, 1, 0, 1]`.

### 2. The Rosetta Stone (Alignment)
- Both agents are given a small "Litmus" set of 5 shared examples (e.g., a Unit Square, a 2x2 Square, a Rectangle).
- Each agent computes its own L3 delta matrix ($M_A$, $M_B$) for the transformation "Scale by 2".
- We find the Procrustes rotation $R$ that aligns $M_A$ to $M_B$.

### 3. The Litmus Turn (Verification)
- Agent A defines a new concept: **"Identity"** (Transformation that preserves squareness).
- Agent A sends its L3 "Identity Law" to Agent B.
- Agent B applies the rotation $R$ to Agent A's law.
- **The Test**: Can Agent B use this "translated" law to distinguish a Square from a Triangle in its own Coordinate space?

## HPM cognitive stack roles
- **L1/L2**: Encode the divergent sensory data (Euclidean vs Coordinate).
- **L3**: Encodes the "Law of Squareness" as a relational operator.
- **Alignment**: The "Translator" that maps the relational geometry of Agent A onto Agent B.
- **L5 Monitor**: Detects "Translation Surprise" if the aligned law fails to correctly identify shapes.

## Success Criteria
- **Zero-Shot Transfer**: Agent B correctly identifies a Square using Agent A's law with >90% accuracy.
- **Structural Invariance**: The alignment succeeds even though the feature dimensions and meanings are completely different.
