# Thematic Anchor Extraction Experiment (SP20)

This experiment evaluates the HPM AI's capacity for **Structural Summarization** using a two-tier, multi-process cognitive architecture. It tests the discovery of recurring "Thematic Anchors" from a continuous natural language stream (*Peter Rabbit*).

## Architecture: The Hierarchical Bottleneck

The system organizes 4 processes into two functional tiers:

1.  **Level 1 (L1) Lexical Tier**:
    *   **Worker**: `L1_Lexical`
    *   **Role**: Maps 4-slot context windows into word-identities based on the NLP world model.
    *   **Output**: Passes the predictive $\mu$ of the winning word-concept up to the Governor.
2.  **Level 2 (L2) Relational Tier**:
    *   **Worker**: `L2_Thematic`
    *   **Role**: Ingests a rolling window of L1 identities ($W=2$).
    *   **Dynamics**: Uses high-plasticity HFN creation to stabilize recurring sequences as higher-order "Anchor Nodes."
3.  **Governor**: Manages the message passing, applies a slow weight decay to simulate attention, and extracts the top anchors.

## Theoretical Insight: Negative Selection via Priors

A major challenge in natural language processing is the dominance of "Function Words" (the, and, at). To extract *meaningful* anchors without hardcoding a stop-word list, the experiment uses **HFN Negative Selection**:

*   The L2 worker is seeded with **Background Prior HFNs** representing common function word sequences.
*   Because these sequences are already "explained" by the priors, the Observer's `residual_surprise` remains low, and **no new leaf nodes are created** for grammatical noise.
*   Only novel or semantically heavy sequences (e.g., "ran the", "on barked") trigger high residual surprise, leading to the creation of **Thematic Anchor** nodes.

## Key Findings (2000 Tokens)

- **Structural Compression**: The system successfully compressed 2000 tokens into a handful of dominant L2 nodes.
- **Narrative Motif Discovery**: The top anchors began to capture narrative transitions (e.g., `ran the`, `walked is`, `on barked`).
- **Weight-Driven Saliency**: The use of global weight decay ensured that anchors representing persistent themes maintained higher saliency than one-off occurrences.
- **Fractal Uniformity**: The L2 synthesizer used the exact same `Observer` and `HFN` logic as the L1 specialists, proving the scale-free nature of the architecture.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_thematic_anchor.py
```
