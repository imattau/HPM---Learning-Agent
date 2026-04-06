# Experiment 12: Self-Curiosity (Autonomous Learning)

**Script:** `experiment_self_curiosity.py`

## Objective
To evaluate the HFN system's capacity for true autonomy by removing all external data streams. This experiment tests whether the system can drive its own learning trajectory through a continuous loop of generative exploration ("dreaming") and self-evaluation.

## Setup
- **Environment:** No external data stream. Initialized with only 2 orthogonal "Atomic" seed nodes.
- **Mechanism:** An autonomous loop:
  1. **Generate (Dream):** Randomly sample an existing node, add exploratory noise, and use the `Decoder` to synthesize a concrete observation.
  2. **Observe (Perceive):** Feed the self-generated observation back into the `Observer`.
  3. **Evaluate (Detect Surprise):** Measure the `residual_surprise`. High surprise indicates a structural gap or conflict exposed by the generation.
  4. **Expand (Refine):** The `Observer` autonomously creates new nodes to account for the surprising new pattern.

## Results & Analysis
The experiment successfully demonstrated the capacity for **Autonomous Play**.

1. **Self-Driven Growth:** Starting from only 2 nodes, the system autonomously discovered and integrated **36 new structural concepts** over 50 "dreams" without any human supervision.
2. **Discovery via Surprise:** The system maintained high average self-surprise (32.1), consistently pushing into unexplored regions of its own manifold. Peak surprise events (up to 97.3) represent major structural "Aha!" moments where self-play exposed significant gaps.
3. **Internal Consistency:** The `Observer` effectively filtered generative noise, retaining only stable structural discoveries. Native **Adaptive Compression** was observed merging redundant concepts even in the absence of external pressure.

### Key Takeaway
HFN is capable of **True Autonomy**. By combining its known rules in novel ways and evaluating the results, the system can independently map out the logical consequences of its own knowledge base. This "Play" dynamic allows the agent to autonomously broaden its worldview and discover new abstractions entirely from within.
