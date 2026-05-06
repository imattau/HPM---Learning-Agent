# Symbolic Pattern Matching Benchmark Design (SP54+)

## Overview
The **Symbolic Pattern Matching Benchmark** tests the HPM v5 core's ability to recognize structural invariants in natural language and map them to specific tool invocation patterns. This benchmark validates that the hierarchical, delta-based learning engine can be used for robust Natural Language Understanding (NLU) by treating symbols as structural deltas.

## Lessons Learnt from Development

### 1. State Pinning for "Absolute" Recognition
HPM v5 is natively a relative/delta-based engine. In natural language, tokens like "Paris" or "Tokyo" must be recognized as symbolic invariants regardless of their sentence position.
- **Design Resolution:** Before processing each token in a symbolic sequence, the engine state is reset to a fixed `START` state. This forces the engine to treat the symbol itself as the primary delta, effectively converting the pattern store into a high-dimensional associative memory for symbolic recognition.

### 2. Multi-View "Skeleton" Polygraphs
Recognition is most robust when surface noise is removed.
- **Design Resolution:** The `NLPPolygraphGenerator` generates a **Skeleton View** that filters out punctuation and common particles. This view proved to be the most "structurally invariant," allowing the engine to focus on the high-signal relationship between verbs (actions) and parameters (objects).

### 3. Deterministic "Unified" Vocabulary
Python's default hashing is randomized per process, which is catastrophic for a pattern store that relies on stable numeric coordinates.
- **Design Resolution:** All symbolic tokens MUST be passed through a `UnifiedVocabulary` or a stable hashing function (like MD5-based numeric mapping). This ensures that a "word" always maps to the same "coordinate" in the engine's latent space across training epochs and test sessions.

### 4. Utility-Driven Intent Binding
In HPM, "utility" is typically a measure of predictive success. In NLU, utility is the bridge to **Intent**.
- **Design Resolution:** By applying rewards to patterns that appear during specific tool-labeled training examples, we bind structural invariants to functional intents. The engine learns that certain sequences are "useful" for specific outcomes (like `get_weather`), allowing it to perform intent classification via pattern utility.

### 5. Identification over Prediction
While HPM is built for forecasting, its **Identification** capability (mapping observations to the `last_match`) is the primary driver for symbolic tasks.
- **Design Resolution:** Confidence in a symbolic match is derived from the inverse distance of the `last_match` in the pattern store. A "vote-based" heuristic across a token sequence allows for high-accuracy tool recognition even when individual tokens are novel.

## Benchmark Metrics
- **Tool Accuracy:** Percentage of test queries correctly mapped to the target tool schema.
- **Parameter F1:** Accuracy of identifying variable placeholders within the sequence.
- **Distractor Rejection:** Ability to maintain low confidence (or "novel" status) for out-of-domain queries.
