# Implementation Plan: Wikipedia Character-Stream Simulation

**Goal:** Implement a Wikipedia character-stream simulation that trains an HPMAgent on raw text and exposes learned structure through a programmatic reasoning interface.

## File Structure

| File | Responsibility |
|---|---|
| `hpm_ai_v4/simulations/wikipedia_sim.py` | `WikipediaStream` class + `run_simulation` function |
| `hpm_ai_v4/simulations/text_reasoning.py` | `TextReasoningInterface` class |
| `hpm_ai_v4/simulations/data/get_corpus.py` | Download Simple English Wikipedia sample |
| `hpm_ai_v4/tests/test_wikipedia_sim.py` | All unit + integration tests |

## Tasks

### Phase 1: WikipediaStream + vocabulary
- Create `hpm_ai_v4/simulations/wikipedia_sim.py` with 96-char vocabulary (ASCII 32-126 + newline).
- Implement `WikipediaStream` for looping file I/O.

### Phase 2: TextReasoningInterface (encode/decode)
- Create `hpm_ai_v4/simulations/text_reasoning.py`.
- Implement `encode`/`decode` methods.

### Phase 3: Reasoning Layer Integration
- Implement `next_char_predict`, `word_complete`, `plan_to_boundary`, `counterfactual_shift`, and `explain_best_pattern` in `TextReasoningInterface`.

### Phase 4: Training Loop & Data
- Implement `run_simulation` in `wikipedia_sim.py`.
- Implement `download_corpus` in `hpm_ai_v4/simulations/data/get_corpus.py`.

### Phase 5: Verification
- Run full suite in `hpm_ai_v4/tests/test_wikipedia_sim.py`.
