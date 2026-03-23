# Sub-project 14: Linguistic Register Shift — The Social pH Benchmark

## Goal
Implement a linguistic benchmark that tests the HPM stack's ability to detect shifts in "Social Register" (Tone). Like the pH shift in Chem-Logic, this introduces a hidden environmental variable—**Tone (Formal vs. Informal)**—that reverses or alters the expected output for a given input.

## Architecture

### 1. Linguistic Substrate (`benchmarks/linguistic_sim.py`)
A simulated semantic environment that maps base verbs to their transformed versions across two registers.
- **Formal Register**: "Ask" -> "Inquire", "Help" -> "Assist", "Tell" -> "Inform".
- **Informal Register**: "Ask" -> "Hit up", "Help" -> "Back up", "Tell" -> "Fill in".
- **Task Generation**: Returns a sequence of 3 Formal training pairs followed by an Informal test "trap" pair.

### 2. HPM 5-Level Mapping
- **L1 (Syntax)**: Character frequency/distribution of the words.
- **L2 (Structural Anatomy)**: Semantic features of the root word (simulated embeddings).
- **L3 (Relational Law)**: The transformation delta (The "Register Rule").
- **L4 (Generative Head)**: Predicts the transformed word features given the root and previous observations.
- **L5 (Strategy/Monitor)**: Detects **Linguistic Surprise**. When the "Formal" intuition fails to predict the "Informal" outcome, L5 must flag high surprise ($S > 0.2$) and lower strategic confidence.

## Success Criteria
- **Surprise Detection**: L5 correctly identifies the "Register Shift" by spiking surprise on the Informal test task.
- **Robustness**: `l4l5_full` demonstrates better error recovery than `l4_only` by reverting to analytical L3 matching when the register shifts.
- **Domain Portability**: Validates that the same L5 logic used for Chemistry works identically for Language.
