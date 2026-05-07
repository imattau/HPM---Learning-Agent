# Structural NLP Benchmark (SNLP) Implementation Plan

## Objective
Implement the SNLP benchmark for HPM v5 to evaluate structural invariant learning in natural language. This tests delta induction, canonical filtering, skeleton recognition, slot-filling, and hierarchical meta-patterns.

## User Steering Acknowledgment
The user steered the direction towards a pure NLP structural benchmark (SNLP) to validate the "Structural Sequential Induction" theory discussed previously. I will execute this via the Task Conductor workflow, creating specific adapters and an experiment harness.

## Key Components

### 1. New Adapters (`hpm_ai_v5/adapter/nlp.py` or new file `hpm_ai_v5/adapter/nlp_structural.py`)
- **`SkeletonExtractor`**: Takes `pos_tags` or dependency parse labels from `NLPTokenizer` and creates a sequence of structural IDs (using `UnifiedVocabulary`).
- **`DeltaEncoder`**: Takes sequential skeleton states and computes a delta (either as a tuple `(state_n-1, state_n)` or numeric subtraction if mapped to continuous features) to explicitly test transition learning, or rely on HPM's internal template matching if applicable.

### 2. Dataset & Corpus Generation
- Generate a small synthetic corpus based on ATIS / GeoQuery styles.
- Include templates:
  - Base queries ("Show flights to Boston").
  - Word salad ("flights to Show Boston").
  - Synonym replacements ("Display flights to Boston").
  - Multi-sentence discourse ("What is the capital of Texas? How large is it?").

### 3. SNLP Benchmark Harness (`hpm_ai_v5/experiments/run_snlp_benchmark.py`)
- Define the 5 tasks (T1 to T5) with clear metrics.
- Set up an `HPMPipeline` for NLP with the new adapters.
- **T1 (Skeleton)**: Train on sentences, match on unseen phrases with same skeleton.
- **T2 (Delta Induction)**: Test engine's ability to predict the next state given a current state in the sequence.
- **T3 (Slot-filling)**: Use polygraphs + KB to fill slots with novel synonyms.
- **T4 (Word Salad)**: Pass anomalous phrasing; check if `res.action.confidence` drops.
- **T5 (Discourse)**: Use `PatternEngine.promote_to_meta` to learn paragraph-level structures.

## Implementation Steps

- [ ] **Step 1:** Create/Update Adapters. Extend `nlp.py` with `SkeletonExtractor` and `DeltaEncoder`. Update `NLPPolygraphGenerator` to include a `skeleton_view` and `delta_view`.
- [ ] **Step 2:** Create the Benchmark Harness. Write `run_snlp_benchmark.py` setting up the 5 evaluation tasks.
- [ ] **Step 3:** Implement Corpus Generator. Hardcode or use a simple generator for the training and test sets.
- [ ] **Step 4:** Run Benchmark and Validate Metrics. Ensure pass thresholds (>90% accuracy, >0.95 AUC, etc.) are met.
