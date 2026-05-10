# Session State Checkpoint
Generated: 2026-05-08
Reason: Context threshold exceeded (95%)

## Execution Mode

**Mode**: interactive
**Auto-Continue**: false

## Current Task

Diagnose and fix ATIS benchmark B1 accuracy at 42.44% (target >60%).

## Progress Summary

- SNLP benchmark: all T1-T5 validated and passing
- PatternVariant: implemented (variant.py, store.py, pattern_manager.py)
- ATIS benchmark implemented at hpm_ai_v5/experiments/run_atis_benchmark.py
- StructuralNLPPolygraphGenerator switched in (skeleton + bigram views, 2 engines)
- Majority vote intent assignment implemented (_intent_votes dict)
- Checkpoint deleted and fresh retrain run: B1 still 42.44% (confirmed not a stale checkpoint issue)
- Only 1874 patterns from 4978 utterances (max_patterns=8192, so not saturation)
- B3/B4 still 0%: consolidation_distance=0.8 doesn't fire with near_threshold=1.5

## Root Cause Hypothesis

With StructuralNLPPolygraphGenerator (only 2 view engines: skeleton_view, skeleton_bigram_view),
patterns in those views match skeletal structure, NOT full token sequences. The skeleton of
"show flights from denver" and "show flights from chicago" are identical — so the SAME pattern
matches multiple intents, causing _intent_votes to be split between flight-related intents.

Previously with NLPPolygraphGenerator, token_view + canonical_view engines matched
specific token sequences → each pattern had a dominant intent. 

The skeleton views are too coarse for 18-class intent disambiguation.

## Fix Options

**Option A**: Restore token_view and canonical_view in StructuralNLPPolygraphGenerator
  - Keeps semantic views excluded (no view explosion)
  - Gives more discriminative views for intent voting
  - StructuralNLPPolygraphGenerator becomes: token + canonical + skeleton + bigram

**Option B**: Use NLPPolygraphGenerator but exclude semantic views (WordNet candidates)
  - Same as Option A effectively

**Option C**: Use the pattern_intent dict differently — instead of voting across all views,
  only use the primary engine (engine.last_match) for intent assignment, not view_matches
  - Simpler, may work if primary engine matches on canonical tokens

## Key Files

- `hpm_ai_v5/experiments/run_atis_benchmark.py` — _train_utterance, _predict_intent
- `hpm_ai_v5/polygraphs/nlp.py` — StructuralNLPPolygraphGenerator, NLPPolygraphGenerator
- `hpm_ai_v5/adapter/nlp.py` — NLPTokenizer, CanonicalPhraser, NamedEntityCanonicaliser

## Continuation Instructions

1. Read run_atis_benchmark.py and polygraphs/nlp.py to understand current state
2. Add token_view and canonical_view back to StructuralNLPPolygraphGenerator (Option A)
   - These give per-token discrimination without WordNet view explosion
   - Only exclude semantic_view_* (WordNet candidates)
3. Delete checkpoints and rerun: `rm -f checkpoints/atis_* && uv run python -m hpm_ai_v5.experiments.run_atis_benchmark 2>&1 | tail -15`
4. Target: B1 > 60%, B2 > 30%
5. After B1 fixed, proceed with polygraph view refactor plan at:
   docs/superpowers/plans/2026-05-08-polygraph-view-refactor.md
6. Then implement SCB benchmark per:
   docs/superpowers/plans/2026-05-07-scb.md
