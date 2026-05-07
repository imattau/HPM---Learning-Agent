# Structural Creativity Benchmark (SCB) Design

**Date**: 2026-05-07
**Branch**: hpm-ai-v5
**Status**: Approved for implementation

---

## Overview

The SCB tests whether HPM v5 can generate novel, syntactically valid Python function skeletons by extending learned `U_*` structural sequences. It uses only existing v5 machinery — no new pattern levels, no execution required.

Extension path: once PatternVariant consolidation (ATIS plan) is complete, L2 can be added using variant centroids as recombination seeds.

---

## Corpus and Training

**Corpus**: ~200 short Python functions extracted from stdlib (`os`, `io`, `pathlib`). Each function is processed through:

```
NLPTokenizer → NL2CodeBridgeAdapter → PatternEngine
```

`NL2CodeBridgeAdapter` maps NL/keyword tokens to `U_*` structural IDs (`U_IF`, `U_WHILE`, `U_TRY`, `U_FOR`, `U_ASSIGN`, `U_RETURN`, `U_CALL`, `U_THROW`). The engine learns `Pattern` and `PatternSequence` objects over these `U_*` state tuples.

**Polygraph**: `CodePolygraphGenerator` runs in parallel, producing `ast_types` and `skeleton` views from the raw Python source. This gives the engine both semantic (`U_*`) and structural (AST) pattern views during training.

**Split**: 80% train / 20% held-out (stratified by dominant `U_*` token).

---

## Generation Mechanism

For each held-out function:
1. Extract the first 1-2 `U_*` tokens as the **seed**
2. Call `PatternSequence.simulate()` from the seed state
3. The engine extends the sequence using learned transitions
4. Pass the generated `U_*` sequence to `UCodeRenderer`
5. Evaluate the rendered Python skeleton

This tests genuine generalisation: the seed comes from the held-out set, not artificial inputs.

---

## UCodeRenderer (new postprocessor)

Maps a `U_*` sequence to a syntactically valid Python skeleton. Implements the `Postprocessor` protocol.

| U_* token | Python skeleton |
|-----------|----------------|
| `U_IF` | `if condition:\n    pass` |
| `U_WHILE` | `while condition:\n    pass` |
| `U_FOR` | `for item in iterable:\n    pass` |
| `U_TRY` | `try:\n    pass\nexcept Exception:\n    pass` |
| `U_ASSIGN` | `result = value` |
| `U_RETURN` | `return result` |
| `U_CALL` | `func()` |
| `U_THROW` | `raise Exception()` |

Output is wrapped in `def generated_fn():\n` and validated with `ast.parse()`. Invalid output counts as a correctness failure.

**File**: `hpm_ai_v5/postprocessors/code.py`

---

## Metrics

| Metric | Definition | Pass threshold |
|--------|------------|----------------|
| **Syntactic correctness** | `ast.parse()` succeeds on rendered output | >90% |
| **Structural novelty** | Generated `U_*` sequence not identical to any training sequence | >50% of outputs |
| **Recombination** | Generated sequence uses transitions from ≥2 distinct training patterns | ≥60% of outputs |

**Baseline**: random `U_*` sequence of the same length as the target. Provides a floor — expected near-0% correctness and high novelty.

---

## Benchmark Structure

```
Train: 160 functions → PatternEngine learns U_* patterns + sequences
Test:  40 held-out functions → seed = first 1-2 U_* tokens
       For each: simulate → render → evaluate
       Report: correctness, novelty, recombination across 40 outputs
Baseline: random U_* sequences of matching lengths, same 3 metrics
```

No unit tests, no execution. AST validity is the sole correctness gate.

---

## Files

| File | Action |
|------|--------|
| `hpm_ai_v5/postprocessors/code.py` | **Create** — `UCodeRenderer` postprocessor |
| `hpm_ai_v5/experiments/run_scb_benchmark.py` | **Create** — SCB harness |
| `hpm_ai_v5/experiments/corpus/extract_stdlib.py` | **Create** — extract 200 functions from stdlib |
| `tests/test_ucode_renderer.py` | **Create** — unit tests for UCodeRenderer |

Existing files used without modification:
- `hpm_ai_v5/adapter/nlp.py` — `NL2CodeBridgeAdapter`
- `hpm_ai_v5/polygraphs/code.py` — `CodePolygraphGenerator`
- `hpm_ai_v5/core/sequence.py` — `PatternSequence.simulate()`

---

## Extension Path to L2

Once PatternVariant consolidation is complete (ATIS plan), L2 can be added:
- Use variant centroid interpolation between two learned `U_*` patterns as seed
- This tests recombination of distinct structural abstractions
- No new benchmark infrastructure needed — extend `run_scb_benchmark.py`

---

## Out of Scope

- Code execution or unit test running
- LLM baseline comparisons
- AST tree-edit distance metric (novelty by sequence identity is sufficient for L1)
- Open-ended natural language prompts (L3)
- New pattern levels beyond existing `Pattern` and `PatternSequence`
