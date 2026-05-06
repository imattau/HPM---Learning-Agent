# Code Recognition Benchmark Plan

## Objective
Test the HPM core’s ability to recognise, abstract, and reuse structural patterns in source code, independent of surface syntax. This evaluates pattern discovery, canonicalization, hierarchical structure learning, and cross‑language transfer.

## 1. Adapters (`hpm_ai_v5/adapter/code.py`)
Implement the following adapters:
- **`CodeTokenizer`**: Convert raw code string to a sequence of token types/IDs (using Python's `tokenize` or `ast`).
- **`ASTFlattener`**: Linearise the abstract syntax tree into a structured sequence `(node_type, child1, ...)`.
- **`CFGExtractor`**: Produce a control-flow graph as edge pairs `(from_id, to_id)`.
- **`CanonicalRenamer`**: Normalise user-defined names to placeholders (e.g., `VAR_1`, `FUNC_A`) to ensure surface syntax variations map to the same underlying pattern.

## 2. Polygraphs (`hpm_ai_v5/polygraphs/code.py`)
- **`CodePolygraphGenerator`**: Generates multiple views (Token View, AST View, CFG View, Canonical View) from a single raw code string, feeding them into the HPM core.

## 3. Benchmark Tasks (`hpm_ai_v5/planning/code_recognition.py`)
Implement `CodeRecognitionBenchmark` with the following tasks:

### 3.1. Code Clone Detection (Zero-shot)
- **Setup:** Given 20 pairs of code snippets (clones and non-clones).
- **Task:** Check if the Pattern Engine matches the same `Pattern` or `PatternSequence` for both clones.
- **Metric:** Distance between canonicalised pattern vectors.
- **Success:** True positive rate > 90%.

### 3.2. Idiom Discovery (100 files)
- **Setup:** Process a corpus of Python files containing common idioms (`with open`, list comprehensions).
- **Task:** Verify the engine's `PatternManager` and sequence library discover these idioms organically.
- **Metric:** Number of discovered idioms matching ground truth.
- **Success:** Discover $\ge 4/5$ idioms with support > 2.

### 3.3. Edit Pattern Transfer (Program Repair)
- **Setup:** Train on 10 examples of fixing a specific bug (e.g., null check) in Java/Python. Test on 5 similar bugs in C#/another language.
- **Task:** The core must predict the correct edit structure (delta).
- **Metric:** Exact match of the transformation structure.
- **Success:** > 70% accuracy on the test set.

### 3.4. Code Completion (1000 lines train)
- **Setup:** Provide a sequence of tokens/AST nodes.
- **Task:** Predict the next structural token.
- **Metric:** Top-1 accuracy compared to an n-gram baseline (n=3).
- **Success:** Outperform baseline by > 10% absolute.

## 4. Execution
The tasks will be integrated into the existing harness (`hpm_ai_v5/experiments/run_code_benchmark.py`). All testing will rely purely on the adapter transformations feeding into the unchanged `PatternEngine` and `PatternManager`.