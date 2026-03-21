# ARC Benchmark Design Specification

**Date:** 2026-03-21
**Status:** Draft v1

---

## Overview

A standalone benchmark script (`benchmarks/arc_benchmark.py`) that evaluates the HPM agent on the Abstraction and Reasoning Corpus (ARC). ARC measures human-like general intelligence via visual logic puzzles requiring rule abstraction from a small number of examples.

The benchmark uses a **5-way discrimination protocol**: after training on task examples, the agent must identify the correct output from 4 distractors drawn from other ARC tasks. This is easier than ARC's standard exact-generation scoring but provides a principled, HPM-compatible evaluation.

**Chance baseline:** 20% (1 of 5). Published GPT-4 exact-generation scores: ~0–5% (different protocol; not directly comparable).

---

## 1. Data

**Source:** HuggingFace `datasets` library.

```python
from datasets import load_dataset
ds = load_dataset("fchollet/arc-agi", split="train")
```

400 training tasks. Each task contains:
- `train`: list of 3–5 `{input, output}` pairs (grids of integers 0–9)
- `test`: list of 1 `{input, output}` pairs (output known in training split)

**New dependency:** `datasets>=2.0` (add to `requirements.txt` and `pyproject.toml`).

---

## 2. Grid Encoding

**Max grid size:** 10×10. Tasks where any grid (train or test, input or output) exceeds 10 rows or 10 columns are **excluded**. Expected: ~15% exclusions (~60 tasks), ~340 tasks evaluated.

**Encoding procedure:**
1. Flatten grid to 1D (row-major)
2. Pad with 0 to length 100 (10×10)
3. Divide by 9 → float values in [0, 1]
4. Result: vector of dim 100

**Observation vector:** concatenate `encode(input)` and `encode(output)` → dim 200.

```python
feature_dim = 200
```

---

## 3. Agent Configuration

A **fresh agent is created for each task** — no cross-task state. Per-task config:

```python
AgentConfig(
    feature_dim=200,
    beta_comp=0.0,      # disabled: O(d³) compression too expensive at d=200
    T_recomb=5,
    recomb_cooldown=3,
    init_sigma=2.0,     # diffuse start for 200-dim space
    agent_id="arc_bench",
)
```

`beta_comp=0.0` ensures the compression eigenvalue computation is never triggered, keeping per-step cost manageable at high feature dimension.

---

## 4. Training Phase

For each training pair `(input_grid, output_grid)` in the task's `train` set:

1. Encode: `obs = concatenate(encode(input_grid), encode(output_grid))` → shape (200,)
2. Call `agent.step(obs)` **10 times** (repeated exposure to the same example)

Total training steps per task: 30–50 (3–5 pairs × 10 reps).

---

## 5. Evaluation Phase

### Scoring function

For a candidate vector `v`, the agent's ensemble score is:

```python
score(v) = sum(w_i * pattern_i.log_prob(v) for pattern_i, w_i in agent.store.query(agent.agent_id))
```

**Sign convention:** `GaussianPattern.log_prob(x)` returns the **negative log-likelihood** (NLL), i.e. `-logpdf(x)`. Lower score = more probable under the ensemble. The correct candidate wins if it has the **lowest** score (not highest).

**Empty-store edge case:** If all patterns are pruned and the store is empty, `score()` returns `0.0` for all candidates. This is treated as a tie — the correct candidate loses (conservative tie-breaking: correct must strictly have the lowest score).

### Distractor selection

For task at index `i`, sample 4 distractor tasks using `rng = np.random.default_rng(i)` (deterministic per task). Sample from all task indices **excluding `i`** (to avoid using the task's own output as a distractor). For each distractor task, use the **first training pair's output grid**, encoded and padded to 10×10.

### Candidate set (5 candidates)

All 5 candidates use the **same encoded test input**; only the output slot varies:

| Candidate | Content |
|-----------|---------|
| Correct | `(test_input, correct_test_output)` |
| Distractor 1–4 | `(test_input, output from distractor task j)` |

### Decision rule

The agent answers correctly if `score(correct_candidate) < score(all_distractor_candidates)` (lowest NLL = most probable). Ties broken conservatively: correct must strictly have the lowest score.

---

## 6. Metrics

| Metric | Definition | Chance |
|--------|-----------|--------|
| **Accuracy** | % tasks where correct output ranked #1 | 20.0% |
| **Mean Rank** | Mean rank of correct output (1=best, 5=worst) | 3.0 |
| **vs Chance** | Accuracy − 20.0% | 0.0% |

Tasks excluded due to grid size > 10×10 are reported separately and not counted in accuracy.

### Output table

```
ARC Benchmark (5-way discrimination, HuggingFace fchollet/arc-agi)
────────────────────────────────────────────────────────────────────────
Tasks Run   Excluded   Correct   Accuracy   Mean Rank   vs Chance
340         60         85        25.0%      2.8         +5.0%
```

---

## 7. File Location

```
benchmarks/arc_benchmark.py     # new standalone script
```

Follows the same pattern as existing benchmarks: imports from `benchmarks.common`, runnable with `python benchmarks/arc_benchmark.py`.

---

## 8. Dependencies

New:
```
datasets>=2.0    # HuggingFace datasets for ARC data loading
```

Existing (already in requirements):
- `numpy`
- `hpm.agents.agent.Agent`
- `hpm.config.AgentConfig`
- `benchmarks.common.print_results_table`

---

## 9. Comparability Note

Published ARC results (GPT-4: ~0–5%, humans: ~84%) use **exact grid generation** — the model must produce the correct output pixel-for-pixel. This benchmark uses **5-way discrimination**, which is a strictly easier task. Results are not directly comparable to published figures. The meaningful reference point is the 20% chance baseline.

---

## 10. What Is NOT in Scope

- ARC evaluation split (hidden outputs; not available without competition submission)
- Exact grid generation / output decoding
- Visual rendering of grids
- Per-task difficulty analysis or error categorisation
- Hyperparameter sweep across `T_recomb` or `init_sigma`
- Parallelisation across tasks
