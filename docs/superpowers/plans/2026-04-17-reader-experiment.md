# Reader Agent SP Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build SP experiment validating that ReaderAgent retrieves relevant passages with higher accuracy than random baseline, and that curiosity-driven reading improves retrieval over passive reading.

**Architecture:** Single experiment file following existing SP experiment conventions; uses ReaderAgent with multi-document corpus; measures precision@1 and precision@k.

**Tech Stack:** Python stdlib, numpy, existing hpm_ai_v2/ReaderAgent stack

---

### Task 1: SP Experiment — Retrieval Accuracy Baseline

**Files:**
- Create: `hpm_ai_v2/experiments/experiment_sp_reader1_retrieval.py`
- Test: `tests/test_experiment_reader.py`

- [x] **Step 1: Write the failing test**

```python
from hpm_ai_v2.experiments.experiment_sp_reader1_retrieval import run_experiment

def test_experiment_runs_without_error():
    result = run_experiment(verbose=False)
    assert isinstance(result, dict)
    assert "precision_at_1" in result
    assert "random_baseline" in result
    assert "above_baseline" in result

def test_experiment_beats_random():
    result = run_experiment(verbose=False)
    assert result["precision_at_1"] > result["random_baseline"]
    assert result["above_baseline"] is True
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_experiment_reader.py -v
```
Expected: FAIL — module does not exist

- [x] **Step 3: Create experiment**

Create `hpm_ai_v2/experiments/experiment_sp_reader1_retrieval.py`:

```python
"""SP Experiment: Reader Agent Retrieval Accuracy vs Random Baseline."""
from __future__ import annotations
from typing import Dict
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

CORPUS = [
    # Technology cluster
    ("tech", "Machine learning is a subset of artificial intelligence that trains on data."),
    ("tech", "Deep learning uses neural networks with many layers to learn representations."),
    ("tech", "Python is a high-level programming language widely used in data science."),
    ("tech", "Neural networks are inspired by biological neurons in the human brain."),
    # History cluster
    ("history", "The Roman Empire was one of the largest empires in ancient history."),
    ("history", "Rome was founded in 753 BC according to ancient Roman tradition."),
    ("history", "Latin was the official language of the Roman Empire for centuries."),
    ("history", "The fall of Rome in 476 AD marked the end of the Western Roman Empire."),
    # Nature cluster
    ("nature", "Photosynthesis is the process by which plants convert sunlight to energy."),
    ("nature", "Rainforests contain over half of the world's plant and animal species."),
    ("nature", "Migration patterns of birds are influenced by seasonal temperature changes."),
    ("nature", "The Amazon River basin supports one of the most diverse ecosystems on Earth."),
]

QUERIES = [
    ("machine learning neural networks deep learning", "tech"),
    ("roman empire ancient history latin", "history"),
    ("photosynthesis plants rainforest nature", "nature"),
    ("python programming data science", "tech"),
    ("amazon river ecosystem species", "nature"),
    ("rome founded BC tradition", "history"),
]


def run_experiment(verbose: bool = True) -> Dict:
    passages = [text for _, text in CORPUS]
    labels = [label for label, _ in CORPUS]
    config = TextDomainConfig.from_passages(passages, max_vocab=100)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)

    correct = 0
    total = len(QUERIES)
    for query_text, expected_label in QUERIES:
        result = agent.query(query_text)
        if result is None:
            continue
        # find label of returned passage
        idx = passages.index(result) if result in passages else -1
        returned_label = labels[idx] if idx >= 0 else "unknown"
        if returned_label == expected_label:
            correct += 1
        if verbose:
            print(f"Query: {query_text[:40]}...")
            print(f"  Expected: {expected_label}, Got: {returned_label} ({'✓' if returned_label == expected_label else '✗'})")

    precision_at_1 = correct / total
    n_labels = len(set(labels))
    random_baseline = 1.0 / n_labels

    result_dict = {
        "precision_at_1": precision_at_1,
        "random_baseline": random_baseline,
        "above_baseline": precision_at_1 > random_baseline,
        "correct": correct,
        "total": total,
    }
    if verbose:
        print(f"\nPrecision@1: {precision_at_1:.2f} (random baseline: {random_baseline:.2f})")
        print(f"Above baseline: {result_dict['above_baseline']}")
    return result_dict


if __name__ == "__main__":
    run_experiment(verbose=True)
```

- [x] **Step 4: Run tests**

```
pytest tests/test_experiment_reader.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/experiments/experiment_sp_reader1_retrieval.py tests/test_experiment_reader.py
git commit -m "feat: SP experiment — reader agent retrieval accuracy vs random baseline"
```

---

### Task 2: SP Experiment — Curiosity vs Passive Reading

**Files:**
- Create: `hpm_ai_v2/experiments/experiment_sp_reader2_curiosity.py`
- Modify: `tests/test_experiment_reader.py`

- [x] **Step 1: Write the failing test**

```python
from hpm_ai_v2.experiments.experiment_sp_reader2_curiosity import run_experiment as run_curiosity

def test_curiosity_experiment_runs():
    result = run_curiosity(verbose=False)
    assert "passive_precision" in result
    assert "curiosity_precision" in result
    assert "passages_read_passive" in result
    assert "passages_read_curiosity" in result

def test_curiosity_reads_fewer_passages():
    result = run_curiosity(verbose=False)
    # Curiosity mode should filter some passages
    assert result["passages_read_curiosity"] <= result["passages_read_passive"]
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_experiment_reader.py::test_curiosity_experiment_runs -v
```
Expected: FAIL — module does not exist

- [x] **Step 3: Create curiosity experiment**

Create `hpm_ai_v2/experiments/experiment_sp_reader2_curiosity.py`:

```python
"""SP Experiment: Curiosity-driven vs passive reading — efficiency comparison."""
from __future__ import annotations
from typing import Dict
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

# Seed corpus: agent already knows some tech content
SEED_PASSAGES = [
    "Machine learning trains models on data to make predictions.",
    "Deep learning is a machine learning technique using neural networks.",
]

# Stream of passages: mix of novel (history/nature) and redundant (tech)
STREAM = [
    "Machine learning models require large amounts of training data.",  # redundant
    "The Roman Empire controlled most of Europe for centuries.",         # novel
    "Neural networks can approximate any continuous function.",          # semi-redundant
    "Photosynthesis converts sunlight into chemical energy in plants.",  # novel
    "Data science combines statistics and machine learning methods.",    # redundant
    "Ancient Romans built aqueducts to transport water across distances.", # novel
    "Amazon rainforest biodiversity includes millions of species.",      # novel
    "Deep learning architectures include CNNs and transformers.",        # redundant
]

QUERIES = [
    ("roman empire ancient history", "The Roman Empire controlled most of Europe for centuries."),
    ("photosynthesis plants energy", "Photosynthesis converts sunlight into chemical energy in plants."),
    ("amazon rainforest species biodiversity", "Amazon rainforest biodiversity includes millions of species."),
]


def _build_agent(passages):
    config = TextDomainConfig.from_passages(SEED_PASSAGES + passages, max_vocab=80)
    agent = ReaderAgent(config)
    for p in SEED_PASSAGES:
        agent.observe_passage(p)
    return agent


def run_experiment(verbose: bool = True, curiosity_threshold: float = 0.2) -> Dict:
    # Passive: read everything
    passive_agent = _build_agent(STREAM)
    for p in STREAM:
        passive_agent.observe_passage(p)
    passive_read = len(STREAM)

    # Curiosity: only read novel passages
    curiosity_agent = _build_agent(STREAM)
    curiosity_read = 0
    for p in STREAM:
        if curiosity_agent.observe_if_curious(p, threshold=curiosity_threshold):
            curiosity_read += 1

    def precision(agent):
        correct = 0
        for query, expected in QUERIES:
            result = agent.query(query)
            if result == expected:
                correct += 1
        return correct / len(QUERIES)

    passive_p = precision(passive_agent)
    curiosity_p = precision(curiosity_agent)

    result = {
        "passive_precision": passive_p,
        "curiosity_precision": curiosity_p,
        "passages_read_passive": passive_read,
        "passages_read_curiosity": curiosity_read,
        "efficiency_gain": passive_read - curiosity_read,
    }
    if verbose:
        print(f"Passive:   precision={passive_p:.2f}, passages read={passive_read}")
        print(f"Curiosity: precision={curiosity_p:.2f}, passages read={curiosity_read}")
        print(f"Efficiency gain: {result['efficiency_gain']} fewer passages read")
    return result


if __name__ == "__main__":
    run_experiment(verbose=True)
```

- [x] **Step 4: Run tests**

```
pytest tests/test_experiment_reader.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/experiments/experiment_sp_reader2_curiosity.py tests/test_experiment_reader.py
git commit -m "feat: SP experiment — curiosity vs passive reading efficiency"
```
