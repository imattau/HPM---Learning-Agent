# Reader Agent Core Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add sentence chunking, online vocab expansion, scored query results, and persistence to ReaderAgent.

**Architecture:** Extend existing `fetch_passages`, `TextDomainConfig`, and `ReaderAgent` with new methods; no new files except tests.

**Tech Stack:** Python stdlib (re, json, pickle), numpy, existing hfn/hpm_ai_v2 stack

---

### Task 1: Sentence-Level Chunking

**Files:**
- Modify: `hpm_ai_v2/utils/text_fetcher.py`
- Test: `tests/test_reader_core.py`

- [x] **Step 1: Write the failing test**

```python
from hpm_ai_v2.utils.text_fetcher import fetch_passages

def test_sentence_mode_splits_sentences():
    text = "First sentence here. Second sentence there. Third one too."
    passages = fetch_passages(text=text, min_length=5, mode="sentence")
    assert len(passages) == 3
    assert passages[0] == "First sentence here."

def test_sentence_mode_filters_short():
    text = "Hi. This is a complete and meaningful sentence for testing."
    passages = fetch_passages(text=text, min_length=20, mode="sentence")
    assert len(passages) == 1
    assert "meaningful" in passages[0]

def test_default_mode_unchanged():
    text = "First paragraph.\n\nSecond paragraph here."
    passages = fetch_passages(text=text, min_length=5)
    assert len(passages) == 2
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_reader_core.py -v
```
Expected: FAIL — `fetch_passages` has no `mode` parameter

- [x] **Step 3: Add `split_sentences` and `mode` param to `fetch_passages`**

In `hpm_ai_v2/utils/text_fetcher.py`, add after imports:

```python
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
```

Change `fetch_passages` signature and body:

```python
def fetch_passages(
    url: str = None,
    text: str = None,
    min_length: int = 40,
    mode: str = "paragraph",
) -> List[str]:
    if url is not None:
        text = fetch_url(url)
    if not text:
        return []
    if mode == "sentence":
        chunks = split_sentences(text)
    else:
        chunks = [p.strip() for p in re.split(r'\n\s*\n', text)]
    return [c for c in chunks if len(c) >= min_length]
```

- [x] **Step 4: Run tests**

```
pytest tests/test_reader_core.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/utils/text_fetcher.py tests/test_reader_core.py
git commit -m "feat: add sentence-level chunking mode to fetch_passages"
```

---

### Task 2: Online Vocab Expansion

**Files:**
- Modify: `hpm_ai_v2/domains/text_domain.py`
- Test: `tests/test_reader_core.py`

- [x] **Step 1: Write the failing test**

```python
from hpm_ai_v2.domains.text_domain import TextDomainConfig

def test_expand_vocab_adds_new_terms():
    passages = ["machine learning trains models on data"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    initial_size = len(config.concepts)
    config.expand_vocab(["neural networks deep learning"], max_new=5)
    assert len(config.concepts) > initial_size
    assert "neural" in config.concepts or "networks" in config.concepts

def test_expand_vocab_no_duplicates():
    passages = ["machine learning trains models"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    config.expand_vocab(["machine learning trains models"])
    counts = {w: config.concepts.count(w) for w in config.concepts}
    assert all(v == 1 for v in counts.values())
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_reader_core.py::test_expand_vocab_adds_new_terms -v
```
Expected: FAIL — `TextDomainConfig` has no `expand_vocab` method

- [x] **Step 3: Add `expand_vocab` to `TextDomainConfig`**

In `hpm_ai_v2/domains/text_domain.py`, add method to `TextDomainConfig`:

```python
def expand_vocab(self, new_passages: List[str], max_new: int = 50) -> int:
    existing = set(self.concepts)
    doc_freq: Counter = Counter()
    for p in new_passages:
        doc_freq.update(set(tokenise(p)))
    candidates = [w for w, _ in doc_freq.most_common(max_new * 2) if w not in existing]
    added = candidates[:max_new]
    n_docs = max(len(self._passages) + len(new_passages), 1)
    for w in added:
        self.concepts.append(w)
        self.idf[w] = math.log((n_docs + 1) / (doc_freq[w] + 1)) + 1.0
    self.DIM = len(self.concepts)
    self.m_dim = self.S_DIM + self.DIM
    return len(added)
```

- [x] **Step 4: Run tests**

```
pytest tests/test_reader_core.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/domains/text_domain.py tests/test_reader_core.py
git commit -m "feat: add online vocab expansion to TextDomainConfig"
```

---

### Task 3: Scored Query Results

**Files:**
- Modify: `hpm_ai_v2/agents/reader_agent.py`
- Test: `tests/test_reader_core.py`

- [x] **Step 1: Write the failing test**

```python
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def test_query_scored_returns_tuples():
    passages = [
        "machine learning trains models on data",
        "the cat sat on the mat outside",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    results = agent.query_scored("machine learning data")
    assert len(results) > 0
    assert isinstance(results[0], tuple)
    text, score = results[0]
    assert isinstance(text, str)
    assert isinstance(score, float)

def test_query_scored_orders_by_relevance():
    passages = [
        "machine learning trains models on data science",
        "the cat sat on the mat outside garden",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    results = agent.query_scored("machine learning data")
    assert "machine" in results[0][0] or "learning" in results[0][0]
    assert results[0][1] >= results[-1][1]
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_reader_core.py::test_query_scored_returns_tuples -v
```
Expected: FAIL — `ReaderAgent` has no `query_scored` method

- [x] **Step 3: Add `query_scored` to `ReaderAgent`**

In `hpm_ai_v2/agents/reader_agent.py`, add import at top:
```python
from hfn.hfn import HFN
```
(already imported — verify it's there)

Add method after `query_top_k`:

```python
def query_scored(self, question: str, k: int = 10) -> List[tuple]:
    """Return (text, score) pairs sorted by cosine similarity descending."""
    if not self.config._passages:
        return []
    query_mu = self.config.encode_passage(question)
    query_vec = query_mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
    results = []
    for idx, pvec in enumerate(self.config._passage_vecs):
        p_vec = pvec[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        denom = (np.linalg.norm(query_vec) * np.linalg.norm(p_vec)) + 1e-9
        score = float(np.dot(query_vec, p_vec) / denom)
        results.append((self.config.get_passage(idx), score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
```

- [x] **Step 4: Run tests**

```
pytest tests/test_reader_core.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/agents/reader_agent.py tests/test_reader_core.py
git commit -m "feat: add query_scored method to ReaderAgent"
```

---

### Task 4: Persistence (Save/Load)

**Files:**
- Modify: `hpm_ai_v2/agents/reader_agent.py`
- Test: `tests/test_reader_core.py`

Note: Uses pickle/json consistent with BaseHFNAgent.save_state pattern already in codebase.

- [x] **Step 1: Write the failing test**

```python
import tempfile, os
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def test_save_and_load_restores_passages():
    passages = [
        "machine learning trains models on data",
        "the cat sat on the mat outside",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.save_reader(tmpdir)
        agent2 = ReaderAgent.load_reader(tmpdir)
    result = agent2.query("machine learning")
    assert result is not None
    assert isinstance(result, str)

def test_save_and_load_preserves_query_results():
    passages = [
        "machine learning trains models on data science",
        "the cat sat on the mat outside garden",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.save_reader(tmpdir)
        agent2 = ReaderAgent.load_reader(tmpdir)
    r1 = agent.query("machine learning")
    r2 = agent2.query("machine learning")
    assert r1 == r2
```

- [x] **Step 2: Run test to verify it fails**

```
pytest tests/test_reader_core.py::test_save_and_load_restores_passages -v
```
Expected: FAIL — `ReaderAgent` has no `save_reader`/`load_reader`

- [x] **Step 3: Add persistence to `ReaderAgent`**

In `hpm_ai_v2/agents/reader_agent.py`, add imports:
```python
import json
import pickle
from pathlib import Path
```

Add methods to `ReaderAgent`:

```python
def save_reader(self, directory: str) -> None:
    """Persist passages, vocab, and vectors to directory."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "concepts": self.config.concepts,
        "idf": self.config.idf,
        "passages": self.config._passages,
        "s_dim": self.config.S_DIM,
    }
    with open(path / "reader_meta.json", "w") as f:
        json.dump(meta, f)
    with open(path / "reader_vecs.pkl", "wb") as f:
        pickle.dump(self.config._passage_vecs, f)

@classmethod
def load_reader(cls, directory: str) -> "ReaderAgent":
    """Restore a saved ReaderAgent from directory."""
    path = Path(directory)
    with open(path / "reader_meta.json") as f:
        meta = json.load(f)
    with open(path / "reader_vecs.pkl", "rb") as f:
        vecs = pickle.load(f)
    config = TextDomainConfig(
        concepts=meta["concepts"],
        idf=meta["idf"],
        s_dim=meta["s_dim"],
    )
    config._passages = meta["passages"]
    config._passage_vecs = vecs
    agent = cls(config)
    for idx, text in enumerate(config._passages):
        import numpy as np
        from hfn.hfn import HFN
        mu = config._passage_vecs[idx]
        sigma = np.ones(config.m_dim) * 0.1
        node = HFN(mu=mu, sigma=sigma, id=f"passage_{idx}", use_diag=True)
        node.metadata = {"passage_idx": idx, "text": text}
        agent.observer.register(node, protected=False, initial_weight=1.0)
        agent.patterns[f"passage_{idx}"] = node
    return agent
```

- [x] **Step 4: Run tests**

```
pytest tests/test_reader_core.py -v
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add hpm_ai_v2/agents/reader_agent.py tests/test_reader_core.py
git commit -m "feat: add save_reader/load_reader persistence to ReaderAgent"
```
