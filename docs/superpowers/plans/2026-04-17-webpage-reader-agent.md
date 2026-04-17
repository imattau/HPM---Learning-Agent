# Webpage Reader Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HFN-native agent that reads webpages/text files, stores passage patterns in TieredForest, and retrieves relevant passages in response to queries — no LLM required.

**Architecture:** Text is chunked into passages (sentences/paragraphs), each encoded as a TF-IDF sparse vector stored as an HFN node. The Observer learns which passage types recur across documents. Queries are encoded the same way and retrieved via nearest-neighbour search in TieredForest.

**Tech Stack:** Python stdlib only (`urllib`, `html.parser`) for fetching/parsing. `numpy` for vectors. Existing `hfn`, `TieredForest`, `Observer`, `BaseHFNAgent` unchanged.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `hpm_ai_v2/domains/text_domain.py` | Create | DomainConfig for text; TF-IDF vocabulary; passage encoding |
| `hpm_ai_v2/domains/text_renderer.py` | Create | HFN node → original passage text |
| `hpm_ai_v2/utils/oracle/text_oracle.py` | Create | Cosine similarity evaluator; compute_state for passages |
| `hpm_ai_v2/agents/reader_agent.py` | Create | ReaderAgent: observe_document(), query() |
| `hpm_ai_v2/utils/text_fetcher.py` | Create | URL fetch + HTML strip → plain text paragraphs |
| `tests/test_webpage_reader.py` | Create | End-to-end and unit tests |

---

## Task 1: Text Fetcher

**Files:**
- Create: `hpm_ai_v2/utils/text_fetcher.py`
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_webpage_reader.py
from hpm_ai_v2.utils.text_fetcher import fetch_passages, strip_html

def test_strip_html_removes_tags():
    html = "<p>Hello <b>world</b></p>"
    result = strip_html(html)
    assert "Hello world" in result
    assert "<" not in result

def test_strip_html_removes_scripts():
    html = "<script>alert('x')</script><p>Content</p>"
    result = strip_html(html)
    assert "alert" not in result
    assert "Content" in result

def test_fetch_passages_splits_paragraphs():
    text = "First sentence. Second sentence.\n\nNew paragraph here."
    passages = fetch_passages(text=text, min_length=5)
    assert len(passages) >= 2
    assert all(isinstance(p, str) for p in passages)

def test_fetch_passages_filters_short():
    text = "Hi.\n\nThis is a longer and more meaningful passage."
    passages = fetch_passages(text=text, min_length=20)
    assert len(passages) == 1
    assert "meaningful" in passages[0]
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest tests/test_webpage_reader.py::test_strip_html_removes_tags -v
```
Expected: `ModuleNotFoundError: hpm_ai_v2.utils.text_fetcher`

- [ ] **Step 3: Implement text_fetcher.py**

```python
# hpm_ai_v2/utils/text_fetcher.py
"""Fetch and parse webpage or plain text into passage chunks."""
from __future__ import annotations
import re
import urllib.request
from html.parser import HTMLParser
from typing import List, Optional


class _StripHTMLParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "head", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    """Remove HTML tags and script/style content, return plain text."""
    parser = _StripHTMLParser()
    parser.feed(html)
    text = parser.get_text()
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch URL and return stripped plain text."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return strip_html(raw)


def fetch_passages(
    url: Optional[str] = None,
    text: Optional[str] = None,
    min_length: int = 40,
) -> List[str]:
    """
    Fetch text from URL or accept raw text, split into passages.
    Passages are paragraph-separated chunks filtered by min_length.
    """
    if url is not None:
        text = fetch_url(url)
    if text is None:
        raise ValueError("Provide url or text")

    # Split on blank lines or sentence boundaries
    raw_chunks = re.split(r"\n\n+", text)
    passages = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if len(chunk) >= min_length:
            passages.append(chunk)
    return passages
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_webpage_reader.py::test_strip_html_removes_tags tests/test_webpage_reader.py::test_strip_html_removes_scripts tests/test_webpage_reader.py::test_fetch_passages_splits_paragraphs tests/test_webpage_reader.py::test_fetch_passages_filters_short -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v2/utils/text_fetcher.py tests/test_webpage_reader.py
git commit -m "feat: add text_fetcher for webpage passage extraction"
```

---

## Task 2: Text Domain (TF-IDF Encoding)

**Files:**
- Create: `hpm_ai_v2/domains/text_domain.py`
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_webpage_reader.py
from hpm_ai_v2.domains.text_domain import TextDomainConfig

def test_text_domain_builds_vocab():
    passages = ["the cat sat on the mat", "the dog barked loudly"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    assert len(config.concepts) <= 10
    assert "cat" in config.concepts or "dog" in config.concepts

def test_text_domain_encodes_passage():
    passages = ["the cat sat on the mat", "the dog barked loudly"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    vec = config.encode_passage("cat sat mat")
    assert vec.shape == (config.m_dim,)
    assert vec.sum() > 0

def test_text_domain_similar_passages_closer():
    passages = ["machine learning models train on data",
                "neural networks learn from examples",
                "the cat sat on the mat"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    v1 = config.encode_passage("machine learning neural")
    v2 = config.encode_passage("models train data examples")
    v3 = config.encode_passage("cat mat sat")
    import numpy as np
    sim_close = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    sim_far = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3) + 1e-9)
    assert sim_close > sim_far
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_webpage_reader.py::test_text_domain_builds_vocab -v
```
Expected: `ModuleNotFoundError: hpm_ai_v2.domains.text_domain`

- [ ] **Step 3: Implement text_domain.py**

```python
# hpm_ai_v2/domains/text_domain.py
"""Text domain: TF-IDF encoded passages as HFN pattern substrate."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import List, Dict
import numpy as np
from hpm_ai_v2.domains.base import DomainConfig


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that",
    "i", "we", "you", "he", "she", "they", "my", "our", "your",
}


def tokenise(text: str) -> List[str]:
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


class TextDomainConfig(DomainConfig):
    """
    DomainConfig for text passages.
    Concepts = vocabulary words. State encodes TF-IDF weights.
    """

    def __init__(self, concepts: List[str], idf: Dict[str, float], s_dim: int = 20):
        super().__init__(concepts, s_dim=s_dim)
        self.idf = idf
        # Store original passages for retrieval (text is the substrate)
        self._passages: List[str] = []
        self._passage_vecs: List[np.ndarray] = []

    @classmethod
    def from_passages(
        cls,
        passages: List[str],
        max_vocab: int = 200,
        s_dim: int = 20,
    ) -> "TextDomainConfig":
        """Build vocab and IDF from a corpus of passages."""
        doc_freq: Counter = Counter()
        all_tokens = []
        for p in passages:
            tokens = set(tokenise(p))
            doc_freq.update(tokens)
            all_tokens.extend(tokenise(p))

        # Keep top-N by document frequency
        vocab = [w for w, _ in doc_freq.most_common(max_vocab)]
        n_docs = max(len(passages), 1)
        idf = {w: math.log((n_docs + 1) / (doc_freq[w] + 1)) + 1.0 for w in vocab}
        return cls(vocab, idf, s_dim=s_dim)

    def encode_passage(self, text: str) -> np.ndarray:
        """
        Encode a passage as an HFN mu vector.
        Layout: [s_dim zeros | TF-IDF concept weights | s_dim zeros]
        The concept slice carries the passage's TF-IDF signature.
        """
        tokens = tokenise(text)
        tf: Counter = Counter(tokens)
        n = max(len(tokens), 1)

        concept_vec = np.zeros(self.DIM)
        for i, word in enumerate(self.concepts):
            if word in tf:
                tfidf = (tf[word] / n) * self.idf.get(word, 1.0)
                concept_vec[i] = tfidf

        # Normalise
        norm = np.linalg.norm(concept_vec)
        if norm > 0:
            concept_vec /= norm

        mu = np.zeros(self.m_dim)
        mu[self.S_DIM: self.S_DIM + self.DIM] = concept_vec
        return mu

    def register_passage(self, text: str) -> int:
        """Store passage text; return its index."""
        idx = len(self._passages)
        self._passages.append(text)
        self._passage_vecs.append(self.encode_passage(text))
        return idx

    def get_passage(self, idx: int) -> str:
        return self._passages[idx]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_webpage_reader.py::test_text_domain_builds_vocab tests/test_webpage_reader.py::test_text_domain_encodes_passage tests/test_webpage_reader.py::test_text_domain_similar_passages_closer -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v2/domains/text_domain.py tests/test_webpage_reader.py
git commit -m "feat: add TextDomainConfig with TF-IDF passage encoding"
```

---

## Task 3: Text Renderer

**Files:**
- Create: `hpm_ai_v2/domains/text_renderer.py`
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_webpage_reader.py
from hpm_ai_v2.domains.text_renderer import TextRenderer

def test_renderer_returns_passage_text():
    from hfn.hfn import HFN
    import numpy as np
    passages = ["the cat sat on the mat"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    config.register_passage("the cat sat on the mat")
    renderer = TextRenderer(config)

    mu = config.encode_passage("the cat sat on the mat")
    # Store passage index in first S_DIM slot
    mu[0] = 0.0  # passage idx 0
    node = HFN(mu=mu, sigma=np.ones(config.m_dim), use_diag=True)
    node.metadata = {"passage_idx": 0}

    result = renderer.render(node)
    assert result == "the cat sat on the mat"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_webpage_reader.py::test_renderer_returns_passage_text -v
```
Expected: `ModuleNotFoundError: hpm_ai_v2.domains.text_renderer`

- [ ] **Step 3: Implement text_renderer.py**

```python
# hpm_ai_v2/domains/text_renderer.py
"""Renders an HFN text passage node back to its original string."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hfn.hfn import HFN
from hpm_ai_v2.domains.text_domain import TextDomainConfig


class TextRenderer:
    """Converts HFN passage node → original passage text string."""

    def __init__(self, config: TextDomainConfig) -> None:
        self.config = config

    def render(self, node: "HFN") -> str:
        """Return the stored passage text for this node."""
        metadata = getattr(node, "metadata", {})
        idx = metadata.get("passage_idx")
        if idx is not None and 0 <= idx < len(self.config._passages):
            return self.config.get_passage(idx)
        # Fallback: reconstruct from top concept weights
        concept_slice = node.mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
        top_indices = concept_slice.argsort()[::-1][:10]
        words = [self.config.concepts[i] for i in top_indices if concept_slice[i] > 0]
        return " ".join(words)
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_webpage_reader.py::test_renderer_returns_passage_text -v
```
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v2/domains/text_renderer.py tests/test_webpage_reader.py
git commit -m "feat: add TextRenderer returning original passage text from HFN node"
```

---

## Task 4: Text Oracle

**Files:**
- Create: `hpm_ai_v2/utils/oracle/text_oracle.py`
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_webpage_reader.py
from hpm_ai_v2.utils.oracle.text_oracle import TextOracle
import numpy as np

def test_text_oracle_state_shape():
    passages = ["machine learning trains models on data"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    oracle = TextOracle(config)
    state = oracle.compute_state(
        outputs=["machine learning"],
        errors=[None],
        code="",
    )
    assert state.shape == (config.S_DIM,)

def test_text_oracle_similar_query_high_score():
    passages = ["machine learning trains models on data",
                "the cat sat on the mat"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    oracle = TextOracle(config)
    state_relevant = oracle.compute_state(
        outputs=["machine learning models"], errors=[None]
    )
    state_irrelevant = oracle.compute_state(
        outputs=["cat mat sat"], errors=[None]
    )
    # First component = similarity score; relevant query should score higher
    assert state_relevant[0] >= state_irrelevant[0]
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_webpage_reader.py::test_text_oracle_state_shape -v
```
Expected: `ModuleNotFoundError: hpm_ai_v2.utils.oracle.text_oracle`

- [ ] **Step 3: Implement text_oracle.py**

```python
# hpm_ai_v2/utils/oracle/text_oracle.py
"""Oracle that evaluates text passage retrieval quality via cosine similarity."""
from __future__ import annotations
from typing import Any, List, Optional
import numpy as np
from hpm_ai_v2.utils.oracle.base import BaseOracle
from hpm_ai_v2.domains.text_domain import TextDomainConfig


class TextOracle(BaseOracle):
    """
    Evaluates how well a retrieved passage answers a query.
    compute_state() encodes the query/output text and returns a
    similarity-based state vector that the agent uses for retrieval scoring.
    """

    def __init__(self, config: TextDomainConfig) -> None:
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
        inputs: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Encode output text as a state vector.
        outputs[0] is the query string.
        Returns S_DIM vector where [0] = max cosine similarity to stored passages.
        """
        state = np.zeros(self.config.S_DIM)
        if not outputs or not isinstance(outputs[0], str):
            return state

        query_mu = self.config.encode_passage(outputs[0])
        query_vec = query_mu[self.config.S_DIM: self.config.S_DIM + self.config.DIM]

        if not self.config._passage_vecs:
            return state

        # Compute cosine similarities to all stored passages
        sims = []
        for pvec in self.config._passage_vecs:
            pvec_concept = pvec[self.config.S_DIM: self.config.S_DIM + self.config.DIM]
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(pvec_concept)) + 1e-9
            sims.append(float(np.dot(query_vec, pvec_concept) / denom))

        state[0] = max(sims) if sims else 0.0
        state[1] = float(np.mean(sims)) if sims else 0.0
        return state
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_webpage_reader.py::test_text_oracle_state_shape tests/test_webpage_reader.py::test_text_oracle_similar_query_high_score -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v2/utils/oracle/text_oracle.py tests/test_webpage_reader.py
git commit -m "feat: add TextOracle computing cosine similarity retrieval state"
```

---

## Task 5: Reader Agent

**Files:**
- Create: `hpm_ai_v2/agents/reader_agent.py`
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_webpage_reader.py
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def test_reader_agent_observe_and_query():
    passages = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "The cat sat on the mat near the window.",
        "Deep learning uses many layers of neural networks.",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=50)
    agent = ReaderAgent(config)

    for p in passages:
        agent.observe_passage(p)

    result = agent.query("machine learning neural networks")
    assert result is not None
    assert isinstance(result, str)
    # Should retrieve a relevant passage, not the cat one
    assert "cat" not in result.lower() or "learning" in result.lower()

def test_reader_agent_observe_document():
    config = TextDomainConfig.from_passages(
        ["this is a test passage for the reader agent"],
        max_vocab=30,
    )
    agent = ReaderAgent(config)
    text = "First paragraph about machine learning.\n\nSecond paragraph about cats."
    agent.observe_document(text)
    result = agent.query("machine learning")
    assert result is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_webpage_reader.py::test_reader_agent_observe_and_query -v
```
Expected: `ModuleNotFoundError: hpm_ai_v2.agents.reader_agent`

- [ ] **Step 3: Implement reader_agent.py**

```python
# hpm_ai_v2/agents/reader_agent.py
"""ReaderAgent: observes text passages and retrieves relevant ones for queries."""
from __future__ import annotations
from typing import Optional, List
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.text_renderer import TextRenderer
from hpm_ai_v2.utils.oracle.text_oracle import TextOracle
from hpm_ai_v2.utils.text_fetcher import fetch_passages


class ReaderAgent(BaseHFNAgent):
    """
    HFN-native agent that reads text/webpages and retrieves relevant passages.

    Usage:
        agent = ReaderAgent(config)
        agent.observe_url("https://example.com")   # or observe_document(text)
        result = agent.query("what is machine learning?")
    """

    def __init__(self, config: TextDomainConfig, **kwargs) -> None:
        renderer = TextRenderer(config)
        super().__init__(config, renderer=renderer, **kwargs)
        self.oracle = TextOracle(config)
        self.counting_oracle.wrapped = self.oracle
        self._next_passage_idx = 0

    def observe_passage(self, text: str) -> None:
        """Encode a single passage as an HFN node and register in forest."""
        idx = self.config.register_passage(text)
        mu = self.config.encode_passage(text)
        sigma = np.ones(self.config.m_dim) * 0.1
        node = HFN(
            mu=mu,
            sigma=sigma,
            id=f"passage_{idx}",
            use_diag=True,
        )
        node.metadata = {"passage_idx": idx, "text": text}
        self.observer.register(node, protected=False, initial_weight=1.0)
        self.patterns[f"passage_{idx}"] = node

    def observe_document(self, text: str, min_length: int = 40) -> int:
        """Split text into passages and observe each one. Returns passage count."""
        passages = fetch_passages(text=text, min_length=min_length)
        for p in passages:
            self.observe_passage(p)
        return len(passages)

    def observe_url(self, url: str, min_length: int = 40) -> int:
        """Fetch URL, parse into passages, observe each. Returns passage count."""
        passages = fetch_passages(url=url, min_length=min_length)
        for p in passages:
            self.observe_passage(p)
        return len(passages)

    def query(self, question: str, top_k: int = 1) -> Optional[str]:
        """
        Retrieve the most relevant passage for the given query string.
        Returns passage text, or None if no passages observed yet.
        """
        if not self.config._passages:
            return None

        query_mu = self.config.encode_passage(question)
        query_node = HFN(
            mu=query_mu,
            sigma=np.ones(self.config.m_dim),
            use_diag=True,
        )
        candidates = self.retriever.retrieve(query_node, k=top_k)
        if not candidates:
            return None

        best = candidates[0]
        return self.renderer.render(best)

    def query_top_k(self, question: str, k: int = 3) -> List[str]:
        """Retrieve top-k most relevant passages for the query."""
        if not self.config._passages:
            return []

        query_mu = self.config.encode_passage(question)
        query_node = HFN(
            mu=query_mu,
            sigma=np.ones(self.config.m_dim),
            use_diag=True,
        )
        candidates = self.retriever.retrieve(query_node, k=k)
        return [self.renderer.render(c) for c in candidates]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_webpage_reader.py::test_reader_agent_observe_and_query tests/test_webpage_reader.py::test_reader_agent_observe_document -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v2/agents/reader_agent.py tests/test_webpage_reader.py
git commit -m "feat: add ReaderAgent with observe_passage, observe_document, observe_url, query"
```

---

## Task 6: End-to-End Integration Test

**Files:**
- Test: `tests/test_webpage_reader.py`

- [ ] **Step 1: Write integration test**

```python
# Add to tests/test_webpage_reader.py

def test_end_to_end_multi_document():
    """Agent observes two documents and retrieves from correct one."""
    doc1 = (
        "Python is a high-level programming language.\n\n"
        "It is widely used in data science and machine learning.\n\n"
        "Python supports object-oriented and functional programming."
    )
    doc2 = (
        "The Roman Empire was one of the largest empires in history.\n\n"
        "Rome was founded in 753 BC according to tradition.\n\n"
        "Latin was the official language of the Roman Empire."
    )

    all_passages = (
        [p for p in doc1.split("\n\n") if len(p) > 10] +
        [p for p in doc2.split("\n\n") if len(p) > 10]
    )
    config = TextDomainConfig.from_passages(all_passages, max_vocab=100)
    agent = ReaderAgent(config)
    agent.observe_document(doc1)
    agent.observe_document(doc2)

    result_tech = agent.query("programming language python data science")
    result_hist = agent.query("roman empire latin history")

    assert result_tech is not None
    assert result_hist is not None
    # Tech query should NOT return Roman history passage
    assert "roman" not in result_tech.lower() or "python" in result_tech.lower()
    # History query should NOT return Python passage
    assert "python" not in result_hist.lower() or "roman" in result_hist.lower()

def test_query_top_k_returns_multiple():
    passages = [
        "machine learning trains models",
        "deep learning uses neural networks",
        "the cat sat on the mat",
        "artificial intelligence is broad field",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=40)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)

    results = agent.query_top_k("machine learning artificial intelligence", k=2)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/test_webpage_reader.py -v
```
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_webpage_reader.py
git commit -m "test: add end-to-end integration tests for ReaderAgent"
```

---

## Self-Review

### Spec Coverage
- ✅ Fetch webpages (observe_url via text_fetcher.fetch_url)
- ✅ Read plain text documents (observe_document)
- ✅ HFN-only, no LLM (TF-IDF encoding, no external models)
- ✅ Query/retrieval (query, query_top_k)
- ✅ Hierarchy L2 sentence-level patterns stored in TieredForest
- ✅ Observer registers passages as patterns
- ✅ Follows existing domain/renderer/oracle/agent structure

### Placeholder Scan
- No TBD, TODO, or "implement later" in any task
- All code blocks complete
- All type signatures consistent across tasks

### Type Consistency
- `TextDomainConfig` used consistently in domain, renderer, oracle, agent
- `TextRenderer(config)` signature matches usage in ReaderAgent.__init__
- `TextOracle(config)` signature matches usage in ReaderAgent.__init__
- `observe_passage(text: str)` consistent across test and implementation
- `query(question: str) -> Optional[str]` consistent

---

Plan complete and saved to `docs/superpowers/plans/2026-04-17-webpage-reader-agent.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints

**Which approach?**
