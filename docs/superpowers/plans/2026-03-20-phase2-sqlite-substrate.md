# HPM Phase 2: SQLiteStore + ExternalSubstrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent pattern storage (SQLiteStore) and external substrate support (LocalFileSubstrate + WikipediaSubstrate), so agents can resume learning across sessions and draw field frequency signals from real-world sources.

**Architecture:** Two independent subsystems extending Phase 1. SQLiteStore implements the PatternStore Protocol using sqlite3 (stdlib) with JSON serialisation. ExternalSubstrate provides a Protocol with two implementations: LocalFileSubstrate (offline, deterministic) and WikipediaSubstrate (live Wikipedia REST API, no key required). Text is vectorised via a deterministic hash trick. Agent gains optional `substrate` parameter; `ext_field_freq` is computed and logged each step (blending into totals deferred to Phase 3 social evaluator).

**Tech Stack:** Python 3.11+, numpy, scipy, sqlite3 (stdlib), requests, pytest, uv

---

## File Map

```
hpm/
  config.py                        # Modify: add alpha_int field
  patterns/
    __init__.py                    # Modify: add PATTERN_REGISTRY + pattern_from_dict
  store/
    sqlite.py                      # Create: SQLiteStore backend
  substrate/
    __init__.py                    # Create: empty package marker
    base.py                        # Create: ExternalSubstrate Protocol + hash_vectorise
    local_file.py                  # Create: LocalFileSubstrate
    wikipedia.py                   # Create: WikipediaSubstrate
  agents/
    agent.py                       # Modify: add substrate param + ext_field_freq in step()

tests/
  store/
    test_sqlite.py                 # Create
  substrate/
    __init__.py                    # Create: empty
    test_base.py                   # Create
    test_local_file.py             # Create
    test_wikipedia.py              # Create
  integration/
    test_phase2.py                 # Create

pyproject.toml                     # Modify: add requests dependency
```

---

## Task 1: Pattern Registry + SQLiteStore

**Files:**
- Modify: `hpm/patterns/__init__.py`
- Create: `hpm/store/sqlite.py`
- Create: `tests/store/test_sqlite.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/store/test_sqlite.py
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.sqlite import SQLiteStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_patterns.db")


@pytest.fixture
def store(db_path):
    return SQLiteStore(db_path)


@pytest.fixture
def pattern():
    return GaussianPattern(mu=np.array([1.0, 2.0]), sigma=np.eye(2))


def test_save_and_load(store, pattern):
    store.save(pattern, weight=0.7, agent_id="agent1")
    loaded_p, loaded_w = store.load(pattern.id)
    assert loaded_p.id == pattern.id
    assert loaded_w == pytest.approx(0.7)
    assert np.allclose(loaded_p.mu, pattern.mu)
    assert np.allclose(loaded_p.sigma, pattern.sigma)


def test_query_returns_agent_patterns(store, pattern):
    other = GaussianPattern(mu=np.array([5.0, 5.0]), sigma=np.eye(2))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    results = store.query("agent1")
    assert len(results) == 1
    assert results[0][0].id == pattern.id


def test_update_weight(store, pattern):
    store.save(pattern, 0.5, "agent1")
    store.update_weight(pattern.id, 0.9)
    _, w = store.load(pattern.id)
    assert w == pytest.approx(0.9)


def test_delete(store, pattern):
    store.save(pattern, 1.0, "agent1")
    store.delete(pattern.id)
    assert store.query("agent1") == []


def test_query_all(store, pattern):
    other = GaussianPattern(mu=np.array([3.0, 4.0]), sigma=np.eye(2))
    store.save(pattern, 0.6, "agent1")
    store.save(other, 0.4, "agent2")
    all_records = store.query_all()
    assert len(all_records) == 2


def test_persistence_across_connections(db_path, pattern):
    """Patterns saved in one connection are readable in a new one."""
    store1 = SQLiteStore(db_path)
    store1.save(pattern, 0.8, "agent1")

    store2 = SQLiteStore(db_path)
    results = store2.query("agent1")
    assert len(results) == 1
    assert results[0][0].id == pattern.id
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/store/test_sqlite.py -v
```

Expected: ImportError (module not found).

- [ ] **Step 3: Add PATTERN_REGISTRY to hpm/patterns/__init__.py**

```python
# hpm/patterns/__init__.py
from .gaussian import GaussianPattern

PATTERN_REGISTRY: dict[str, type] = {
    'gaussian': GaussianPattern,
}


def pattern_from_dict(d: dict):
    """Deserialise a pattern dict using the type registry."""
    cls = PATTERN_REGISTRY.get(d['type'])
    if cls is None:
        raise ValueError(f"Unknown pattern type: {d['type']}")
    return cls.from_dict(d)
```

- [ ] **Step 4: Implement SQLiteStore**

```python
# hpm/store/sqlite.py
import json
import sqlite3
from hpm.patterns import pattern_from_dict


class SQLiteStore:
    """
    SQLite-backed PatternStore. Persists patterns across sessions.
    Schema: patterns(id TEXT PK, agent_id TEXT, pattern_json TEXT, weight REAL).
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        pattern_json TEXT NOT NULL,
        weight REAL NOT NULL
    )
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(self._SCHEMA)

    def save(self, pattern, weight: float, agent_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, agent_id, pattern_json, weight) "
                "VALUES (?, ?, ?, ?)",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )

    def load(self, pattern_id: str) -> tuple:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
        if row is None:
            raise KeyError(pattern_id)
        return pattern_from_dict(json.loads(row[0])), row[1]

    def query(self, agent_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), r[1]) for r in rows]

    def query_all(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pattern_json, weight, agent_id FROM patterns"
            ).fetchall()
        return [(pattern_from_dict(json.loads(r[0])), r[1], r[2]) for r in rows]

    def delete(self, pattern_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))

    def update_weight(self, pattern_id: str, weight: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE patterns SET weight = ? WHERE id = ?",
                (weight, pattern_id),
            )
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/store/test_sqlite.py -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm/patterns/__init__.py hpm/store/sqlite.py tests/store/test_sqlite.py
git commit -m "feat: SQLiteStore persistent backend + pattern_from_dict registry"
```

---

## Task 2: ExternalSubstrate Protocol + hash_vectorise + alpha_int config

**Files:**
- Create: `hpm/substrate/__init__.py`
- Create: `hpm/substrate/base.py`
- Create: `tests/substrate/__init__.py`
- Create: `tests/substrate/test_base.py`
- Modify: `hpm/config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/substrate/test_base.py
import numpy as np
import pytest
from hpm.substrate.base import hash_vectorise


def test_hash_vectorise_returns_correct_dim():
    vec = hash_vectorise("hello world", dim=16)
    assert vec.shape == (16,)


def test_hash_vectorise_is_normalised():
    vec = hash_vectorise("hello world test text", dim=16)
    assert abs(vec.sum() - 1.0) < 1e-6


def test_hash_vectorise_empty_text():
    vec = hash_vectorise("", dim=16)
    assert vec.shape == (16,)
    assert vec.sum() == pytest.approx(0.0)


def test_hash_vectorise_deterministic():
    v1 = hash_vectorise("the quick brown fox", dim=32)
    v2 = hash_vectorise("the quick brown fox", dim=32)
    assert np.allclose(v1, v2)


def test_config_has_alpha_int():
    from hpm.config import AgentConfig
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert hasattr(cfg, 'alpha_int')
    assert cfg.alpha_int == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/substrate/test_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create package skeleton**

```bash
touch hpm/substrate/__init__.py tests/substrate/__init__.py
```

- [ ] **Step 4: Implement ExternalSubstrate Protocol + hash_vectorise**

```python
# hpm/substrate/base.py
from typing import Protocol, Iterator, runtime_checkable
import numpy as np


def hash_vectorise(text: str, dim: int = 32) -> np.ndarray:
    """
    Hash-trick text vectoriser. Maps words to a fixed-dim float array.
    Uses hashlib.md5 for cross-process determinism — Python's hash() is
    randomised per process (PYTHONHASHSEED), md5 is stable across runs.
    Returns a normalised word-frequency vector.
    """
    import hashlib
    vec = np.zeros(dim)
    words = text.lower().split()
    for word in words:
        digest = hashlib.md5(word.encode()).digest()
        idx = int.from_bytes(digest[:4], 'little') % dim
        vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


@runtime_checkable
class ExternalSubstrate(Protocol):
    def fetch(self, query: str) -> list[np.ndarray]: ...
    def field_frequency(self, pattern) -> float: ...
    def stream(self) -> Iterator[np.ndarray]: ...
```

- [ ] **Step 5: Add alpha_int to AgentConfig**

In `hpm/config.py`, add after the `rho` field:

```python
    alpha_int: float = 0.8       # internal/external field blend (1.0 = agents only, §3.8)
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
uv run pytest tests/substrate/test_base.py -v
```

Expected: all 5 PASS.

- [ ] **Step 7: Commit**

```bash
git add hpm/substrate/ tests/substrate/test_base.py tests/substrate/__init__.py hpm/config.py
git commit -m "feat: ExternalSubstrate Protocol, hash_vectorise utility, alpha_int config"
```

---

## Task 3: LocalFileSubstrate

**Files:**
- Create: `hpm/substrate/local_file.py`
- Create: `tests/substrate/test_local_file.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/substrate/test_local_file.py
import numpy as np
import pytest
from hpm.substrate.local_file import LocalFileSubstrate
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def text_dir(tmp_path):
    (tmp_path / "doc1.txt").write_text("the cat sat on the mat")
    (tmp_path / "doc2.txt").write_text("dogs and cats are pets")
    (tmp_path / "doc3.txt").write_text("machine learning is useful")
    return str(tmp_path)


@pytest.fixture
def substrate(text_dir):
    return LocalFileSubstrate(text_dir, feature_dim=16)


def test_fetch_returns_vectors(substrate):
    vecs = substrate.fetch("cat")
    assert len(vecs) > 0
    assert all(v.shape == (16,) for v in vecs)


def test_fetch_no_match_returns_all_docs(substrate):
    vecs = substrate.fetch("nonexistent_xyz_term")
    # Falls back to all documents
    assert len(vecs) == 3


def test_stream_yields_vectors(substrate):
    gen = substrate.stream()
    v = next(gen)
    assert v.shape == (16,)


def test_field_frequency_returns_float_in_range(substrate):
    pattern = GaussianPattern(mu=np.zeros(16), sigma=np.eye(16))
    freq = substrate.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_caching_returns_same_result(substrate):
    vecs1 = substrate.fetch("cat")
    vecs2 = substrate.fetch("cat")
    assert len(vecs1) == len(vecs2)
    for v1, v2 in zip(vecs1, vecs2):
        assert np.allclose(v1, v2)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/substrate/test_local_file.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement LocalFileSubstrate**

```python
# hpm/substrate/local_file.py
import os
from typing import Iterator
import numpy as np
from .base import hash_vectorise


class LocalFileSubstrate:
    """
    ExternalSubstrate backed by local .txt files.
    Deterministic — safe for offline testing and CI.
    Caches fetch results to avoid redundant file reads.
    """

    def __init__(self, directory: str, feature_dim: int = 32):
        self.directory = directory
        self.feature_dim = feature_dim
        self._cache: dict[str, list[np.ndarray]] = {}
        self._texts: list[str] = self._load_texts()

    def _load_texts(self) -> list[str]:
        texts = []
        for fname in sorted(os.listdir(self.directory)):
            fpath = os.path.join(self.directory, fname)
            if os.path.isfile(fpath) and fname.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        return texts

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]
        q_lower = query.lower()
        results = [
            hash_vectorise(text, self.feature_dim)
            for text in self._texts
            if q_lower in text.lower()
        ]
        if not results:
            results = [hash_vectorise(t, self.feature_dim) for t in self._texts]
        self._cache[query] = results
        return results

    def field_frequency(self, pattern) -> float:
        """
        Fraction of documents with positive cosine similarity to pattern.mu.
        Returns float in [0, 1].
        """
        if not self._texts:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        count = sum(
            1 for text in self._texts
            if float(np.dot(mu_unit, hash_vectorise(text, dim))) > 0
        )
        return count / len(self._texts)

    def stream(self) -> Iterator[np.ndarray]:
        """Cycle through all text files indefinitely."""
        idx = 0
        while True:
            if not self._texts:
                return
            yield hash_vectorise(self._texts[idx % len(self._texts)], self.feature_dim)
            idx += 1
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/substrate/test_local_file.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/substrate/local_file.py tests/substrate/test_local_file.py
git commit -m "feat: LocalFileSubstrate — offline text file external substrate with caching"
```

---

## Task 4: WikipediaSubstrate

**Files:**
- Modify: `pyproject.toml`
- Create: `hpm/substrate/wikipedia.py`
- Create: `tests/substrate/test_wikipedia.py`

- [ ] **Step 1: Add requests to pyproject.toml**

Edit `pyproject.toml` — add `"requests>=2.31"` to the `dependencies` list:

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "requests>=2.31",
]
```

Install:

```bash
uv pip install -e ".[dev]"
```

- [ ] **Step 2: Write failing tests**

```python
# tests/substrate/test_wikipedia.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from hpm.substrate.wikipedia import WikipediaSubstrate
from hpm.patterns.gaussian import GaussianPattern


@pytest.fixture
def substrate():
    return WikipediaSubstrate(feature_dim=32)


def _mock_response(status: int, extract: str = ""):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = {'title': 'Test', 'extract': extract}
    return mock


def test_fetch_returns_vectors(substrate):
    with patch('requests.get', return_value=_mock_response(200, "A cat is a small furry animal that purrs.")):
        vecs = substrate.fetch("cat")
    assert len(vecs) > 0
    assert all(v.shape == (32,) for v in vecs)


def test_fetch_404_returns_empty(substrate):
    with patch('requests.get', return_value=_mock_response(404)):
        vecs = substrate.fetch("nonexistent_page_xyz_abc")
    assert vecs == []


def test_field_frequency_returns_float_in_range(substrate):
    pattern = GaussianPattern(mu=np.zeros(32), sigma=np.eye(32))
    with patch('requests.get', return_value=_mock_response(200, "test content here")):
        freq = substrate.field_frequency(pattern)
    assert 0.0 <= freq <= 1.0


def test_caching_avoids_repeated_requests(substrate):
    with patch('requests.get', return_value=_mock_response(200, "A dog is a domestic animal.")) as mock_get:
        substrate.fetch("dog")
        substrate.fetch("dog")
    assert mock_get.call_count == 1
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/substrate/test_wikipedia.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement WikipediaSubstrate**

```python
# hpm/substrate/wikipedia.py
from typing import Iterator
import numpy as np
import requests
from .base import hash_vectorise

_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaSubstrate:
    """
    ExternalSubstrate backed by the Wikipedia REST API.
    No API key required. Caches responses per query.
    Uses hash_vectorise to convert text summaries to fixed-dim float vectors.
    """

    def __init__(self, feature_dim: int = 32, timeout: float = 5.0):
        self.feature_dim = feature_dim
        self.timeout = timeout
        self._cache: dict[str, list[np.ndarray]] = {}

    def fetch(self, query: str) -> list[np.ndarray]:
        if query in self._cache:
            return self._cache[query]

        title = query.replace(' ', '_').title()
        url = _WIKI_API.format(title=title)
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException:
            self._cache[query] = []
            return []

        if resp.status_code != 200:
            self._cache[query] = []
            return []

        extract = resp.json().get('extract', '')
        if not extract:
            self._cache[query] = []
            return []

        sentences = [s.strip() for s in extract.split('.') if s.strip()]
        vecs = [hash_vectorise(s, self.feature_dim) for s in sentences]
        self._cache[query] = vecs
        return vecs

    def field_frequency(self, pattern) -> float:
        """
        Mean cosine similarity between pattern.mu and vectorised Wikipedia summary.
        Uses pattern.label as query if present, else 'knowledge'.
        Returns float in [0, 1].
        """
        query = str(getattr(pattern, 'label', None) or 'knowledge')
        vecs = self.fetch(query)
        if not vecs:
            return 0.0
        dim = self.feature_dim
        mu = np.array(pattern.mu, dtype=float)
        if len(mu) > dim:
            mu = mu[:dim]
        elif len(mu) < dim:
            mu = np.pad(mu, (0, dim - len(mu)))
        mu_norm = np.linalg.norm(mu)
        if mu_norm == 0:
            return 0.0
        mu_unit = mu / mu_norm
        mean_sim = float(np.mean([np.dot(mu_unit, v) for v in vecs]))
        return float(np.clip((mean_sim + 1.0) / 2.0, 0.0, 1.0))

    def stream(self) -> Iterator[np.ndarray]:
        """Stream vectorised sentences from rotating Wikipedia articles."""
        topics = ["mathematics", "science", "history", "language", "cognition"]
        idx = 0
        while True:
            vecs = self.fetch(topics[idx % len(topics)])
            for v in vecs:
                yield v
            idx += 1
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/substrate/test_wikipedia.py -v
```

Expected: all 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml hpm/substrate/wikipedia.py tests/substrate/test_wikipedia.py
git commit -m "feat: WikipediaSubstrate — Wikipedia REST API with sentence vectorisation and caching"
```

---

## Task 5: Agent integration + Phase 2 integration test

**Files:**
- Modify: `hpm/agents/agent.py`
- Create: `tests/integration/test_phase2.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/integration/test_phase2.py
import numpy as np
import pytest
from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.domains.concept import ConceptLearningDomain, Concept
from hpm.store.sqlite import SQLiteStore
from hpm.substrate.local_file import LocalFileSubstrate


def make_domain(seed=0):
    return ConceptLearningDomain(
        concepts=[
            Concept(np.array([1.0, 0.0]), [np.array([0.5, 0.5])], label=0),
            Concept(np.array([0.0, 1.0]), [np.array([0.2, 0.8])], label=1),
        ],
        noise=0.05,
        seed=seed,
    )


def test_agent_resumes_across_sessions(tmp_path):
    """Patterns trained in session 1 are present in session 2 via SQLiteStore."""
    db = str(tmp_path / "agent.db")
    cfg = AgentConfig(agent_id="persist_agent", feature_dim=4)
    domain = make_domain()

    # Session 1: train for 30 steps
    agent1 = Agent(cfg, store=SQLiteStore(db))
    for _ in range(30):
        agent1.step(domain.observe())
    ids1 = {p.id for p, _ in agent1.store.query("persist_agent")}
    assert len(ids1) >= 1

    # Session 2: new Agent instance on same DB — patterns from session 1 present
    agent2 = Agent(cfg, store=SQLiteStore(db))
    ids2 = {p.id for p, _ in agent2.store.query("persist_agent")}
    assert ids1 == ids2


def test_agent_with_substrate_does_not_error(tmp_path):
    """Agent.step() runs without error when substrate is attached."""
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "a.txt").write_text("concept one definition here")
    (text_dir / "b.txt").write_text("concept two explanation there")

    substrate = LocalFileSubstrate(str(text_dir), feature_dim=4)
    cfg = AgentConfig(agent_id="substrate_agent", feature_dim=4)
    agent = Agent(cfg, substrate=substrate)
    domain = make_domain()

    for _ in range(10):
        result = agent.step(domain.observe())
    assert 'mean_accuracy' in result
    assert 'ext_field_freq' in result


def test_step_returns_ext_field_freq(tmp_path):
    """step() returns ext_field_freq key when substrate is set."""
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "c.txt").write_text("learning patterns neural networks")

    substrate = LocalFileSubstrate(str(text_dir), feature_dim=4)
    cfg = AgentConfig(agent_id="freq_agent", feature_dim=4)
    agent = Agent(cfg, substrate=substrate)

    result = agent.step(np.zeros(4))
    assert 'ext_field_freq' in result
    assert 0.0 <= result['ext_field_freq'] <= 1.0


def test_step_without_substrate_has_zero_ext_field_freq():
    """Without substrate, ext_field_freq is 0.0."""
    cfg = AgentConfig(agent_id="no_sub_agent", feature_dim=4)
    agent = Agent(cfg)
    result = agent.step(np.zeros(4))
    assert result['ext_field_freq'] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/integration/test_phase2.py -v
```

Expected: ImportError or AttributeError on `substrate` / `ext_field_freq`.

- [ ] **Step 3: Modify Agent to add substrate parameter and ext_field_freq**

Replace `hpm/agents/agent.py` with:

```python
# hpm/agents/agent.py
import numpy as np
from ..config import AgentConfig
from ..patterns.gaussian import GaussianPattern
from ..evaluators.epistemic import EpistemicEvaluator
from ..evaluators.affective import AffectiveEvaluator
from ..dynamics.meta_pattern_rule import MetaPatternRule
from ..store.memory import InMemoryStore


class Agent:
    """
    Single HPM agent. Wires PatternLibrary, EvaluatorPipeline, and Dynamics.
    Backed by a PatternStore (InMemoryStore by default; SQLiteStore for persistence).
    Optionally connected to an ExternalSubstrate for external field frequency signals.

    Data flow per step (Phase 1/2, §7):
      1. Compute ell_i(t) for each pattern
      2. Update L_i(t) -> A_i(t) via EpistemicEvaluator
      3. Compute E_aff_i(t) via AffectiveEvaluator
      4. Total_i(t) = A_i(t) + beta_aff * E_aff_i(t)
      5. MetaPatternRule -> new weights
      6. Prune + update store
      7. If substrate set: compute ext_field_freq (logged; blending into totals in Phase 3)
    """

    def __init__(self, config: AgentConfig, store=None, substrate=None):
        self.config = config
        self.agent_id = config.agent_id
        self.store = store or InMemoryStore()
        self.substrate = substrate
        self.epistemic = EpistemicEvaluator(lambda_L=config.lambda_L)
        self.affective = AffectiveEvaluator(
            k=config.k,
            c_opt=config.c_opt,
            sigma_c=config.sigma_c,
            alpha_r=config.alpha_r,
        )
        self.dynamics = MetaPatternRule(
            eta=config.eta,
            beta_c=config.beta_c,
            epsilon=config.epsilon,
        )
        self._t = 0
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        # NOTE: mu=np.zeros(...) is unchanged from Phase 1. No semantic change here.
        if not self.store.query(self.agent_id):
            init = GaussianPattern(
                mu=np.zeros(self.config.feature_dim),
                sigma=np.eye(self.config.feature_dim) * self.config.init_sigma,
            )
            self.store.save(init, 1.0, self.agent_id)

    def step(self, x: np.ndarray, reward: float = 0.0) -> dict:
        records = self.store.query(self.agent_id)
        patterns = [p for p, _ in records]
        weights = np.array([w for _, w in records])

        accuracies = []
        e_affs = []
        for pattern in patterns:
            acc = self.epistemic.update(pattern, x)
            e_aff = self.affective.update(pattern, acc, reward)
            accuracies.append(acc)
            e_affs.append(e_aff)

        totals = np.array([
            acc + self.config.beta_aff * e_aff
            for acc, e_aff in zip(accuracies, e_affs)
        ])

        new_weights = self.dynamics.step(patterns, weights, totals)

        # Prune and persist
        for p in patterns:
            self.store.delete(p.id)
        for p, w in zip(patterns, new_weights):
            if w >= self.config.epsilon:
                updated = p.update(x)
                self.store.save(updated, float(w), self.agent_id)

        # External substrate: compute field frequencies (logged; not yet in totals — Phase 3)
        ext_field_freq = 0.0
        if self.substrate is not None:
            freqs = [self.substrate.field_frequency(p) for p in patterns]
            ext_field_freq = float(np.mean(freqs)) if freqs else 0.0

        self._t += 1
        return {
            't': self._t,
            'n_patterns': int(np.sum(new_weights >= self.config.epsilon)),
            'mean_accuracy': float(np.mean(accuracies)),
            'max_weight': float(new_weights.max()),
            'ext_field_freq': ext_field_freq,
        }
```

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/integration/test_phase2.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest --tb=short
```

Expected: all Phase 1 + Phase 2 tests passing (46+ tests, 0 failures).

- [ ] **Step 6: Commit**

```bash
git add hpm/agents/agent.py tests/integration/test_phase2.py
git commit -m "feat: Agent supports SQLiteStore persistence and ExternalSubstrate field frequency (Phase 2)"
```

---

## Summary

Phase 2 delivers:
- `SQLiteStore` — cross-session pattern persistence via stdlib sqlite3
- `PATTERN_REGISTRY` + `pattern_from_dict` — extensible type-safe deserialisation
- `ExternalSubstrate` Protocol + `hash_vectorise` — deterministic text→vector conversion
- `LocalFileSubstrate` — offline deterministic substrate for testing
- `WikipediaSubstrate` — live Wikipedia REST API with sentence vectorisation + caching
- `alpha_int` field in `AgentConfig` — ready for Phase 3 social field blending
- `Agent.substrate` — field frequency computed and logged each step; blending deferred to Phase 3

**Next:** Phase 3 plan (multi-agent PatternField + SocialEvaluator + alpha_int blending into totals).
