# PatternVariant Consolidation + ATIS Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PatternVariant consolidation to the HPM v5 core (promoting near-identical patterns at episode end rather than evicting them), upgrade KnowledgeBaseLookup to WordNet, then validate at scale with an ATIS intent-recognition benchmark.

**Architecture:** `PatternVariant` is a new dataclass in `hpm_ai_v5/core/variant.py`; `PatternStore` gains a `variants` dict and variant-level fallback in `match()`; `PatternManager.end_episode()` runs `_consolidate_variants()` after the existing promotion pass; `CoreConfig` gains `consolidation_threshold`. The ATIS benchmark uses a new `IntentLabelAdapter` and a WordNet-backed `KnowledgeBaseLookup`.

**Tech Stack:** Python 3.12, spaCy en_core_web_sm, NLTK WordNet, HuggingFace datasets, uv, pytest

---

## File Map

| File | Action |
|------|--------|
| `hpm_ai_v5/core/variant.py` | **Create** — PatternVariant dataclass + make_variant factory |
| `hpm_ai_v5/core/config.py` | **Modify** — add consolidation_threshold: float = 0.8 |
| `hpm_ai_v5/core/store.py` | **Modify** — add variants dict, register_variant(), variant fallback in match() |
| `hpm_ai_v5/core/pattern_manager.py` | **Modify** — add _consolidate_variants(), call from end_episode() |
| `hpm_ai_v5/core/__init__.py` | **Modify** — export PatternVariant |
| `hpm_ai_v5/adapter/nlp.py` | **Modify** — replace static KB dict with WordNet in KnowledgeBaseLookup |
| `hpm_ai_v5/adapter/atis.py` | **Create** — IntentLabelAdapter + load_atis() dataset loader |
| `hpm_ai_v5/experiments/run_atis_benchmark.py` | **Create** — ATIS benchmark harness (B1-B4) |
| `tests/test_pattern_variant.py` | **Create** — unit tests for consolidation and variant matching |
| `tests/test_wordnet_kb.py` | **Create** — unit tests for WordNet KB |
| `tests/test_atis_benchmark.py` | **Create** — smoke tests for ATIS pipeline |
| `pyproject.toml` | **Modify** — add nltk, datasets dependencies |

---

## Task 1: PatternVariant dataclass

**Files:**
- Create: `hpm_ai_v5/core/variant.py`
- Create: `tests/test_pattern_variant.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pattern_variant.py
from hpm_ai_v5.core.variant import PatternVariant, make_variant

def test_pattern_variant_creation():
    v = PatternVariant(
        name="variant_0",
        member_names=["pattern_1", "pattern_2"],
        centroid=(1.0, 2.0, 3.0),
        hit_count=10,
        context_signature={"domain": "nlp"},
        score=0.75,
    )
    assert v.name == "variant_0"
    assert v.member_names == ["pattern_1", "pattern_2"]
    assert v.centroid == (1.0, 2.0, 3.0)
    assert v.hit_count == 10
    assert v.score == 0.75

def test_make_variant_centroid():
    from hpm_ai_v5.core.pattern import Pattern
    p1 = Pattern(name="p1", template=(1.0, 2.0), support=3, utility=0.8)
    p2 = Pattern(name="p2", template=(1.2, 2.2), support=5, utility=0.6)
    v = make_variant([p1, p2], name="variant_0")
    assert v.member_names == ["p1", "p2"]
    assert abs(v.centroid[0] - 1.1) < 1e-6
    assert abs(v.centroid[1] - 2.1) < 1e-6
    assert v.hit_count == 8
```

- [ ] **Step 2: Run test — verify it fails**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: ModuleNotFoundError — variant.py does not exist yet.

- [ ] **Step 3: Create hpm_ai_v5/core/variant.py**

```python
"""PatternVariant — promoted abstraction over near-identical patterns."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .pattern import Pattern


@dataclass
class PatternVariant:
    name: str
    member_names: list[str]
    centroid: tuple[float, ...]
    hit_count: int
    context_signature: dict[str, Any]
    score: float


def make_variant(patterns: list[Pattern], name: str) -> PatternVariant:
    templates = [p.template for p in patterns if p.template]
    if not templates:
        centroid: tuple[float, ...] = ()
    else:
        min_len = min(len(t) for t in templates)
        arr = np.array([list(t[:min_len]) for t in templates], dtype=float)
        centroid = tuple(float(x) for x in arr.mean(axis=0))
    hit_count = sum(p.support for p in patterns)
    ctx: dict[str, Any] = {}
    for p in patterns:
        if hasattr(p, "context") and p.context:
            ctx.update(p.context)
    score = float(np.mean([p.utility for p in patterns]))
    return PatternVariant(
        name=name,
        member_names=[p.name for p in patterns],
        centroid=centroid,
        hit_count=hit_count,
        context_signature=ctx,
        score=score,
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v5/core/variant.py tests/test_pattern_variant.py
git commit -m "feat: add PatternVariant dataclass and make_variant factory"
```

---

## Task 2: CoreConfig consolidation_threshold

**Files:**
- Modify: `hpm_ai_v5/core/config.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_pattern_variant.py`:

```python
def test_core_config_has_consolidation_threshold():
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig()
    assert hasattr(config, "consolidation_threshold")
    assert config.consolidation_threshold == 0.8

def test_core_config_custom_threshold():
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(consolidation_threshold=0.5)
    assert config.consolidation_threshold == 0.5
```

- [ ] **Step 2: Run test — verify it fails**

```bash
uv run pytest tests/test_pattern_variant.py::test_core_config_has_consolidation_threshold -v
```
Expected: AssertionError — attribute does not exist.

- [ ] **Step 3: Add field to CoreConfig**

In `hpm_ai_v5/core/config.py`, add after `max_sequences`:

```python
    consolidation_threshold: float = 0.8
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v5/core/config.py tests/test_pattern_variant.py
git commit -m "feat: add consolidation_threshold to CoreConfig (default 0.8)"
```

---

## Task 3: PatternStore variants dict and register_variant

**Files:**
- Modify: `hpm_ai_v5/core/store.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pattern_variant.py`:

```python
def test_store_has_variants_dict():
    from hpm_ai_v5.core.store import PatternStore
    store = PatternStore()
    assert hasattr(store, "variants")
    assert isinstance(store.variants, dict)

def test_store_register_variant():
    from hpm_ai_v5.core.store import PatternStore
    from hpm_ai_v5.core.variant import PatternVariant
    store = PatternStore()
    v = PatternVariant(
        name="v0", member_names=["p1"], centroid=(1.0,),
        hit_count=3, context_signature={}, score=0.5,
    )
    store.register_variant(v)
    assert "v0" in store.variants
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_pattern_variant.py::test_store_has_variants_dict tests/test_pattern_variant.py::test_store_register_variant -v
```
Expected: AttributeError.

- [ ] **Step 3: Add variants to PatternStore**

In `hpm_ai_v5/core/store.py`, add import at top:
```python
from .variant import PatternVariant
```

Add field to PatternStore dataclass (after `_pattern_index`):
```python
    variants: dict[str, PatternVariant] = field(default_factory=dict, init=False)
```

Add method after `get()`:
```python
    def register_variant(self, variant: PatternVariant) -> None:
        self.variants[variant.name] = variant
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v5/core/store.py tests/test_pattern_variant.py
git commit -m "feat: add variants dict and register_variant to PatternStore"
```

---

## Task 4: PatternStore variant-level fallback in match()

**Files:**
- Modify: `hpm_ai_v5/core/store.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_pattern_variant.py`:

```python
def test_store_match_falls_back_to_variant():
    from hpm_ai_v5.core.store import PatternStore
    from hpm_ai_v5.core.variant import PatternVariant
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(near_threshold=2.0, max_patterns=32)
    store = PatternStore(config=config)
    v = PatternVariant(
        name="v0", member_names=[], centroid=(1.0, 2.0),
        hit_count=5, context_signature={}, score=0.8,
    )
    store.register_variant(v)
    result = store.match((1.1, 2.1))
    assert result.status == "variant"
    assert result.pattern is None
    assert result.variant is not None
    assert result.variant.name == "v0"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
uv run pytest tests/test_pattern_variant.py::test_store_match_falls_back_to_variant -v
```
Expected: AttributeError — MatchResult has no variant field.

- [ ] **Step 3: Update MatchResult**

In `hpm_ai_v5/core/store.py`, update MatchResult:
```python
@dataclass(frozen=True, slots=True)
class MatchResult:
    status: str
    pattern: Pattern | None
    distance: float
    residual: tuple[Any, ...] = field(default_factory=tuple)
    variant: PatternVariant | None = None
```

Add `import numpy as np` at top of store.py if not already present.

At the end of `match()`, before the final `return MatchResult(status="novel", ...)`:
```python
        if self.variants:
            best_variant = None
            best_vdist = float("inf")
            for v in self.variants.values():
                if not v.centroid:
                    continue
                c = candidate[:len(v.centroid)]
                vdist = float(np.linalg.norm(
                    np.array(c, dtype=float) - np.array(v.centroid, dtype=float)
                ))
                if vdist < best_vdist:
                    best_vdist = vdist
                    best_variant = v
            nt = self.near_threshold or self.config.near_threshold
            if best_variant is not None and best_vdist <= nt:
                return MatchResult(
                    status="variant", pattern=None,
                    distance=best_vdist, residual=candidate,
                    variant=best_variant,
                )
```

- [ ] **Step 4: Run all variant tests**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Verify no regressions**

```bash
uv run pytest tests/ -v --ignore=tests/test_atis_benchmark.py --ignore=tests/test_wordnet_kb.py 2>&1 | tail -20
```
Expected: no failures.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v5/core/store.py tests/test_pattern_variant.py
git commit -m "feat: variant-level fallback in PatternStore.match() returns status=variant"
```

---

## Task 5: PatternManager _consolidate_variants

**Files:**
- Modify: `hpm_ai_v5/core/pattern_manager.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pattern_variant.py`:

```python
def test_consolidation_promotes_near_duplicates():
    from hpm_ai_v5.core import PatternEngine, PatternManager
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(
        max_patterns=10, near_threshold=0.5, consolidation_threshold=0.5,
    )
    engine = PatternEngine(config=config)
    manager = PatternManager(promotion_threshold=0.01)
    for i in range(6):
        engine.store.learn((1.0 + i * 0.05, 2.0 + i * 0.05))
    manager.start_episode(engine)
    manager.end_episode(engine)
    assert len(engine.store.variants) >= 1

def test_consolidation_retains_concrete_patterns():
    from hpm_ai_v5.core import PatternEngine, PatternManager
    from hpm_ai_v5.core.config import CoreConfig
    config = CoreConfig(max_patterns=10, near_threshold=0.5, consolidation_threshold=0.5)
    engine = PatternEngine(config=config)
    manager = PatternManager(promotion_threshold=0.01)
    for i in range(6):
        engine.store.learn((1.0 + i * 0.05, 2.0 + i * 0.05))
    count_before = len(engine.store.patterns)
    manager.start_episode(engine)
    manager.end_episode(engine)
    assert len(engine.store.patterns) == count_before
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_pattern_variant.py::test_consolidation_promotes_near_duplicates tests/test_pattern_variant.py::test_consolidation_retains_concrete_patterns -v
```
Expected: AssertionError — no variants created.

- [ ] **Step 3: Add _consolidate_variants to PatternManager**

In `hpm_ai_v5/core/pattern_manager.py`, add import at top:
```python
from .variant import make_variant
```

Add method to PatternManager:
```python
    def _consolidate_variants(self, engine: "PatternEngine") -> int:
        threshold = engine.config.max_patterns * engine.config.consolidation_threshold
        if len(engine.store.patterns) < threshold:
            return 0
        patterns = list(engine.store.patterns)
        used: set[str] = set()
        promoted = 0
        for i, p1 in enumerate(patterns):
            if p1.name in used:
                continue
            cluster = [p1]
            for p2 in patterns[i + 1:]:
                if p2.name in used:
                    continue
                dist = p1.distance(
                    p2.template,
                    canonicalization_mode=engine.store.canonicalization_mode,
                    distance_scale=engine.store.distance_scale or 1.0,
                )
                if dist < engine.config.near_threshold:
                    cluster.append(p2)
                    used.add(p2.name)
            if len(cluster) >= 2:
                used.add(p1.name)
                vname = f"variant_{len(engine.store.variants)}"
                engine.store.register_variant(make_variant(cluster, name=vname))
                promoted += 1
        return promoted
```

In `end_episode()`, after `promoted_names = self.promote_from(engine, context=context)`:
```python
        consolidated = self._consolidate_variants(engine)
```

Add `"consolidated": consolidated` to the returned dict.

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_pattern_variant.py -v
```
Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v5/core/pattern_manager.py tests/test_pattern_variant.py
git commit -m "feat: PatternManager._consolidate_variants promotes near-duplicates at episode end"
```

---

## Task 6: Export PatternVariant from core

**Files:**
- Modify: `hpm_ai_v5/core/__init__.py`

- [ ] **Step 1: Add export**

In `hpm_ai_v5/core/__init__.py`, add:
```python
from .variant import PatternVariant
```
Add `"PatternVariant"` to `__all__`.

- [ ] **Step 2: Verify**

```bash
uv run python -c "from hpm_ai_v5.core import PatternVariant; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_atis_benchmark.py --ignore=tests/test_wordnet_kb.py 2>&1 | tail -20
```
Expected: no failures.

- [ ] **Step 4: Commit**

```bash
git add hpm_ai_v5/core/__init__.py
git commit -m "feat: export PatternVariant from hpm_ai_v5.core"
```

---

## Task 7: WordNet-backed KnowledgeBaseLookup

**Files:**
- Modify: `hpm_ai_v5/adapter/nlp.py`
- Modify: `pyproject.toml`
- Create: `tests/test_wordnet_kb.py`

- [ ] **Step 1: Install NLTK and download WordNet**

```bash
uv add nltk
uv run python -m nltk.downloader wordnet omw-1.4
```
Expected: `[nltk_data] Done downloading collection omw-1.4`

- [ ] **Step 2: Write failing tests**

```python
# tests/test_wordnet_kb.py
from hpm_ai_v5.adapter.nlp import KnowledgeBaseLookup
from hpm_ai_v5.adapter.packet import AdapterPacket

def _make_packet(tokens):
    return AdapterPacket(raw=" ".join(tokens), context={"tokens": tokens}, states=[])

def test_wordnet_kb_returns_synonyms_for_known_word():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["flight"]))
    candidates = result.context.get("semantic_candidates", [])
    assert len(candidates) > 0

def test_wordnet_kb_caps_at_five():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["run"]))
    candidates = result.context.get("semantic_candidates", [])
    assert len(candidates) <= 5

def test_wordnet_kb_unknown_word_returns_empty():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["xyzzy123abc"]))
    candidates = result.context.get("semantic_candidates", [])
    assert candidates == []
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/test_wordnet_kb.py -v
```
Expected: tests fail — static KB has no entry for "flight".

- [ ] **Step 4: Replace KnowledgeBaseLookup with WordNet implementation**

In `hpm_ai_v5/adapter/nlp.py`, replace the `KnowledgeBaseLookup` class entirely:

```python
@dataclass(slots=True)
class KnowledgeBaseLookup(Adapter):
    """WordNet-backed synonym lookup for semantic candidate views."""

    name: str = "kb_lookup"
    max_candidates: int = 5
    requires: list[str] = field(default_factory=lambda: ["nlp_tokenizer"])
    provides: list[str] = field(default_factory=lambda: ["semantic_candidates"])

    def _synonyms(self, token: str) -> list[str]:
        try:
            from nltk.corpus import wordnet
            syns: set[str] = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ").lower()
                    if name != token:
                        syns.add(name)
            return list(syns)[: self.max_candidates]
        except Exception:
            return []

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = packet.context.get("tokens", [])
        all_candidates: list[str] = []
        for token in tokens:
            all_candidates.extend(self._synonyms(token.lower()))
        packet.context["semantic_candidates"] = list(set(all_candidates))
        return packet
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/test_wordnet_kb.py -v
```
Expected: 3 PASSED.

- [ ] **Step 6: Verify SNLP still passes**

```bash
uv run python -m hpm_ai_v5.experiments.run_snlp_benchmark 2>&1 | grep "^T[1-5]:"
```
Expected: T2 >= 80%, T3 = 100%, T5 = 100%.

- [ ] **Step 7: Commit**

```bash
git add hpm_ai_v5/adapter/nlp.py tests/test_wordnet_kb.py pyproject.toml uv.lock
git commit -m "feat: replace static KnowledgeBaseLookup with WordNet synonyms (cap 5)"
```

---

## Task 8: ATIS adapter and dataset loader

**Files:**
- Create: `hpm_ai_v5/adapter/atis.py`
- Modify: `pyproject.toml`
- Create: `tests/test_atis_benchmark.py`

- [ ] **Step 1: Install datasets**

```bash
uv add datasets
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_atis_benchmark.py
def test_atis_loads_train_split():
    from hpm_ai_v5.adapter.atis import load_atis
    train, test = load_atis()
    assert len(train) > 1000
    assert len(test) > 100
    assert "text" in train[0]
    assert "intent" in train[0]

def test_intent_label_adapter_injects_label():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label="flight")
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert result.context.get("intent_label") == "flight"

def test_intent_label_adapter_withheld_in_inference():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label=None)
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert "intent_label" not in result.context
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/test_atis_benchmark.py -v
```
Expected: ImportError — atis.py does not exist.

- [ ] **Step 4: Create hpm_ai_v5/adapter/atis.py**

```python
"""ATIS dataset loader and IntentLabelAdapter for HPM v5."""
from __future__ import annotations
from dataclasses import dataclass, field
from .base import Adapter
from .packet import AdapterPacket


def load_atis() -> tuple[list[dict], list[dict]]:
    """Return (train, test) as lists of {text, intent} dicts."""
    from datasets import load_dataset
    ds = load_dataset("tuetschek/atis", trust_remote_code=True)

    def _extract(split: str) -> list[dict]:
        return [{"text": row["text"], "intent": row["intent"]} for row in ds[split]]

    return _extract("train"), _extract("test")


@dataclass(slots=True)
class IntentLabelAdapter(Adapter):
    """Inject gold intent label during training; omit during inference."""

    name: str = "intent_label"
    label: str | None = None
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["intent_label"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if self.label is not None:
            packet.context["intent_label"] = self.label
        return packet
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/test_atis_benchmark.py -v
```
Expected: 3 PASSED (dataset downloaded on first run).

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v5/adapter/atis.py tests/test_atis_benchmark.py pyproject.toml uv.lock
git commit -m "feat: ATIS dataset loader and IntentLabelAdapter"
```

---

## Task 9: ATIS benchmark harness

**Files:**
- Create: `hpm_ai_v5/experiments/run_atis_benchmark.py`

- [ ] **Step 1: Create the harness**

```python
"""ATIS Intent Recognition Benchmark (B1-B4) for HPM v5."""
from __future__ import annotations
import random
from collections import defaultdict
import numpy as np

from hpm_ai_v5.adapter.nlp import (
    NLPTokenizer, CanonicalPhraser, SkeletonExtractor,
    SkeletonNgramAdapter, KnowledgeBaseLookup,
)
from hpm_ai_v5.adapter.atis import load_atis
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import PatternEngine, PatternManager, PatternStore
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.polygraphs.nlp import NLPPolygraphGenerator


class ATISBenchmark:
    def __init__(self, consolidation: bool = True):
        self.config = CoreConfig(
            max_patterns=2048,
            max_sequences=512,
            history_limit=100,
            near_threshold=0.4,
            consolidation_threshold=0.8,
        )
        self.engine = PatternEngine(config=self.config)
        self.manager = PatternManager(promotion_threshold=0.01)
        self.consolidation = consolidation
        self.pipeline = HPMPipeline(
            preprocessor=NLPTokenizer(),
            engine=self.engine,
            postprocessor=ValidationOnlyAdapter(),
            polygraph_generator=NLPPolygraphGenerator(),
            polygraph_confidence_skip=1.1,
        )
        self.pipeline.register_preprocessor(CanonicalPhraser())
        self.pipeline.register_preprocessor(SkeletonExtractor())
        self.pipeline.register_preprocessor(SkeletonNgramAdapter())
        self.pipeline.register_preprocessor(KnowledgeBaseLookup())
        self.intent_patterns: dict[str, list[str]] = defaultdict(list)

    def _reset(self):
        self.engine.current_state = None
        self.engine.history = []
        self.pipeline.view_engines.clear()
        self.engine.store = PatternStore(config=self.config)
        self.intent_patterns.clear()
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()

    def _train_utterance(self, text: str, intent: str):
        self.engine.current_state = None
        self.engine.history = []
        self.pipeline.step(text)
        if self.engine.last_match and self.engine.last_match.pattern:
            self.intent_patterns[intent].append(self.engine.last_match.pattern.name)

    def _predict_intent(self) -> str | None:
        match = self.engine.last_match
        if match is None:
            return None
        if match.pattern:
            for intent, names in self.intent_patterns.items():
                if match.pattern.name in names:
                    return intent
        if match.status == "variant" and match.variant:
            for member in match.variant.member_names:
                for intent, names in self.intent_patterns.items():
                    if member in names:
                        return intent
        return None

    def run_b1(self, train: list[dict], test: list[dict]) -> float:
        print("\nB1: Intent Recognition...")
        self._reset()
        random.shuffle(train)
        train_set = train[:int(len(train) * 0.8)]
        self.manager.start_episode(self.engine)
        for item in train_set:
            self._train_utterance(item["text"], item["intent"])
        if self.consolidation:
            self.manager.end_episode(self.engine)
        correct = sum(
            1 for item in test
            if self._run_and_predict(item["text"]) == item["intent"]
        )
        acc = correct / len(test)
        print(f"  Accuracy: {acc:.2%} ({correct}/{len(test)})")
        return acc

    def _run_and_predict(self, text: str) -> str | None:
        self.engine.current_state = None
        self.engine.history = []
        self.pipeline.step(text)
        return self._predict_intent()

    def run_b2(self, train: list[dict], test: list[dict]) -> float:
        print("\nB2: Slot Generalisation...")
        train_tokens: set[str] = set()
        for item in train:
            train_tokens.update(item["text"].lower().split())
        novel = [i for i in test if any(w not in train_tokens for w in i["text"].lower().split())]
        if not novel:
            print("  No novel-token items — skipping")
            return 0.0
        correct = sum(1 for item in novel if self._run_and_predict(item["text"]) == item["intent"])
        acc = correct / len(novel)
        print(f"  Accuracy: {acc:.2%} ({correct}/{len(novel)} novel-token items)")
        return acc

    def run_b3(self, train: list[dict]) -> dict:
        print("\nB3: Consolidation Effectiveness...")
        subset = train[:1000]
        # Without consolidation
        b_no = ATISBenchmark(consolidation=False)
        b_no._reset()
        for item in subset:
            b_no._train_utterance(item["text"], item["intent"])
        size_no = len(b_no.engine.store.patterns)
        # With consolidation
        b_yes = ATISBenchmark(consolidation=True)
        b_yes._reset()
        b_yes.manager.start_episode(b_yes.engine)
        for item in subset:
            b_yes._train_utterance(item["text"], item["intent"])
        b_yes.manager.end_episode(b_yes.engine)
        size_yes = len(b_yes.engine.store.patterns)
        variants = len(b_yes.engine.store.variants)
        reduction = (size_no - size_yes) / max(size_no, 1)
        print(f"  Without consolidation: {size_no} patterns")
        print(f"  With consolidation:    {size_yes} patterns, {variants} variants")
        print(f"  Reduction: {reduction:.1%}")
        return {"size_without": size_no, "size_with": size_yes, "variants": variants, "reduction": reduction}

    def run_b4(self, test: list[dict]) -> float:
        print("\nB4: Variant Match Rate...")
        variant_hits = concrete_hits = 0
        for item in test:
            self.engine.current_state = None
            self.engine.history = []
            self.pipeline.step(item["text"])
            pred = self._predict_intent()
            if pred == item["intent"]:
                match = self.engine.last_match
                if match and match.status == "variant":
                    variant_hits += 1
                else:
                    concrete_hits += 1
        total = variant_hits + concrete_hits
        rate = variant_hits / max(total, 1)
        print(f"  Concrete correct: {concrete_hits}, Variant correct: {variant_hits}")
        print(f"  Variant contribution: {rate:.2%}")
        return rate

    def run_all(self):
        print("Loading ATIS...")
        train, test = load_atis()
        print(f"  Train: {len(train)}, Test: {len(test)}")
        b1 = self.run_b1(train, test)
        b2 = self.run_b2(train, test)
        b3 = self.run_b3(train)
        b4 = self.run_b4(test)
        print("\n" + "=" * 40)
        print("ATIS RESULTS")
        print("=" * 40)
        print(f"B1 Intent Accuracy:      {b1:.2%}  (target >60%)")
        print(f"B2 Slot Generalisation:  {b2:.2%}  (target >70%)")
        print(f"B3 Store Reduction:      {b3['reduction']:.1%}  (target >30%)")
        print(f"B4 Variant Rate:         {b4:.2%}  (target >0%)")
        print("=" * 40)


if __name__ == "__main__":
    ATISBenchmark().run_all()
```

- [ ] **Step 2: Run smoke test**

```bash
uv run python -m hpm_ai_v5.experiments.run_atis_benchmark 2>&1 | tail -10
```
Expected: benchmark runs to completion, prints B1-B4 summary.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v5/experiments/run_atis_benchmark.py
git commit -m "feat: ATIS benchmark harness B1-B4 with PatternVariant integration"
```

---

## Task 10: Final validation

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -v 2>&1 | tail -30
```
Expected: all PASSED.

- [ ] **Step 2: SNLP regression check**

```bash
for i in 1 2 3; do uv run python -m hpm_ai_v5.experiments.run_snlp_benchmark 2>&1 | grep "^T[1-5]:"; echo "---"; done
```
Expected: T2 >= 80%, T3 = 100%, T5 = 100% across all 3 runs.

- [ ] **Step 3: ATIS benchmark**

```bash
uv run python -m hpm_ai_v5.experiments.run_atis_benchmark 2>&1 | tail -10
```
Expected: B3 reduction > 0%, B4 variant rate > 0% (confirms PatternVariant is active).

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "feat: PatternVariant consolidation + ATIS benchmark complete"
```
