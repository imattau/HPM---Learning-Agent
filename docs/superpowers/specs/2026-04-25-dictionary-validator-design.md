# DictionaryValidator — Design Spec

**Date:** 2026-04-25
**Branch:** hpm-ai-v4-dev
**Status:** Draft

---

## Goal

Add a pluggable, read-only dictionary module that `Reasoner` and `InstitutionalField` can optionally use for lexical awareness during planning and evaluation. Core learning (online EM, replicator dynamics) is entirely unaffected when the dictionary is absent.

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `hpm_ai_v4/dictionary.py` | CREATE | `DictionaryValidator` ABC + `SimpleWordList` implementation |
| `hpm_ai_v4/agents/reasoning.py` | MODIFY | `Reasoner.__init__` accepts `dictionary=None`; `plan()` adds word-boundary bonus |
| `hpm_ai_v4/field.py` | MODIFY | `InstitutionalField.evaluate()` accepts `dictionary=None`; adds `lexical_bonus` |
| `hpm_ai_v4/tests/test_dictionary.py` | CREATE | Unit tests for all of the above |

---

## DictionaryValidator ABC

```python
from abc import ABC, abstractmethod
from typing import List

class DictionaryValidator(ABC):
    @abstractmethod
    def contains(self, word: str) -> bool: ...

    @abstractmethod
    def is_prefix(self, prefix: str) -> bool: ...

    @abstractmethod
    def completions(self, prefix: str, max_suggestions: int = 5) -> List[str]: ...

    @abstractmethod
    def score_word(self, word: str) -> float: ...
```

All methods operate on normalised lowercase strings.

---

## SimpleWordList

- **Storage:** Trie (nested `dict`) for O(L) prefix/membership checks where L = word length
- **Load sources:** `word_set` (a Python `set[str]`) or `filepath` (one word per line)
- **Normalisation:** lowercase on insert and on every query
- **Completions:** depth-first traversal of the trie capped at `max_suggestions`
- **score_word:** returns `1.0` if the word is in the dictionary, `0.0` otherwise
- **Empty construction:** `SimpleWordList()` with no arguments produces a valid but empty validator

---

## Reasoner Integration

### Signature changes

```python
# Before
def __init__(self, agent):

# After
def __init__(self, agent, dictionary=None):
    ...
    self.dictionary = dictionary
```

```python
# Before
def plan(self, goal_state: int, horizon: int = 5, num_rollouts: int = 10) -> List[int]:

# After
def plan(self, goal_state: int, horizon: int = 5, num_rollouts: int = 10,
         use_dictionary: bool = True) -> List[int]:
```

### Behaviour in plan()

After scoring each rollout by goal proximity (`score = -abs(seq[-1] - goal_state)`), if `self.dictionary` is set and `use_dictionary=True`, add a small word-boundary bonus:

```python
if self.dictionary and use_dictionary:
    # Obs class 2 = space. If the last obs before a space (or the
    # final obs) was class 0 (letter) and the run length > 2,
    # award a word-boundary bonus of 0.1.
    word_bonus = 0.1 if len(seq) > 2 and seq[-1] != 2 else 0.0
    score += word_bonus
```

### Critical limitation: char-class observations

`HPMAgent.obs_buffer` contains **character class IDs** (0–4), not actual character IDs, because the pipeline uses `CharClassAdapter.encode()` before storing observations. The classes are:

| Class | Meaning |
|-------|---------|
| 0 | letter (A–Z, a–z) — any letter, unknown which |
| 1 | digit (0–9) |
| 2 | space |
| 3 | punctuation |
| 4 | newline |

This means it is **impossible** to reconstruct an actual word from `obs_buffer`. A sequence `[0, 0, 0, 2]` tells us "three letters then a space", but not which three letters. Therefore:

- **Prefix pruning of specific letter sequences is not possible** in `plan()`.
- **Full word validation (actual word check)** is not possible in `plan()`.
- What is possible: detecting **word boundaries** (class 2 after a run of class 0) and awarding a generic bonus for "a word-like sequence was produced".
- True lexical validation requires a pipeline that operates on raw character IDs (e.g., via `TextAdapter`, `obs_dim=256`) rather than char-class IDs.

This limitation must be clearly communicated to callers via the `plan()` docstring.

### explain()

If `self.dictionary` is set, `explain()` can append a note:

```python
# In explain():
if self.dictionary:
    return base_description + " Dictionary active: rewards word-like sequences at boundaries."
```

---

## InstitutionalField Integration

`InstitutionalField.evaluate()` exists (confirmed in `hpm_ai_v4/field.py`) and returns a float bonus/penalty. It is appropriate to add an optional `lexical_bonus` here.

### Signature change

```python
# Before
def evaluate(self, pattern, obs_seq):

# After
def evaluate(self, pattern, obs_seq, dictionary=None):
```

### Behaviour

After computing the existing `success_rate`-based return value, if `dictionary` is provided and the `obs_seq` contains at least one space boundary (class 2) separating letter runs:

```python
if dictionary is not None:
    # Count word-boundary events: transitions [letter-run ... space]
    word_boundaries = sum(
        1 for i in range(1, len(obs_seq))
        if obs_seq[i] == 2 and obs_seq[i-1] == 0
    )
    lexical_bonus = 0.05 * min(word_boundaries, 4)  # cap at 0.20
    return existing_return + lexical_bonus
```

This reflects the same char-class limitation: we reward word-boundary frequency, not actual word validity.

### Note on YAGNI

`InstitutionalField` already exists with `evaluate()`. Adding an optional `dictionary` parameter is safe (backward-compatible, keyword-only in practice). No new class is created.

---

## What NOT to do

- Do NOT modify online EM update, replicator dynamics, or `meta_pattern_update`.
- Do NOT make `dictionary` mandatory in any constructor or method.
- Do NOT attempt to decode char-class IDs back to actual characters — this is architecturally impossible without the original character input.
- Do NOT create `InstitutionalField` if it does not exist (it does exist; this note is for future readers of the spec).
- Do NOT add dictionary loading to `HPMAgent.__init__` — callers construct `Reasoner` directly when they need lexical awareness.

---

## Practical Use Cases

The dictionary is most useful when:

1. **Raw character pipeline**: The agent uses `TextAdapter` (obs_dim=256) rather than `CharClassAdapter` (obs_dim=5). In this case actual character IDs are stored in `obs_buffer` and the full `contains()`/`is_prefix()` API becomes meaningful.
2. **TextReasoningInterface / word_complete()**: Any interface that already decodes char classes to class names can use `score_word()` after reconstructing a word from decoded output.
3. **Word-boundary density metric**: Even with char-class obs, `evaluate()` can use word-boundary count as a proxy for "word-like production fluency".

---

## Design Principles (HPM alignment)

- **Pattern evaluator**: `DictionaryValidator` acts as a lightweight evaluator/gatekeeper — it scores candidate sequences without modifying pattern substrates or dynamics.
- **Pattern field**: Plugging a dictionary into `InstitutionalField` is consistent with HPM's notion of the social/environmental field shaping which patterns are reinforced.
- **Optional prior**: The word list represents an innate prior (lexical knowledge) that individual learning refines — consistent with HPM's stance that agents need not start as blank slates.
