# Wikipedia Character-Stream Simulation — Design Spec

**Date**: 2026-04-24
**Branch**: hpm-ai-v4-dev
**Status**: Approved for implementation

---

## 1. Goal

Validate that HPM v4 can learn hierarchical linguistic structure from a raw character stream and expose that structure through programmatic reasoning queries. Specifically:

- An `HPMAgent` trained on Wikipedia text (character IDs) should achieve next-character prediction accuracy above 50% (versus 1/95 baseline).
- The learned patterns should support word completion, planning to word boundaries, counterfactual analysis, and self-explanation — all via programmatic (char-ID) interfaces, not natural language.

This is a *validation* simulation, not a production system. YAGNI applies throughout.

---

## 2. Character Vocabulary

The vocabulary covers printable ASCII plus newline:

| Symbol set | Range | Count |
|---|---|---|
| Printable ASCII | 32–126 inclusive | 95 |
| Newline | `\n` (ASCII 10) | 1 |
| **Total** | | **96** |

Mapping:

```
char_to_id['\n'] = 0
char_to_id[chr(32)] = 1   # space
char_to_id[chr(33)] = 2
...
char_to_id[chr(126)] = 95
```

Space character is ID 1. This is the word-boundary sentinel used by planning and word completion.

`VOCAB_SIZE = 96`

---

## 3. WikipediaStream Class

**File**: `hpm_ai_v4/simulations/wikipedia_sim.py`

```python
class WikipediaStream:
    def __init__(self, filepath: str):
        ...
    def __iter__(self) -> Iterator[int]:
        # yields char IDs, loops forever when file exhausted
        ...
    @staticmethod
    def char_to_id(ch: str) -> int:
        ...
    @staticmethod
    def id_to_char(i: int) -> str:
        ...
```

Behaviour:
- Reads the file as UTF-8, ignoring characters outside the vocabulary (keep only newlines and ASCII 32–126).
- Yields one integer (char ID) per `__next__` call.
- When the file is exhausted, seeks back to position 0 and continues (infinite loop).
- `char_to_id` and `id_to_char` are pure functions, no state.

---

## 4. Pattern Architecture

### Decision: Use existing `HierarchicalPattern` as-is (YAGNI)

The existing `HierarchicalPattern` in `hpm_ai_v4/pattern.py` is a 3-level HMM (z3→z2→z1→x). It accepts `latent_dim` and `obs_dim` as constructor arguments. This is sufficient for the validation goal.

The user-provided design mentioned a 3-level hierarchy (z3=topic K=8, z2=phrase K=16, z1=char-class K=16). We implement this as:

```python
HierarchicalPattern(pattern_id=i, latent_dim=16, obs_dim=96)
```

The `latent_dim=16` covers both z2 and z1 levels within the existing class. The `obs_dim=96` matches the vocabulary. No new pattern class is introduced.

**Rationale**: adding a purpose-built 3-level class would duplicate forward-backward logic before the simulation validates any learning. If the simulation succeeds and the 3-level separation proves necessary, a `ThreeLevelPattern` can be extracted then.

### Population initialisation

```python
num_initial_patterns = 5
HPMAgent(num_initial_patterns=5, obs_dim=96)
```

The agent initialises 5 `HierarchicalPattern` instances plus one flat baseline, all with `obs_dim=96`.

---

## 5. Training Loop

**File**: `hpm_ai_v4/simulations/wikipedia_sim.py` (function `run_simulation`)

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| `total_chars` | 100,000 | Training budget |
| `window` | 100 | `obs_buffer` max length (already enforced by `HPMAgent`) |
| `decay` | 0.995 | Replicator decay per step — override default 0.005 |
| `eta` | 0.05 | Replicator update step size |
| `recombination_every` | 500 | Override agent's default 20 |
| `log_every` | 1,000 | Log metrics to stdout |

### How to override decay/eta

`meta_pattern_update` in `hpm_ai_v4/operators/dynamics.py` accepts `eta` and `decay` keyword arguments. The agent calls it in `perceive_and_learn`. To use custom values, subclass `HPMAgent` or monkey-patch before training:

```python
from hpm_ai_v4.operators import dynamics as dyn
_orig = dyn.meta_pattern_update
dyn.meta_pattern_update = lambda *a, **kw: _orig(*a, eta=0.05, decay=0.005, **{k:v for k,v in kw.items() if k not in ('eta','decay')})
```

Alternatively, pass `eta` and `decay` as constructor overrides if the agent is extended. The plan will use the simpler subclass approach.

### Training loop pseudocode

```python
stream = WikipediaStream(corpus_path)
agent = HPMAgent(num_initial_patterns=5, obs_dim=96)
stream_iter = iter(stream)

for step in range(total_chars):
    char_id = next(stream_iter)
    agent.perceive_and_learn(char_id)
    
    if step % log_every == 0:
        log_metrics(agent, step)
```

### Online EM inside the agent

`HPMAgent.perceive_and_learn` already calls:
- `p.observe(obs)` — single-step sufficient-statistics decay
- `p.adapt(obs_buffer[-20:])` — full forward-backward EM on last 20 chars
- `meta_pattern_update` — replicator dynamics

The `adapt` window of 20 chars is the effective EM window for each update step. The sliding `obs_buffer` of 100 chars provides recency context for scoring.

---

## 6. TextReasoningInterface

**File**: `hpm_ai_v4/simulations/text_reasoning.py`

Wraps `Reasoner` to handle string↔char-ID conversion. All internal calls use integer char IDs. The public API accepts and returns Python strings for human convenience, but no natural-language generation is involved.

```python
VOCAB_SIZE = 96

class TextReasoningInterface:
    def __init__(self, agent: HPMAgent):
        self.agent = agent
        self.reasoner = agent.reasoner  # Reasoner instance

    def encode(self, text: str) -> List[int]:
        """Convert string to list of char IDs, skipping out-of-vocab chars."""
        ...

    def decode(self, ids: List[int]) -> str:
        """Convert list of char IDs to string."""
        ...

    def next_char_predict(self, prefix: str) -> List[Tuple[str, float]]:
        """
        Return the top-5 predicted next characters with their probabilities.

        Args:
            prefix: string context

        Returns:
            List of (char, probability) tuples, sorted descending by probability,
            length exactly 5. Probabilities sum to <= 1.0 (they are marginals of
            the full 96-dim distribution).
        """
        obs_seq = self.encode(prefix)
        patterns = self.reasoner.get_relevant_patterns(obs_seq, top_k=5)
        dist = self.reasoner.compose_predictions(patterns, obs_seq)  # shape (96,)
        dist = dist / (dist.sum() + 1e-12)
        top5_ids = np.argsort(dist)[-5:][::-1]
        return [(WikipediaStream.id_to_char(i), float(dist[i])) for i in top5_ids]

    def word_complete(self, prefix: str, max_chars: int = 10) -> str:
        """
        Greedily extend prefix until a space (ID 1) is predicted or max_chars reached.

        Args:
            prefix: string context (may end mid-word)
            max_chars: maximum characters to append

        Returns:
            Completed string (prefix + generated chars, stopping before space).
            The space itself is not appended.
        """
        obs_seq = self.encode(prefix)
        result = list(prefix)
        SPACE_ID = WikipediaStream.char_to_id(' ')

        for _ in range(max_chars):
            patterns = self.reasoner.get_relevant_patterns(obs_seq, top_k=5)
            dist = self.reasoner.compose_predictions(patterns, obs_seq)
            dist = dist / (dist.sum() + 1e-12)
            next_id = int(np.argmax(dist))
            if next_id == SPACE_ID:
                break
            result.append(WikipediaStream.id_to_char(next_id))
            obs_seq.append(next_id)
            if len(obs_seq) > 100:
                obs_seq = obs_seq[-100:]

        return ''.join(result)

    def plan_to_boundary(self, horizon: int, num_rollouts: int) -> str:
        """
        Use Reasoner.plan to find a char sequence reaching a space (word boundary).

        Args:
            horizon: planning depth (number of chars to look ahead)
            num_rollouts: number of stochastic rollouts

        Returns:
            Decoded string of the planned char sequence (may or may not end at space).
        """
        SPACE_ID = WikipediaStream.char_to_id(' ')
        seq = self.reasoner.plan(goal_state=SPACE_ID, horizon=horizon, num_rollouts=num_rollouts)
        return self.decode(seq)

    def counterfactual_shift(self, context: str, forced_char: str) -> List[Tuple[str, float]]:
        """
        Compute the predictive distribution after forcing a specific next character,
        and return the top-5 next-next-char predictions.

        Args:
            context: string context
            forced_char: the character to force as the next observation

        Returns:
            Top-5 (char, probability) tuples after the intervention, sorted descending.
        """
        obs_seq = self.encode(context)
        forced_id = WikipediaStream.char_to_id(forced_char)
        patterns = self.reasoner.get_relevant_patterns(obs_seq, top_k=5)

        intervened_dists = []
        for p in patterns:
            _orig, intervened = self.reasoner.counterfactual(p, obs_seq, forced_id)
            intervened_dists.append((p.weight, intervened))

        total_w = sum(w for w, _ in intervened_dists) + 1e-12
        blended = np.zeros(VOCAB_SIZE)
        for w, d in intervened_dists:
            # d may be shorter than VOCAB_SIZE if a flat pattern; pad
            blended[:len(d)] += (w / total_w) * d
        blended = blended / (blended.sum() + 1e-12)

        top5_ids = np.argsort(blended)[-5:][::-1]
        return [(WikipediaStream.id_to_char(i), float(blended[i])) for i in top5_ids]

    def explain_best_pattern(self) -> str:
        """
        Return a human-readable description of the highest-weight pattern.

        Returns:
            String description from Reasoner.explain.
        """
        if not self.agent.patterns:
            return "No patterns in population."
        best = max(self.agent.patterns, key=lambda p: p.weight)
        return self.reasoner.explain(best)
```

### Type summary

| Method | Input types | Return type |
|---|---|---|
| `encode` | `str` | `List[int]` |
| `decode` | `List[int]` | `str` |
| `next_char_predict` | `str` | `List[Tuple[str, float]]` (len=5) |
| `word_complete` | `str`, `int` | `str` |
| `plan_to_boundary` | `int`, `int` | `str` |
| `counterfactual_shift` | `str`, `str` | `List[Tuple[str, float]]` (len=5) |
| `explain_best_pattern` | — | `str` |

---

## 7. Metrics and Success Criteria

All metrics computed over a held-out 1,000-char evaluation window taken from a separate position in the corpus (not the training stream).

| Metric | Definition | Threshold |
|---|---|---|
| Prediction accuracy | Fraction of chars where `argmax(next_char_predict)` equals ground truth | > 50% |
| Word completion rate | Fraction of 100 sampled word prefixes where `word_complete` recovers the correct word | > 30% |
| Planning success | Fraction of `plan_to_boundary(horizon=5, num_rollouts=20)` calls where last char is space (ID 1) | > 60% |
| Counterfactual KL | KL(intervened \|\| unforced) averaged over 50 random contexts; measures that interventions shift the distribution | > 0.1 nats |
| Compression MI | `pattern.compression(obs_buffer)` on best pattern, measured at step 0 and step 100k | Increases from ~0 to > 0.2 nats |

Baseline for accuracy: uniform random over 96 chars = 1.04%. The 50% threshold is ambitious but achievable because English text has strong bigram/trigram structure.

---

## 8. Files

| File | Purpose |
|---|---|
| `hpm_ai_v4/simulations/wikipedia_sim.py` | `WikipediaStream` class + `run_simulation` function |
| `hpm_ai_v4/simulations/text_reasoning.py` | `TextReasoningInterface` class |
| `hpm_ai_v4/simulations/data/get_corpus.py` | Download Simple English Wikipedia sample |
| `hpm_ai_v4/tests/test_wikipedia_sim.py` | Unit + integration tests |

No existing files are modified. The simulation integrates with the existing `HPMAgent`, `Reasoner`, and `HierarchicalPattern` via their public APIs only.

---

## 9. Out of Scope

- Natural language query interface (all reasoning is programmatic char-ID in/out)
- GPU acceleration (numpy-only, consistent with existing codebase)
- Multi-agent social evaluators (single agent for this validation)
- 3-level bespoke HMM class (deferred unless 2-level proves insufficient)
- Tokenisation above character level
