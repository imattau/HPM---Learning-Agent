# ToolSelector Design
Date: 2026-04-22

## Problem
The agent fails NLP tool tasks not because it lacks the tools but because pattern
selection is driven purely by replicator weights — which have no semantic signal.
The agent can't connect "Count words in: the quick brown fox" to `str.split` or
"Is this positive?" to `TextBlob.sentiment` without many failed trials first.

## Solution
A `ToolSelector` class that uses `LanguageModelPattern` embeddings to compute
semantic similarity between the current task text and each population pattern's
description. This similarity score biases pattern selection weights in `act()` —
providing a short-term relevance signal alongside the replicator's long-term
fitness signal.

The LM acts as a **pattern evaluator** in HPM terms: one of the four required
roles, operating on semantic similarity rather than reward history.

## Architecture

```
act() in base_discovery.py
    ↓
weights = [p.weight for p in population]          # replicator fitness
    +
selector_bias = ToolSelector.score(task_text, population)  # semantic relevance
    ↓
combined = weights * (1 + alpha * selector_bias)  # alpha=0.5 default
    ↓
Pattern selected by combined probability
```

The replicator still drives long-term selection. The selector provides a
short-term boost to semantically relevant patterns — especially useful early
in an episode before the replicator has meaningful signal.

## ToolSelector

File: `hpm_ai_v3/tools/tool_selector.py`

```
ToolSelector
    lm: LanguageModelPattern        # embedding source
    alpha: float = 0.5              # bias strength
    _cache: Dict[str, List[float]]  # embed cache (avoids re-embedding same text)

    score(task_text, patterns) -> np.ndarray
        # Returns similarity score [0,1] per pattern
        # task_embed = lm._embed(task_text)
        # for each pattern: pattern_embed = lm._embed(pattern.tool_description)
        # similarity = cosine_similarity(task_embed, pattern_embed)

    apply(task_text, weights, patterns) -> np.ndarray
        # Returns adjusted weights: weights * (1 + alpha * score(...))
```

## Pattern Description

Each `ActionPattern` needs a `tool_description` property — a short text the LM
embeds to represent what the tool does. Derived automatically:

```python
@property
def tool_description(self) -> str:
    if self.module and self.function:
        return f"{self.module}.{self.function}: call {self.function} from {self.module}"
    return self.action_type
```

NLP-relevant patterns get richer descriptions via a small registry:
```python
TOOL_DESCRIPTIONS = {
    "textblob.TextBlob": "sentiment analysis polarity positive negative text",
    "re.findall": "extract pattern match numbers text regex",
    "str.split": "split words count tokenize text",
    "spacy.nlp": "entity noun parse sentence structure",
}
```

If the pattern's `tool_name` is in the registry, use that description instead.

## Integration Point

`base_discovery.py` line 155-158 — replace:
```python
weights = np.array([p.weight for p in self.population.patterns])
total = weights.sum()
probs = weights / total if total > 1e-6 else np.ones(len(weights)) / len(weights)
```

With:
```python
weights = np.array([p.weight for p in self.population.patterns])
if self.tool_selector and self.current_task:
    weights = self.tool_selector.apply(
        self.current_task.get("text", ""),
        weights,
        self.population.patterns
    )
total = weights.sum()
probs = weights / total if total > 1e-6 else np.ones(len(weights)) / len(weights)
```

`tool_selector` is set on the agent in `__init__` — defaults to `None` (no
behaviour change if LM not available).

## Cosine Similarity

```python
def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0
```

Similarity is clipped to [0, 1] before applying as bias (negative similarity = 0,
no penalty — selector only boosts, never suppresses).

## Files

- **CREATE**: `hpm_ai_v3/tools/tool_selector.py` — `ToolSelector` class + `TOOL_DESCRIPTIONS` registry
- **MODIFY**: `hpm_ai_v3/agents/base_discovery.py` — add `tool_description` property to `ActionPattern`, wire `tool_selector` into `act()`
- **MODIFY**: `hpm_ai_v3/agents/discovery_agent.py` — instantiate `ToolSelector` with LM in `__init__`
- **CREATE**: `hpm_ai_v3/tools/test_tool_selector.py` — unit + integration tests

## Success Criteria
- `ToolSelector.score("Count words in hello world", patterns)` returns highest score for `str.split` pattern
- `ToolSelector.score("Is this positive?", patterns)` returns highest score for `TextBlob.TextBlob` pattern
- Agent with ToolSelector advances through NLP Tool Mastery phase faster than without
- No regression: agent still advances through arithmetic phases at same rate
- `tool_selector=None` leaves `act()` behaviour completely unchanged
