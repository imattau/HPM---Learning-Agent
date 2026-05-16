# Knowledge Frontier Design

**Date:** 2026-05-17
**Status:** Approved

## Overview

The model currently gets stuck learning the same Wikipedia content repeatedly — the entropy filter marks previously read articles as "already known" and no genuinely new content flows in. The fix: a `KnowledgeFrontier` that uses WordNet's semantic graph to progressively expand what the model learns, always targeting concepts with the sparsest pattern coverage.

## Architecture

Two mechanisms work together:
1. `_nominate_uncertain_topics()` identifies what the model is most uncertain about (sparse pager edge density) — these are the **seeds**
2. `KnowledgeFrontier` expands from those seeds via WordNet — these are the **next Wikipedia fetch targets**

`_nominate_uncertain_topics` stays but now feeds the frontier rather than directly driving Wikipedia fetches. The frontier handles WordNet expansion and ensures variety.

## KnowledgeFrontier Class

Lives in `hpm_ai_v6/cli/quiz_cli.py` (or `hpm_ai_v6/cli/knowledge_frontier.py` if it grows large).

### State (persisted to `hpm_ai_v6/data/quiz_banks/knowledge_frontier.json`)

```json
{
  "known_seeds": ["paris", "capital", "france"],
  "frontier": ["metropolis", "city-state", "europe", "seine"],
  "exhausted": ["geography", "general"],
  "hop_depth": 2
}
```

- `known_seeds` — terms used as WordNet expansion seeds (not re-expanded)
- `frontier` — WordNet-derived candidates not yet fetched
- `exhausted` — topics fetched that yielded no novel sentences (skip on re-fetch)
- `hop_depth` — current expansion depth, increments each loop, max 5

### Key Methods

**`add_learned_seeds(terms, reader)`**
- For each term not already in `known_seeds`:
  - Add to `known_seeds`
  - Call WordNet at current `hop_depth` to get hypernyms + hyponyms
  - Filter candidates: remove stopwords, POS tags (`pos_`, `word_` prefixes), single chars, terms already in `known_seeds` or `exhausted`
  - Score each candidate by pager edge count across all agents in reader (ascending = sparse = most to learn)
  - Add top 8 to `frontier`

**`next_topics(reader, n=4)`**
- Re-score current frontier by live pager edge density
- Return `n` candidates with fewest edges (most to learn)
- Remove returned topics from `frontier`, add to `exhausted` after fetch

**`increment_hop()`**
- `hop_depth = min(hop_depth + 1, 5)`

**`save()` / `load(path)`**
- JSON serialise/deserialise state to `hpm_ai_v6/data/quiz_banks/knowledge_frontier.json`

### WordNet Expansion

```python
from nltk.corpus import wordnet

def _wordnet_candidates(term, hop_depth):
    synsets = wordnet.synsets(term.lower().replace(" ", "_"))[:3]
    candidates = set()
    for syn in synsets:
        for hop1 in syn.hypernyms() + syn.hyponyms():
            name = hop1.lemmas()[0].name().replace("_", " ")
            candidates.add(name)
            if hop_depth >= 2:
                for hop2 in hop1.hypernyms() + hop1.hyponyms():
                    candidates.add(hop2.lemmas()[0].name().replace("_", " "))
    return candidates
```

### Edge Density Scoring

```python
def _edge_density(term, reader):
    word = term.lower().split()[0]
    count = 0
    for agent in reader.agents.values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        for payload in pager.iter_index_payloads():
            if word in str(payload.get("name", "")).lower():
                count += 1
    return count
```

Lower count = sparser = higher learning priority.

## Integration into Quiz Loop

### Initialisation in `main()`

```python
frontier = KnowledgeFrontier.load("hpm_ai_v6/data/quiz_banks/knowledge_frontier.json")
```

### After each quiz round (replacing model-nominated topics)

```python
# 1. Train on weak topics from failed questions (unchanged)
train_on_weak_topics(reader, weak_topics, ...)

# 2. Get uncertain topics as seeds for frontier expansion
seeds = _nominate_uncertain_topics(dataset_agent, reasoning_agent, n=4, ...)

# 3. Expand frontier from seeds via WordNet
frontier.add_learned_seeds(seeds, reader)

# 4. Get next fetch targets from frontier
next_topics = frontier.next_topics(reader, n=4)

# 5. Fetch Wikipedia for frontier targets
if next_topics:
    train_on_weak_topics(reader, [(t, t) for t in next_topics], ...)
    
# 6. Expand hop depth and save
frontier.increment_hop()
frontier.save()
```

## Files

| File | Action |
|---|---|
| `hpm_ai_v6/cli/quiz_cli.py` | Add KnowledgeFrontier class; update main() loop |
| `hpm_ai_v6/data/quiz_banks/knowledge_frontier.json` | Created at first run |
| `hpm_ai_v6/tests/test_knowledge_frontier.py` | Create — unit tests |
