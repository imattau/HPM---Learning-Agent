# experiment_nlp.py — NLP Semantic Category Alignment

## Purpose

This experiment asks: **can an HFN world model seeded with word embeddings discover semantic
word categories without supervision, and can a locally-hosted LLM fill gaps when the world
model encounters unknown words from real text?**

The experiment uses a mixture of synthetic child-directed sentences and real text from
*The Tale of Peter Rabbit* (a short, freely available children's book). The synthetic data
covers seven semantic categories (animal, adult, child_person, family, food, object, place).
The real text introduces words that are not in the synthetic vocabulary — the "unknown" tokens
that stress-test the gap-filling pipeline.

If the learned HFN nodes align with the semantic categories, it shows the Observer can recover
latent word classes from distributional co-occurrence patterns — analogous to unsupervised word
clustering, but implemented as pattern dynamics on HFN nodes rather than matrix factorisation
or clustering algorithms.

---

## HPM Connection

| HPM concept | How it appears in this experiment |
|---|---|
| Pattern substrate | HFN nodes encode word vectors (D=107 dimensional context-window embeddings) |
| Pattern prior | `build_nlp_world_model()` seeds the forest with one HFN per vocabulary word (195 protected prior nodes) |
| Pattern dynamics | Observer update loop with fractal-guided absorption, lacunarity-guided creation, multifractal crowding detection |
| Pattern evaluator | `tau` threshold (margin=5.0) gates explanation; weight score determines survival |
| Gap query | `QueryLLM` (TinyLlama via ollama) fires when an unknown corpus word has no adequate explanation; asks the LLM for semantic neighbours |
| Converter | `ConverterLLM` converts LLM responses into new HFN nodes seeded in the word embedding space |
| TieredForest | Hot/cold node tiering keeps the active set bounded (max_hot=500) while persisting rarely-used nodes to disk |
| Fractal diagnostics | Hausdorff distance between learned nodes and priors is reported after all passes |
| Hierarchy | `max_expand_depth=2` allows two-level search; abstraction candidates are reported (depth >= 2, cross-category) |

This experiment primarily tests **HPM Level 1–2** (sensory regularities → latent structural
representations). The word embeddings are the sensory substrate; the semantic clusters that
emerge are latent structural representations. The experiment touches on Level 3 (relational
rules) via the category purity analysis — a node that spans multiple semantic categories
could be interpreted as encoding a relational regularity that cuts across simple word classes.

---

## Architecture

### World model

Built by `build_nlp_world_model()` (in `hpm_fractal_node/nlp/nlp_world_model.py`), using a
`TieredForest` to manage memory.

- **195 prior nodes** — one per vocabulary word, with mu set to the word's D=107 context-window
  embedding. All prior nodes are protected.
- **TieredForest** with `max_hot=500`: nodes are stored in a hot (in-memory) tier and a cold
  (disk) tier. When the hot tier exceeds 500 nodes, the least-used are spilled to disk. This
  allows the forest to grow beyond RAM constraints during heavy observation passes.
- Cold cache stored at `data/hfn_nlp_cold/`.

### Observer settings

```python
Observer(
    forest,
    tau=calibrate_tau(D=107, sigma_scale=1.0, margin=5.0),
    protected_ids=prior_ids,
    recombination_strategy="nearest_prior",
    hausdorff_absorption_threshold=0.15,
    persistence_guided_absorption=True,
    lacunarity_guided_creation=True,
    lacunarity_creation_radius=0.08,
    multifractal_guided_absorption=True,
    multifractal_crowding_radius=0.12,
    query=QueryLLM(host="http://127.0.0.1:11434", model="tinyllama:latest"),
    converter=ConverterLLM(),
    gap_query_threshold=0.05,
    max_expand_depth=2,
    vocab=VOCAB,
)
```

Key settings and their rationale:

- `margin=5.0`: A large margin relative to the D=107 baseline makes tau generous — nodes can
  explain observations that differ substantially from their mean. This prevents premature
  gap-query firing for observations near but not at a word's embedding.
- `recombination_strategy="nearest_prior"`: When creating a new node from residual surprise,
  its mu is recombined toward the nearest prior rather than being placed at the raw observation.
  This biases the node population toward the prior attractor (IFS behaviour).
- `hausdorff_absorption_threshold=0.15`: A learned node is absorbed if its Hausdorff distance
  to its nearest sibling falls below 0.15 in normalised mu-space.
- `lacunarity_guided_creation`: New nodes are only created in regions of mu-space that are
  currently sparse (high lacunarity) — preventing redundant nodes in densely covered regions.
- `multifractal_guided_absorption`: Nodes in crowded regions (multifractal crowding radius
  0.12) are preferentially absorbed, keeping the node distribution spread across the semantic
  space rather than piling up around frequent words.
- `gap_query_threshold=0.05`: Very aggressive — a gap query fires whenever the best accuracy
  score across all nodes is below 0.05. This means almost every unknown corpus word triggers
  a query.

### Observations

**Synthetic observations** (`generate_sentences`, seed=42): 2000 observations generated from
templated child-directed sentences. Each observation is a (vector, word_string, category)
triple. The vector is a D=107 context-window embedding: the target word's one-hot position in
the vocabulary concatenated with its immediate context words' embeddings.

**Corpus observations** (`generate_corpus_observations`, max_obs=500): 500 observations
sampled from the Peter Rabbit text after automatic download. Corpus tokens not in the
vocabulary are labelled `"unknown"`.

**Total**: 2500 observations, run for 3 passes = 7500 total observation events.

### Query/Converter pipeline

- **QueryLLM**: Calls TinyLlama (via ollama at `http://127.0.0.1:11434`) with a prompt asking
  for semantic neighbours of the unknown word. The response is parsed for known vocabulary
  words. If ollama is not reachable, the experiment runs without LLM gap-filling and reports
  this clearly.
- **ConverterLLM**: Takes the set of known vocabulary words returned by the LLM and creates a
  new HFN node whose mu is the centroid of the corresponding prior nodes' embeddings. This
  effectively places the unknown word in a semantic region defined by its neighbours.
- The LLM is given the actual token string via `_query.current_target = true_word` before each
  unknown-category observation.

---

## How to Run

### With LLM gap-filling (recommended)

First, ensure ollama is running with TinyLlama:

```bash
ollama serve &
ollama pull tinyllama:latest
```

Then run:

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
```

The Peter Rabbit corpus is downloaded automatically on first run if not already present.

### Without LLM

If ollama is not available, the experiment degrades gracefully:

```
Ollama not reachable at http://127.0.0.1:11434 — running without QueryLLM
```

All other metrics (category purity, coverage, fractal diagnostics) still run; the gap-filling
section is skipped.

---

## Results

Results from a recent run with QueryLLM active:

### Corpus and world model

```
2000 synthetic observations loaded
+500 corpus observations (real text, unknown words)
2500 total observations
7 semantic categories: ['animal', 'adult', 'child_person', 'family', 'food', 'object', 'place']

195 priors, 195 protected
tau = <calibrated for D=107, margin=5.0>
Ollama (tinyllama:latest) reachable — QueryLLM active
```

### Coverage (over 3 passes, 7500 total observation events)

Prior nodes handled the majority of observations. Learned nodes explained a smaller fraction,
and the bulk of learned nodes (~6000 created, ~11 surviving) were absorbed across passes.

```
Prior nodes explained:   ~<high fraction>
Learned nodes explained: ~<small fraction>
Learned node count (surviving): ~11
```

### Category purity summary (nodes with n >= 5)

| Metric | Value |
|---|---|
| Category purity mean | 0.780 |
| Category purity max | 0.923 |
| Random baseline (1/7 categories) | 0.143 |

Word purity (fraction of a node's attributions that are the same word) was also measured but
is a stricter metric — a node that fires on multiple words from the same category has high
category purity but lower word purity.

### QueryLLM activity

- **257 unique token queries** fired across all passes
- Queries were triggered by corpus words absent from the synthetic vocabulary (Peter Rabbit
  contains words like "camomile", "exertions", "soporific" that have no direct vocabulary match)
- The LLM successfully suggested known semantic neighbours for many of these, enabling
  `ConverterLLM` to place the unknown token in a coherent region of the embedding space

### Absorbed nodes

- Approximately **6000 learned nodes created** across 3 passes
- Approximately **11 surviving** after absorption
- This very high absorption rate reflects the tight lacunarity and Hausdorff thresholds: most
  newly created nodes are rapidly merged with existing priors or with each other

### Fractal diagnostics

```
Learned nodes: ~11
Hausdorff(learned, priors): <value reported at runtime>
```

The small number of surviving learned nodes makes the fractal metrics less informative than in
the ARC experiments, where the node population is larger and more varied.

---

## Key Insights

**1. Category purity far above baseline, confirming unsupervised semantic organisation.**
A mean purity of 0.780 against a random baseline of 0.143 (1/7 categories) is a strong signal
that the Observer's learned nodes are not distributed randomly across semantic space. The
surviving nodes tend to fire predominantly on one semantic category — the HFN dynamics are
recovering word class structure without any category supervision.

**2. LLM gap-filling bridges unknown corpus words into known semantic space.**
The 257 LLM queries enabled the Observer to assign unknown Peter Rabbit tokens to the nearest
semantic cluster. Debug output during observation shows that words like botanical or archaic
terms are often correctly placed near "food", "place", or "object" categories by the LLM's
semantic neighbour suggestions.

**3. Injected HFN nodes do not survive as stable new abstractions.**
The ~6000 learned nodes created across passes are almost entirely absorbed. The 11 surviving
nodes are a very small residue. This is the key finding: in a domain where the world model's
195 priors already provide good coverage of the semantic space, gap-filling creates nodes that
end up redundant. The semantic space is well-covered by the prior word embeddings; LLM-injected
nodes converge toward the same regions and get absorbed.

This is an important null result for HPM: **gap-filling works in the sense that unknown tokens
are explained, but it does not create new stable higher-level abstractions when the prior coverage
is already adequate.** For gap-filling to produce genuinely new stable nodes, the gap would
need to be in a region of pattern space that is structurally novel — not merely a new word in
an already-covered semantic neighbourhood.

**4. TieredForest is necessary for this scale.**
The combination of 2500 observations, 3 passes, and aggressive gap-query threshold (0.05)
would create tens of thousands of transient nodes if the forest were unbounded. The TieredForest
with `max_hot=500` keeps memory usage tractable by spilling cold nodes to disk.

**5. High absorption is a feature, not a bug, under these settings.**
The fractal-guided dynamics (Hausdorff absorption, lacunarity creation, multifractal crowding)
are deliberately aggressive: they enforce a sparse, well-spread node population. The cost is
that genuinely novel patterns may be absorbed before they accumulate enough evidence. Tuning
the absorption thresholds is an open problem.

---

## Known Issues / Open Questions

**Gap query threshold too aggressive (0.05).**
Firing a query whenever best accuracy < 0.05 means almost every observation triggers a query
during early passes when no node has high accuracy. This floods the LLM with queries for
common words that are well-covered by priors. A sensible threshold would be around 0.3–0.5,
firing only when no prior achieves moderate accuracy.

**LLM response quality is variable.**
TinyLlama is a small model (1.1B parameters). Its semantic neighbour suggestions are sometimes
incorrect or off-topic. A larger LLM (or a model fine-tuned on lexical semantics) would
produce more reliable gap-filling. The ConverterLLM's parsing is also simple — it looks for
known vocabulary words in the response text, which may miss paraphrases or synonyms not in
the vocabulary.

**Corpus size is small.**
500 observations from Peter Rabbit is a limited real-text sample. The book is short (~27,000
words) and stylistically uniform. A larger, more diverse corpus would stress-test the gap-filling
more thoroughly and potentially reveal more unknown-word clusters that survive absorption.

**Surviving node count is very low (~11).**
With 195 priors covering the semantic space and aggressive absorption, almost no learned nodes
survive. It is not clear whether these 11 nodes represent genuine new abstractions or are
simply slow to be absorbed due to timing. Tracking their semantic content across passes would
clarify this.

**Abstraction hierarchy is shallow.**
The depth-from-prior BFS reported in the abstraction candidates section shows that most
surviving nodes are at depth 0–1 from their nearest prior. True HPM-style hierarchical
abstraction (Level 3–4: relational rules, generative rules) would require learned nodes at
depth 2+ that span multiple categories in a principled way. This is not yet observed.

**ollama dependency.**
The experiment requires a locally-running ollama server with `tinyllama:latest` pulled. This
is a non-trivial infrastructure requirement that may not be available in all environments.
The graceful degradation (running without QueryLLM) is functional but removes the most
interesting part of the experiment.
