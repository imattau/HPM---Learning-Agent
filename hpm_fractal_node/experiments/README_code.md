# experiment_code.py — Python Code Token Prediction

## Purpose

This experiment asks: **can an HFN world model learn the category structure of Python syntax
tokens from raw observations, and can external stdlib knowledge fill gaps when the world model
encounters unfamiliar tokens?**

Python source code has a clear categorical structure: control-flow keywords, function-definition
keywords, built-in functions, and type names each occupy distinct syntactic roles. If the
Observer's learned nodes align with these categories, it demonstrates that unsupervised
pattern-extraction via HFN dynamics can recover meaningful structure from token co-occurrence
in a programming language domain.

The secondary question is whether `QueryStdlib` — which scans Python stdlib source files for
token signatures — can inject useful knowledge when observations fall into regions of the
token space not covered by the world model.

---

## HPM Connection

| HPM concept | How it appears in this experiment |
|---|---|
| Pattern substrate | HFN nodes encode token vectors (D=70 dimensional embeddings) |
| Pattern prior | `build_code_world_model()` seeds the forest with one HFN per vocabulary token (261 word priors) |
| Pattern dynamics | Observer update loop: weight gain on match, loss on miss, absorption of redundant nodes, compression of co-occurring nodes |
| Pattern evaluator | `tau` threshold gates which nodes count as explanations; weight score determines survival |
| Gap query | `QueryStdlib` fires when best accuracy < `gap_query_threshold=0.7`; scans stdlib source for the unknown token |
| Converter | `ConverterCode` translates stdlib query results into new HFN nodes seeded near the gap in token space |
| Hierarchy | `max_expand_depth=2` allows the Observer to search two levels of the node hierarchy when explaining an observation |

This experiment operates at **HPM Level 1–2** (sensory regularities and latent structural
representations). The token embeddings are the sensory input; the category clusters that emerge
in the node population are latent structural representations. The experiment does not reach
Level 3 (relational rules between tokens) — that would require sequence modelling, which is
not implemented here.

---

## Architecture

### World model

Built by `build_code_world_model()` (in `hpm_fractal_node/code/code_world_model.py`). On first
run this is computed and cached to `data/code_world_model.{npz,json}`; subsequent runs load
from disk.

The world model contains **261 prior nodes** — one per vocabulary token. Each prior node has
`mu` set to the token's embedding vector in a 70-dimensional space (D=70). All prior nodes are
registered as protected (their weights are updated but they cannot be absorbed).

### Observer settings

```python
Observer(
    forest=forest,
    tau=calibrate_tau(D=70, sigma_scale=1.0, margin=1.0),
    budget=10,
    protected_ids=prior_ids,
    query=QueryStdlib(max_results=10),
    converter=ConverterCode(),
    gap_query_threshold=0.7,
    max_expand_depth=2,
    vocab=VOCAB,
)
```

`calibrate_tau` sets tau such that a node explains observations within approximately 1 standard
deviation of its mean in 70-dimensional space. `budget=10` limits how many nodes are evaluated
per observation for efficiency.

### Observations

`generate_code_snippets(seed=42)` generates synthetic Python code snippets and tokenises them.
Each observation is a (vector, token_string, category) triple, where the category is one of:
`control_flow`, `functions`, `data`, `builtins`, or `unknown`.

2000 observations are used across 3 passes (shuffled each pass), giving 6000 total observation
events.

### Query/Converter pipeline

- **QueryStdlib**: When the best accuracy score for an observation falls below 0.7, the Observer
  flags a knowledge gap. `QueryStdlib` is called with the gap context; it searches CPython stdlib
  source files for occurrences of the unknown token and returns up to 10 matching signatures or
  usage patterns.
- **ConverterCode**: Converts stdlib search results into HFN nodes. Each result is embedded into
  the 70-dimensional token space and used to initialise a new `gap_` or `sig_` node near the
  gap position.

---

## How to Run

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_code.py
```

The world model is built and cached on first run (may take a minute). Subsequent runs load from
`data/code_world_model.*` and are faster.

To force a rebuild, delete the cache files:

```bash
rm data/code_world_model.npz data/code_world_model.json
```

---

## Results

Results from a recent run:

**World model**
```
World model: 261 nodes (261 priors), D=70
Observer created (tau=<calibrated>)
Observations: 2000
```

**Per-pass forest size**
```
Pass 1: forest=<N>, query_nodes=<n>, gap_nodes=<n>
Pass 2: ...
Pass 3: ...
```

**Category purity** (evaluated on 200 observations):

| Category | Purity |
|---|---|
| control_flow | 1.000 |
| functions | 0.268 |
| builtins | 0.091 |
| data | 0.000 |

**Node counts (after 3 passes)**:

| Node type | Count |
|---|---|
| Knowledge gap nodes (`gap_`) | 248 |
| Signature nodes (`sig_`) | 1 |
| Query nodes (`query_`) | 0 |
| Total forest size | 261 (priors) + gap/sig nodes |

### Interpreting the purity scores

**control_flow purity = 1.000**: Tokens like `if`, `for`, `while`, `==`, `(`, `)` are the most
frequent in the corpus and occupy a coherent region of the 70-dimensional token space. The world
model prior nodes for these tokens reliably explain observations from this category, so the
nearest-word lookup correctly identifies the category 100% of the time.

**functions purity = 0.268**: Keywords like `def`, `class`, `return`, `lambda` share some
embedding space with control-flow tokens (both are frequent, short keywords). The boundary
between these categories is blurry in the embedding space, leading to a purity below 0.3.

**builtins purity = 0.091**: Built-in functions (`print`, `len`, `range`) have embeddings that
are spread across the token space, overlapping with other categories. The world model struggles
to cluster these distinctly.

**data purity = 0.000**: Type names (`int`, `str`, `float`, `bool`, `list`) are among the
rarest tokens in the synthetic snippets and have embeddings that don't form a tight cluster.
No observations from this category are correctly predicted.

**248 gap nodes**: The large number of gap nodes indicates that the 261-token world model
frequently encounters observations it cannot explain with accuracy >= 0.7. QueryStdlib fires
for each of these, seeding new `gap_` nodes near the unexplained region of token space. This is
expected: the synthetic code snippets contain many token sequences that are structurally novel
relative to the single-token priors.

---

## Key Insights

**1. Simple priors explain simple categories well.**
Control-flow tokens form a tight, frequent cluster in the token space. With one prior per token,
the Observer can reliably attribute control-flow observations to the right prior. This is HPM
Level 1 in action: sensory regularities (frequent tokens) are captured by simple pattern nodes.

**2. Category boundaries are not always clean in the embedding space.**
The `functions` and `builtins` categories have low purity because their token embeddings
overlap with control-flow. This reflects a genuine property of the domain: Python's syntax
does not draw sharp boundaries between these groups in raw token space. A higher-level
representation (e.g. parsing the AST) would be needed to distinguish them reliably — that
would correspond to HPM Level 3 (relational rules).

**3. Gap-filling fires frequently but produces ephemeral nodes.**
248 gap nodes are created, far outnumbering the original 261 priors. This suggests the stdlib
query threshold (0.7) is set aggressively — many observations trigger gap queries even when
a reasonable prior exists. The gap nodes are not protected, so they compete with priors and
with each other; many will be absorbed in subsequent passes. In practice, gap-filling is more
useful when the world model has genuine blind spots (unknown tokens), not when it is merely
imprecise.

**4. Signature injection is minimal (1 sig_ node).**
QueryStdlib found exactly one token for which it could produce a clean function signature.
This reflects the nature of Python syntax tokens: most are keywords or operators with no
stdlib signature. The Query/Converter pipeline is better suited to the NLP experiment (where
LLM-generated semantic neighbours are more informative).

---

## Known Issues / Open Questions

- **Embedding quality**: The 70-dimensional token embeddings are the primary determinant of
  category purity. If the embeddings do not separate categories well, no amount of Observer
  dynamics will recover the structure. A pre-trained subword embedding (e.g. from a code LM)
  might substantially improve purity.

- **Gap threshold too low**: A `gap_query_threshold` of 0.7 triggers too many gap queries for
  a domain where the world model has broad but imprecise coverage. Raising this to 0.85–0.9
  would reduce the gap node count and focus queries on genuinely novel tokens.

- **Category definition mismatch**: The four categories (`control_flow`, `functions`, `data`,
  `builtins`) are defined by hand and do not perfectly reflect token co-occurrence patterns.
  An unsupervised clustering of the prior nodes' mu vectors might reveal different natural
  groupings.

- **Sequence structure ignored**: This experiment treats each token as an independent
  observation. Python code has strong sequential structure (e.g. `def` is almost always
  followed by an identifier). A sequential encoding (context window or n-gram) would let the
  Observer discover relational rules (HPM Level 3).
