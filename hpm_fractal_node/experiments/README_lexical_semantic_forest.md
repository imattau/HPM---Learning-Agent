# `experiment_lexical_semantic_forest.py`

## Summary
Builds a large WordNet-backed prior forest and exercises it with a real text corpus. This is the most direct "large external ontology" experiment in the suite and the first one that shows HFN operating over several thousand protected priors rather than a small synthetic set.

## What It Does
- Builds lemma, synset, relation, POS, lexname, depth, and ontology-root nodes from NLTK WordNet.
- Uses Peter Rabbit text as the observation stream.
- Compares compact and large ontology modes.
- Measures coverage, learned-node growth, category purity, and abstraction usage.

## What It Aims To Achieve
- Test whether HFN can hold and reuse a several-thousand-node structured prior library.
- Show that the model can ground real text against a layered ontology.
- Probe whether the system behaves more like a semantic memory substrate than a toy clustering model.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_semantic_forest.py
```

## Main Signals
- Large mode builds a much richer ontology than compact mode.
- The observation stream is still fully covered by priors in smoke runs, which means the ontology is already strong enough to explain simple corpus text.
- Mean explaining layer gives a quick read on whether the system leans on surface nodes or higher abstractions.

## Insights
- The experiment supports the idea that HFN can act as a structured knowledge store.
- The model can keep many protected nodes alive while still explaining a real text corpus.
- The large ontology is the first concrete sign that the system can work with AI-like semantic scaffolding rather than only hand-built toy priors.

## Issues / Limits
- Full coverage means the run does not force much novel learning.
- Learned-node pressure is low on simple corpora, so this experiment is stronger as a representation test than as a learning test.
- Runtime grows with ontology size, so validation can be slow.

## Notes
- Uses diagonal sigma nodes.
- The WordNet data must be available locally through NLTK.
