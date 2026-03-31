# `experiment_lexical_transfer.py`

## Summary
Compares an in-domain Peter Rabbit / repository-text stream against out-of-domain vocabulary sampled from NLTK `words`, both grounded in the same WordNet ontology. This is the experiment that tests whether the ontology still covers a lexical shift when direct anchors disappear.

## What It Does
- Reuses the WordNet-backed forest from the semantic experiment.
- Builds an in-domain batch from Peter Rabbit and repository markdown.
- Builds an OOD batch from `nltk.corpus.words` with lexical filtering.
- Measures known-token rate, fallback rate, coverage, learned-node pressure, abstraction usage, and RSS.

## What It Aims To Achieve
- Test lexical transfer rather than simple in-domain coverage.
- See whether the ontology can explain unfamiliar surface vocabulary without collapsing.
- Check whether harder vocabulary forces more learned structure or higher-level explanations.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_transfer.py
```

## Main Signals
- In-domain text has some lemma overlap with the ontology.
- OOD text can be made much harder than the smoke run by increasing token length, reducing overlap, and favoring rare forms.
- The most important outputs are coverage, learned-node survival, and mean explaining layer.

## Insights
- Even hard OOD streams can remain fully explainable when the ontology is rich enough.
- OOD tends to push explanations upward into more abstract layers.
- The experiment is a good stress test for prior coverage, but not yet a guaranteed open-ended learner.

## Issues / Limits
- In smoke runs the model still explains everything using priors.
- Learned nodes are sparse unless the stream is large and difficult enough.
- Runtime is substantial at full scale because the experiment builds and drives a large WordNet forest.

## Notes
- Uses diagonal sigma nodes.
- This experiment is useful as a transfer test, not yet as a proof of durable concept acquisition.
