# HPM v2 Text Storage: Structured HFN Hierarchy

**Status:** implemented (storage version 2). Access over this structure is not yet built.
**Context:** responses were thin ([chat design](2026-10-23-hpm-v2-generation-chat-design.md)). An audit showed the cause was mostly what the store held and how, so storage was fixed first.

## 1. Audit: what was stored versus what an HFN is meant to hold

An HFN is a Gaussian identity plus a DAG polygraph of children joined by typed edges, with the same shape at every depth. Ingesting a two-paragraph source with the study trainer gave:

| Intended | Found |
|---|---|
| One node per sentence, part of the document hierarchy | Two: a hierarchy `sentence` node (word children, no offsets, random uuid id) and a separate flat `answer_sentence` node (offsets, no children, not in the hierarchy). Only the flat one was ever ranked, weighted or cited |
| Paragraphs as real structure | One paragraph per source, even across a blank line |
| Typed edges between children | Zero edges anywhere. The "polygraph" was a bag |
| Stable identity | Hierarchy ids were `uuid4`, different on every build |
| Relations across sentences and sources | None. No order, no shared concepts. Concept nodes existed only through supervised `concept_key` |
| Word forms meet | `release` and `released` were unrelated |

Persistence of the hierarchy already worked (children survived a checkpoint round trip). Separately, `ReaderAgent.query_scored_nodes` ranks by looping over a Python list of passage vectors and never queries the forest. That is an access matter, left for the next step.

## 2. Storage version 2

For a source `S`, `SupervisedStudyTrainer.ingest` now calls `observe_passage` with its sentence spans and builds:

```
passage node  ──next──>  (paragraph ... paragraph)        edges between paragraphs
  paragraph   ──next──>  (sentence ... sentence)          edges between sentences
    sentence  = children: word macros
                metadata: source_id, start, end, ordinal, stems, tokens
term_<stem>   = children: every sentence containing the stem, across sources
```

- **One node per sentence.** The sentence node is the hierarchy node. Its id keeps the legacy `answer_sentence_<sha1(source:start:text)>` scheme, so learned facts and ids in older ledgers still resolve. Its Gaussian is the encoding of the sentence text (identical to before), and its Observer weight starts at 0.5, so all rankings are unchanged.
- **Real paragraphs** follow blank-line breaks. Ids are deterministic (`paragraph_<sha1(source:ordinal:start)>`).
- **`next` edges** chain consecutive sentences and consecutive paragraphs, using HFN's own `add_child(relation=)`.
- **Term nodes** are the concept layer. The stem is a Porter stem of content words (stopwords and stems under 3 characters dropped). A stem becomes a node when a second sentence uses it, and not before. Its Gaussian is the centroid of its first two sentences and is not moved afterwards; later sentences are added as children only. A sentence can be a child of many terms.
- **Reverse lookups** (`StructureIndex`) are derived from the nodes and rebuilt on load. HFN has no parent references, so this is the supported way to ask "which paragraph holds this sentence", "what precedes it", "which sentences share a concept", and "which sentences mention this stem" (singletons included, and the query form `sentences_for_query` maps `release` to `released`).
- **Checkpoints** record `storage_version: 2`. A checkpoint without it loads as version 1 (flat) and `storage_problems` says so. Sources of a v1 checkpoint are not rebuilt in place: re-ingest them.

## 3. Checking it

`hpm_ai_v2/training/storage.py::storage_problems(trainer)` returns problems, empty when sound: every source's passage, paragraph and sentence nodes match the source's sentences in order; `next` edges chain children exactly; each sentence is in one paragraph and its offsets reproduce the source text; stored stems equal the sentence's stems; every `answer_node_data` id is inside the hierarchy; every term has two or more children, each mentioning the stem; the stem map and term nodes agree. Tests damage each of these and confirm the checker reports it.

## 4. Evidence

- 124 chat, study and reader tests pass unchanged against the new storage (including the 92 chat and study tests, with rankings identical by construction).
- 12 new structure tests: polygraph shape, one node per sentence, deterministic ids across builds, shared concepts across sources, word forms, order and neighbours across a paragraph break, incremental ingest, checkpoint round trip and growth, legacy detection, damage detection, stem lookup.
- The wider suite has 35 failing or erroring lines in `hpm_fractal_node`, benchmarks, ARC encoders and the old `tests/hpm_ai_v2/test_step*` files. The set is identical on a clean export of `HEAD`, so it is pre-existing and unrelated.
- Chat answers to the earlier probe questions are unchanged. That is expected: only storage moved.

## 5. Limits and risks

- **No answer improves yet.** Access still ranks by vector similarity. What changes is that the missing relations now exist to be used.
- **A single source has few concept links.** The oxygen sentence of a lone photosynthesis source shares no stem with another sentence, so its links are adjacency only. Stem lookup reaches it directly; concept edges grow as sources are added.
- **Porter stemming is English-only.** Other languages would need another stemmer.
- **Term centroids are frozen** at creation. If the vocabulary changes, they are not recomputed.
- **Node count grows** with distinct shared stems. They are created lazily and registered in bulk during ingest, but large corpora should be measured.
- **The Reader's default path is unchanged.** Only the trainer's structured path builds paragraphs, edges and terms. Other callers of `observe_passage` get the old one-paragraph form.

## 6. Next

1. Make access use the structure: seed with the best sentence, expand along `next` edges and shared terms with decay, and choose answer width by marginal gain. Map query stems to sentences with `sentences_for_query`.
2. Add the support-set coverage metric so each step is measured against complete answers, not only top-1 correctness.
3. Decide separately whether the Reader's `query_scored_nodes` should go through the forest.
