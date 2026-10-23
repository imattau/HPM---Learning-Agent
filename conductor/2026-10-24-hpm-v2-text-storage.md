# HPM v2 Text Storage: Structure and Relations

**Status:** implemented. Storage version 2 is committed in `73e3c49`. The relation layer is described in section 3 and was committed afterwards.
**Why:** the first study path stored each source as a flat bag-of-words vector plus a second flat copy of its sentences. The HFN hierarchy that `ReaderAgent` already built was unused, and nothing stored how sentences or concepts relate. Whatever reads this store can only be as good as the structure in it.

## 1. What a source becomes

```
passage ──next── (paragraphs, split at blank lines)
  paragraph ──next── (sentences)
    sentence: children = word macros; metadata = source_id, start, end, ordinal, stems
term_<stem>: children = the sentences that mention the stem (across sources)
proposition: inputs = concept nodes, relation_params = predicate + role order, supported_by sentence
```

- **One node per sentence.** The sentence node is the citation: it carries exact source offsets, and Observer weight lives on it. There is no parallel flat copy.
- **Typed `next` edges** order sentences within a paragraph and paragraphs within a passage, using HFN's own polygraph edges.
- **Deterministic ids** (hash of source id, offset and text), so two builds of the same sources are identical and learned weights can be matched.
- **Concept layer.** A `term` node exists for each word stem (Porter) that occurs in two or more sentences, so "release", "released" and "releasing" meet. A term's Gaussian is the centroid of its first sentences and is not moved afterwards.
- **Derived lookups.** HFN nodes have no parent references, so `StructureIndex` (`hpm_ai_v2/agents/text_structure.py`) derives reverse lookups (neighbours, sentences for a term, related sentences, relation queries) from the nodes. It is rebuilt on load and is never separate state.

## 2. Guarantees, and how they are checked

`storage_problems(trainer)` in `hpm_ai_v2/training/storage.py` is read-only and returns problems. It checks that:
- each passage's sentences equal the source's sentences in order, and every offset reproduces its text;
- each paragraph and passage chains its children with `next` edges;
- no sentence is outside the hierarchy or in two paragraphs;
- every term's children mention its stem;
- every proposition is supported by exactly one stored sentence, its inputs match its role fillers, and its predicate and arguments occur in that sentence.

Checkpoints record `storage_version`. A checkpoint without it loads as version 1 and is reported as legacy, not misread; re-ingesting the sources rebuilds it. Sentence ids keep the legacy `answer_sentence_` prefix so learned facts keep resolving.

## 3. Relations

Concepts alone say what co-occurs. A proposition says what a sentence asserts: for "Oxygen is released as a by-product" there is a relation node over the concept `oxygen` with predicate `release` and role `PATIENT`.

- **Node form.** `relation_type = "proposition"`, `inputs` = the concept nodes in role order (AGENT, PATIENT), `relation_params = {predicate, roles}`, and a `supported_by` edge to its sentence, the same convention the trainer uses for learned facts. A relation argument always gets a concept node, even when mentioned once.
- **Queries.** `StructureIndex.find(agent=, predicate=, patient=)` takes words and stems them ("what releases oxygen"); `relations_of(term)` lists a concept's relations.
- **Extractor.** `HeuristicRoleExtractor` in `hpm_ai_v2/agents/propositions.py` is conservative and dictionary-based (closed-class words, WordNet morphology and sense frequencies). It handles active, passive (by-agent), copular definitions, relative clauses, coordinated objects and shared subjects, and emits nothing for questions or pronoun-only clauses. `ReaderAgent.role_extractor` can be replaced by anything with an `extract(sentence)` method. A stronger parser is the obvious upgrade; none is installed here (no spaCy, no NLTK tagger data), and nothing was downloaded.
- **Measured quality**, on hand-written sentences (`tests/test_hpm_ai_v2_propositions.py`): 30 sentences used to develop the rules (30/30 relations), 30 used to correct them after a first run (first run: precision 0.92, recall 0.85), and 20 never touched during development (precision 0.93, recall 0.93, on 15 relations). The samples are small, so these are rough, and the extractor will miss or mislabel constructions outside them. The test thresholds guard against regressions only.

## 4. Known limits

- Roles are AGENT and PATIENT only. No instruments, locations, time, negation or modality, so "does not release" is not distinguished from "releases".
- Pronouns are not resolved, so "It produces ATP" yields nothing.
- Infinitive clauses ("to fix carbon dioxide") are not extracted.
- Term Gaussians are fixed at creation and the concept layer is not compressed by the Observer; the text path still stores rather than compresses. Retrieval of passages still scans a vector list rather than the forest. Both are separate pieces of work.
- Checkpoints from before storage version 2 keep working but have no structure.
