# Plan: SP‑Web3 – DictionaryAgent – External Knowledge as a Fractal Substrate

**Objective:** Integrate a **DictionaryAgent** that looks up word definitions, synonyms, antonyms, and usage examples from an external source (e.g., a local dictionary file or mock). It stores retrieved information as HFN nodes in the shared forest. The WriterAgent can query this agent to enrich its natural language answers.

**Strategic Intent:** Demonstrate that external deterministic knowledge bases can be seamlessly incorporated into the HPM fractal ecosystem without breaking uniformity.

## Changes

### 1. Dictionary Agent Implementation (`hpm_ai_v2/agents/dictionary_agent.py`)
- [ ] Create `DictionaryAgent(BaseHFNAgent)`:
    - [ ] `__init__`: Accepts `config`, `forest`, and `reader_agent`.
    - [ ] Include a mock dictionary for initial testing (e.g., "neural", "nerve").
    - [ ] `lookup(word: str) -> Optional[HFN]`:
        - Checks if `definition_{word}` already exists in the forest.
        - If not, fetches the entry from the dictionary source.
        - Uses `reader_agent._ensure_word_macro(word)` to get or create the word node.
        - Uses `reader_agent.ingest_text(definition)` to create a document/paragraph node for the definition text.
        - Uses `reader_agent._ensure_word_macro(pos)` to create a POS node.
        - Creates a new `definition` HFN node.
        - Uses `add_child()` to add the word node, pos node, and definition text nodes as children to the new definition node.
        - Adds a `defined_as` edge from the word node to the definition node.
        - Calls `self.observer.register(def_node, protected=False)`.

### 2. Reader Agent Integration (`hpm_ai_v2/agents/reader_agent.py`)
- [ ] Update `ReaderAgent.__init__` to accept an optional `dictionary_agent`.
- [ ] Update `ReaderAgent.observe_passage` (or the equivalent vocabulary expansion point):
    - When new, unknown words are encountered (e.g., during `dynamic_vocab` expansion), proactively call `dictionary_agent.lookup(word)` to immediately build a semantic grounding for the new term.

### 3. Writer Agent Integration (`hpm_ai_v2/agents/writer_agent.py` & `hpm_ai_v2/agents/mixins/writer.py`)
- [ ] Update `WriterAgent.__init__` to accept an optional `dictionary_agent`.
- [ ] Update `WriterMixin.answer_natural(question)`:
    - After generating the base answer, extract key words.
    - If `dictionary_agent` is available, call `lookup(word)` on key terms.
    - If a definition is found, traverse the definition node's children to extract the definition text (or use its metadata).
    - Append the definition to the natural language response (e.g., "... 'Neural' means relating to a nerve or the nervous system.").

### 4. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_web3_dictionary.py`)
- [ ] Initialize `ReaderAgent`, `WriterAgent`, and `DictionaryAgent` sharing a persistent forest (`data/dictionary_forest`).
- [ ] **Phase 1 - Seeding:** `ReaderAgent` observes a passage containing the word "neural".
- [ ] **Phase 2 - Enrichment:** `WriterAgent` answers a question involving "neural". The agent queries the `DictionaryAgent`, expanding the forest with the definition and enriching its textual response.
- [ ] **Phase 3 - Analogy:** `ReaderAgent` observes "nerve" and the forest demonstrates shared root concepts.
- [ ] **Phase 4 - Verification:** Print forest node counts and verify that the definition node is structured correctly with `defined_as` edges.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_web3_dictionary.py`.
- [ ] Ensure that dictionary definitions are properly fractalized into HFN nodes and linked to word macros.
- [ ] Confirm persistent learning is maintained (the forest is saved and not deleted on startup).