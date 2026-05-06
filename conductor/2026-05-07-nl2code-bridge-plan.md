# Natural Language to Code Bridge (NL2Code) Experiment

## Objective
To demonstrate that the HPM v5 core can bridge the structural gap between Natural Language (NL) and source code (Code) by leveraging a shared **UnifiedVocabulary**. This "Cross-Modal Structural Transfer" acts as the foundational capability for Natural Language Programming, where abstract intent is directly mapped to code idioms.

## Concept
1. **The Shared Latent Space**: The `UnifiedVocabulary` acts as a common numeric coordinate system.
2. **Code Side (Training)**: We feed Python code snippets (e.g., a simple loop or if-statement) through the `UnifiedASTFlattener`. The core learns an idiomatic pattern composed of Universal IDs (e.g., `U_IF`, `U_CALL`, `U_ASSIGN`).
3. **NL Side (Bridging)**: We define a "Bridge Adapter" that maps specific semantic NL concepts (like "conditional", "check", "execute", "assign") to their structural equivalents in the Universal Vocabulary (`U_IF`, `U_CALL`, `U_ASSIGN`).
4. **Recognition**: We feed a natural language query (e.g., "If the weather is cold, send a warning"). The NL pipeline translates the skeleton of this sentence into universal IDs (`U_IF`, `U_CALL`). The HPM engine, having already learned the Python pattern, recognizes this sequence and "retrieves" the Python code idiom.

## Architecture

### 1. `NL2CodeBridgeAdapter`
- A new preprocessor adapter.
- Takes the `canonical_tokens` from the NLP pipeline.
- Uses a heuristic map to translate functional English words into `UniversalVocabulary` IDs (e.g., "if" -> `U_IF`, "loop" -> `U_WHILE`, "set" -> `U_ASSIGN`, "call" / "run" / "invoke" -> `U_CALL`).
- Outputs a `bridge_state` consisting of these universal IDs.

### 2. The Experiment Harness (`hpm_ai_v5/experiments/experiment_nl2code_bridge.py`)

#### Phase 1: Code Acquisition
- Engine is trained on a few simple Python scripts using the `CLT` (Cross-Language Transfer) adapters (`UnifiedASTFlattener` -> `UnifiedStateAdapter`).
- Patterns are promoted and rewarded, becoming "known idioms".

#### Phase 2: NL Queries
- The engine is frozen (learning disabled, or just testing).
- Several NL queries are processed through the `NLPTokenizer` -> `CanonicalPhraser` -> `NL2CodeBridgeAdapter`.
- The engine observes the resulting `bridge_state`.
- We evaluate if the `last_match` correctly identifies the structural code idiom learned in Phase 1.

## Proposed Code Idioms & NL Pairs

**Idiom 1: Conditional Action**
- **Code:** `if condition: execute_action()` -> Structure: `[U_IF, U_CALL]`
- **NL Query:** "If the user is admin, allow access." -> Bridge: `[U_IF, U_CALL]`

**Idiom 2: Iteration**
- **Code:** `while running: do_work()` -> Structure: `[U_WHILE, U_CALL]`
- **NL Query:** "Keep running the process until it stops." -> Bridge: `[U_WHILE, U_CALL]`

**Idiom 3: Assignment & Return**
- **Code:** `result = calculate(); return result` -> Structure: `[U_ASSIGN, U_CALL, U_RETURN]`
- **NL Query:** "Set the output to the calculation, then return it." -> Bridge: `[U_ASSIGN, U_CALL, U_RETURN]`

## Execution Plan
1.  **Create `NL2CodeBridgeAdapter`** in `hpm_ai_v5/adapter/nlp.py` or a dedicated file.
2.  **Create the Experiment Script** `hpm_ai_v5/experiments/experiment_nl2code_bridge.py` combining pipelines.
3.  **Run and Validate** the cross-modal transfer.
4.  **Document Results** demonstrating the translation from intent to structure.
