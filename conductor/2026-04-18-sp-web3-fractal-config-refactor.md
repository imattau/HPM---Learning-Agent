# Plan: HFN‑Native Configuration – Fractal Domain Bootstrapping

**Objective:** Refactor the `DomainConfig` architecture so that all domain metadata (vocabulary, dimensions, IDF values, operator lists) is stored directly in the HFN Forest as a structured hierarchy of nodes. This eliminates the need for external persistent files like `config.pkl` and ensures the agent is truly self-deriving from its own memories.

**Strategic Intent:** Transition the system from "External Config + Forest" to a "Pure Forest" architecture where the agent "remembers" how to interpret its domain by reading its own foundational HFNs.

## Changes

### 1. Enhance `TieredForest` for Self‑Awareness (`hfn/tiered_forest.py`)
- [ ] Add `_save_meta()` method to write `forest_meta.json` in the `cold_dir`. This file will store basic structural parameters: `D` (dimension size) and `forest_id`.
- [ ] Update `TieredForest.__init__` to allow `D: int = None`.
- [ ] If `D` is `None`, implement an autodetection sequence:
    1. Check for `forest_meta.json` and read `D`.
    2. Fallback: Scan the `cold_dir` for `.npz` files and infer `D` from the first successfully loaded node's `mu`.
- [ ] Call `_save_meta()` during `save_to_cold()`.

### 2. Recursive Fractal Configuration (`hpm_ai_v2/domains/base.py`)
- [ ] Update `DomainConfig` base class:
    - [ ] Add `save_to_forest(forest: Forest)`:
        - Creates a node with `id=f"config_{self.domain_type}"`.
        - Stores `S_DIM`, `DIM`, and `domain_type` in its `metadata`.
        - Adds all `concepts` as ordered children to this node to preserve the manifold index.
        - Registers the node in the forest.
    - [ ] Add `classmethod load_from_forest(forest: Forest) -> DomainConfig`:
        - Searches for any node with a `config_` prefix and `domain_config` relation type.
        - Reconstructs the `concepts` list from the node's ordered children.
        - Returns a fully populated instance of the appropriate subclass.

### 3. Text Domain Specialization (`hpm_ai_v2/domains/text_domain.py`)
- [ ] Override `save_to_forest` and `load_from_forest` in `TextDomainConfig`:
    - [ ] Store `idf` values inside the metadata of each concept's HFN node (e.g., `word_macro` or `prior_rule`).
    - [ ] Capture `include_char_primitives` flag in the config node metadata.
    - [ ] Reconstruct the `idf` dictionary during load.

### 4. Base Agent Bootstrapping (`hpm_ai_v2/agents/base_agent.py`)
- [ ] Update `BaseHFNAgent.__init__`:
    - [ ] If `config` is provided, proceed normally.
    - [ ] If `config` is `None` and a `forest` (or `cold_dir`) is provided, attempt to call `DomainConfig.load_from_forest(self.forest)`.
- [ ] Update `save_state` to call `self.config.save_to_forest(self.forest)` before the final forest sync.

### 5. Verification Experiment (`hpm_ai_v2/experiments/experiment_sp_web3_fractal_config.py`)
- [ ] Run a Research Marathon to build a large forest (e.g., Topic: "Fractal").
- [ ] **Shutdown:** Save the forest and verify that `config.pkl` is **not** created/used.
- [ ] **Bootstrap:** Re-initialize the Triad of agents by pointing them *only* at the `cold_dir` without providing an initial `config`.
- [ ] **Verify:** Ensure the agents correctly reconstruct their 200+ dimension vocabulary and can answer questions using the accumulated knowledge.

## Verification
- [ ] Run `hpm_ai_v2/experiments/experiment_sp_web3_fractal_config.py`.
- [ ] Confirm `data/research_marathon_forest/config.pkl` is no longer required for loading.
- [ ] Confirm `forest_meta.json` exists and contains the correct `D`.
- [ ] Verify that all domain types (Image, Audio, etc.) could eventually adopt this base class pattern.