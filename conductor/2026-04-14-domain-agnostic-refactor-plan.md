# Domain-Agnostic Agent Layer Refactoring Plan

## Objective
Refactor the agent layer to remove hardcoded, domain-specific global constants (e.g., `S_DIM`, `DIM`, `CONCEPTS` from `hpm_ai_v2/utils/state.py`). Introduce a `DomainConfig` class to encapsulate domain semantics, ensuring that the core HFN and the base agent framework are entirely domain-agnostic and can be instantiated for arbitrary domains (e.g., lists, math, physics).

## Motivation
Currently, `hfn/` and agent utilities import global constants directly from `hpm_ai_v2/utils/state.py`. This tightly couples the generic reasoning architecture to the specific list-transformation domain used in earlier experiments. To scale HPM to new domains (like the Collaborative Math-Physics experiment) and to maintain architectural purity, domain knowledge must be injected dynamically via configuration rather than statically imported.

## Implementation Steps

### 1. Introduce `DomainConfig`
**File:** `hpm_ai_v2/domains/base.py` (New File)
- Define the `DomainConfig` class to encapsulate:
  - `concepts`: List of string concept names.
  - `concept_idx`: Mapping of concept name to index.
  - `DIM`: Length of concepts list.
  - `S_DIM`: State vector dimension (default 20).
  - `m_dim`: Total vector dimension (`S_DIM + DIM + S_DIM`).
  - Helper methods like `get_concept_vector(concept: str)`.

### 2. Create Specific Domain Configurations
**Files:** `hpm_ai_v2/domains/list_domain.py`, `hpm_ai_v2/domains/math_physics_domain.py`
- Extract the current list-processing concepts from `state.py` into a `ListDomainConfig`.
- Create a `MathPhysicsDomainConfig` that includes a superset of concepts required for the Math and Physics agents (e.g., adding `OP_SQUARE`, `OP_SQRT`, `OP_DIV2`).

### 3. Update Agent Initialization
**File:** `hpm_ai_v2/agents/base_agent.py` and subclass definitions.
- Modify `BaseHFNAgent.__init__` to accept a `config: DomainConfig` parameter.
- Store `self.config`, `self.s_dim`, `self.dim`, and `self.m_dim` derived from the config.
- Pass the `config` to utility classes instantiated by the agent (e.g., `ASTRenderer`, `EmpiricalOracle`).

### 4. Update Renderers and Oracles
**Files:** `hpm_ai_v2/utils/renderer.py`, `hpm_ai_v2/utils/oracle.py`
- Modify `ASTRenderer.__init__` to accept `config` and use `config.concepts` and `config.S_DIM` instead of global imports.
- Update `EmpiricalOracle` and `CountingOracle` to utilize `config.S_DIM` and `config.DIM` for vector slicing and state computation.

### 5. Refactor Experiments and Remove `state.py`
**Files:** All scripts in `hpm_ai_v2/experiments/`
- Update experiment scripts to instantiate the appropriate `DomainConfig` and pass it to the agents.
- Remove all imports from `hpm_ai_v2.utils.state`.
- Once all dependencies are removed, deprecate or delete `hpm_ai_v2/utils/state.py`.

## Backward Compatibility
Existing saved forests from list-domain experiments (SP61-SP67) rely on specific dimensions (`S_DIM=20`, `DIM=14` or `17`). The `ListDomainConfig` will precisely match these dimensions, ensuring that legacy cold storage nodes can still be loaded and processed without error.

## Verification
- Run `pytest` to ensure all unit tests pass with the new configuration injection.
- Execute `experiment_ms_sl.py` (using `ListDomainConfig`) and `experiment_math_physics.py` (using `MathPhysicsDomainConfig`) to confirm that multiple domains can be handled seamlessly by the same underlying agent architecture.