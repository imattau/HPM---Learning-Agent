# Oracle Refactoring Plan

## Objective
Refactor the single `hpm_ai_v2/utils/oracle.py` file into a modular, domain-agnostic directory structure (`hpm_ai_v2/utils/oracle/`).

## Scope
- Extract `EmpiricalOracle` (renamed to `ListOracle`), `ImageOracle`, and `AudioOracle` into their own files.
- Introduce an abstract `BaseOracle` interface.
- Change `CountingOracle` from an inheritance-based class to a wrapper/decorator pattern.

## Proposed Solution

1. **Create Directory Structure:**
   - `hpm_ai_v2/utils/oracle/`
     - `__init__.py`
     - `base.py`
     - `list_oracle.py`
     - `image_oracle.py`
     - `audio_oracle.py`

2. **Implement Components:**
   - **`base.py`:** Define `BaseOracle` ABC with `compute_state`. Define `CountingOracle` that accepts a `BaseOracle` instance.
   - **`list_oracle.py`:** Port `EmpiricalOracle` logic here, renaming it to `ListOracle`.
   - **`image_oracle.py`:** Port `ImageOracle` logic here. Remove `call_count` as it will be handled by the wrapper.
   - **`audio_oracle.py`:** Port `AudioOracle` logic here. Remove `call_count`.
   - **`__init__.py`:** Export all classes for backward compatibility (where feasible) and easy imports.

3. **Update Codebase:**
   - Update `BaseHFNAgent` in `hpm_ai_v2/agents/base_agent.py` to import `ListOracle` and wrap it with `CountingOracle` (e.g., `self.counting_oracle = CountingOracle(ListOracle(config))`).
   - Replace usages of `EmpiricalOracle` with `ListOracle` across the project.
   - Update experiment scripts to use the wrapper pattern for their respective oracles.
   - Remove the old `hpm_ai_v2/utils/oracle.py`.

## Verification
- Run SP71 and SP73 experiments to ensure they still pass.
- Run a baseline benchmark (e.g., ARC or synthesis) if applicable, or rely on SP71/SP73 validation to confirm `ListOracle` and the new `CountingOracle` work seamlessly.
