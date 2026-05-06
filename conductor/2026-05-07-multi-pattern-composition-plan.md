# Multi-Pattern Composition (MPC) Benchmark Plan

## Objective
Evaluate the HPM core's ability to combine multiple learned structural patterns (e.g., Resource Management + Null Check) into a larger, coherent structure. This demonstrates hierarchical pattern composition—a core claim of HPM.

## 1. Agent-Layer Components (`hpm_ai_v5/planning/mpc.py`)

### `PatternComposer`
*   **Goal**: Given a natural language instruction or a list of required sub-goals (e.g., `["safe_file_read"]`), retrieve relevant patterns from the `PatternManager` and assemble them into a cohesive AST sequence.
*   **Mechanism**: 
    1.  Query the `PatternManager` for patterns relevant to the required sub-goals (e.g., `U_RESOURCE`, `U_NULL`, `U_THROW`).
    2.  Use the `PatternEngine`'s sequence generation or a beam search to find a sequence of Universal nodes that correctly integrates these patterns.
    3.  Emit the composed sequence (e.g., `U_TRY`, `U_RESOURCE`, `U_IF`, `U_NULL`, `U_THROW`, `U_CATCH`).

### `StructuralValidator`
*   **Goal**: Ensure the composed pattern sequence is structurally valid in the target language.
*   **Mechanism**: Post-processes the composed sequence. For example, if a `U_TRY` is present, there must be a `U_CATCH`. If an invariant is incomplete, it penalizes the composition (feedback via `CLTRefinementAdapter` mechanisms) and rejects it.

## 2. Benchmark Tasks (`hpm_ai_v5/planning/mpc.py`)

### 2.1 Safe File Read
*   **Source (Python Train)**:
    ```python
    with open('f.txt') as f:
        data = f.read()
        if data is None:
            raise ValueError('empty')
    ```
*   **Target (Java Test)**: Generate a sequence matching Java's try-with-resources containing a null check and exception throw.
*   **Patterns to Compose**: Resource Management (`U_RESOURCE`) + Null Check (`U_NULL`) + Exception Handling (`U_THROW`).

### 2.2 Map with Filter
*   **Source (Python Train)**:
    ```python
    res = [x*2 for x in items if x is not None]
    ```
*   **Target (Java Test)**: Generate a sequence matching Java Streams: `items.stream().filter(x -> x != null).map(x -> x * 2).collect(...)`
*   **Patterns to Compose**: Map Operation (`MAP_TRANSFORM`) + Filter Operation (`FILTER_TRANSFORM`).

### 2.3 Retry Wrapper
*   **Source (Python Train)**:
    ```python
    for _ in range(3):
        try:
            do_work()
            break
        except:
            time.sleep(1)
    ```
*   **Target (Go Test)**: A `for` loop with a retry/error check and delay.
*   **Patterns to Compose**: Loop (`U_FOR` / `U_WHILE`) + Exception/Error Check (`U_CATCH` / `U_IF` `err != nil`).

## 3. Evaluation Protocol
*   **Zero-Shot Composition**: Test the `PatternComposer` to see if it can produce the combined sequence for the target language after only seeing the individual invariants via CLT, or after seeing a composed sequence in Python.
*   **Metrics**:
    *   **Structural correctness**: Adheres to the target language AST rules.
    *   **Pattern completeness**: All required sub-patterns are present.
    *   **Ordering**: The sub-patterns are correctly nested (e.g., null check *inside* resource block).
    *   **Success Rate**: > 80% required to pass.

## 4. Execution Harness (`hpm_ai_v5/experiments/run_mpc_benchmark.py`)
Executes the `MultiPatternCompositionBenchmark`, reports metrics on correctness, completeness, and ordering, and provides visibility into the `PatternComposer`'s assembly process.