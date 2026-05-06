# Cross-Language Transfer (CLT) Benchmark Plan

## Objective
Evaluate the HPM core's ability to learn structural invariants in one programming language (e.g., Python) and apply them zero-shot or few-shot to another language (e.g., Java). This will prove that HPM's symbolic abstraction transcends surface syntax.

## 1. Adapters (`hpm_ai_v5/adapter/clt.py`)
To bridge the syntactic gap between languages, we need adapters that map language-specific constructs into a shared canonical vocabulary:

*   **`LanguageDetector`**: Analyzes the raw code string to determine the source language (e.g., `python`, `java`).
*   **`UnifiedTokenizer`**: Tokenizes the code based on the detected language and maps common operators (e.g., `+`, `-`, `==`, `is`) to shared universal token IDs.
*   **`UnifiedASTFlattener`**: Linearizes the AST. We will use the `tree-sitter` library (with `tree-sitter-python` and `tree-sitter-java`) to provide a standardized, cross-language parsing interface. This eliminates the need for custom pseudo-parsers and inherently aligns similar structural nodes (e.g., mapping both to `if_statement`).
*   **`UnifiedCanonicalRenamer`**: 
    *   Renames user-defined variables/functions to `SYM_X`.
    *   Maps language-specific keywords to a shared meta-vocabulary:
        *   `None` (Py), `null` (Java) $\rightarrow$ `NULL_VAL`
        *   `with` (Py), `try` (Java with resources) $\rightarrow$ `RESOURCE_BLOCK`
        *   `except` (Py), `catch` (Java) $\rightarrow$ `CATCH_BLOCK`

## 2. Polygraphs (`hpm_ai_v5/polygraphs/clt.py`)
*   **`CLTPolygraphGenerator`**: Generates views similar to the Code benchmark (`ast_types`, `skeleton`, `token_types`), but using the mapped universal IDs from the CLT adapters.

## 3. Benchmark Tasks (`hpm_ai_v5/planning/clt.py`)
Implement `CrossLanguageTransferBenchmark` with the following evaluations:

### 3.1. Resource Management Transfer
*   **Train (Python)**: `with open(f) as fd: data = fd.read()`
*   **Test (Java)**: `try (BufferedReader br = new BufferedReader(...)) { String data = br.readLine(); }`
*   **Target**: The Pattern Engine must match the resource acquisition/release sequence.

### 3.2. Null Check + Exception Transfer
*   **Train (Python)**: `if x is None: raise ValueError("error")`
*   **Test (Java)**: `if (x == null) throw new IllegalArgumentException("error");`
*   **Target**: Recognize the pattern `U_IF` $\rightarrow$ `U_NULL` $\rightarrow$ `U_THROW`.

### 3.3. Map Operation Transfer
*   **Train (Python)**: `[x*2 for x in items]`
*   **Test (Java)**: `items.stream().map(x -> x*2).collect(Collectors.toList());`
*   **Target**: Map both structures to a universal `MAP_TRANSFORM` invariant.

## 4. Evaluation Protocol
*   **Zero-Shot Transfer**: Train on $N=10$ Python examples. Test on $M=5$ Java examples immediately. Success: $>80\%$ accuracy (patterns matched).
*   **Few-Shot Fine-Tuning**: If zero-shot fails, provide $K<5$ Java examples as reward-shaped training, then re-test. Success: $>90\%$ accuracy.

## 5. Execution Harness (`hpm_ai_v5/experiments/run_clt_benchmark.py`)
Wire the new pipeline and execute the three tasks, comparing zero-shot and few-shot metrics.