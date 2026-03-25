# Relational Bundle and Structural Message Plan

## Implementation Sequence

1. Add relational primitives.
- Create typed dataclasses for relational edges, bundles, and structural messages.
- Keep them small, serializable, and baseline-friendly.

2. Extend hierarchical helpers.
- Add extraction and encoding helpers for relational bundles.
- Keep existing `LevelBundle` and `encode_bundle()` behavior intact.

3. Extend `PatternField`.
- Add dedicated message queue methods for structural messages.
- Do not alter pattern broadcast behavior.

4. Extend `Agent` and `MultiAgentOrchestrator`.
- Add optional structural message emission and receipt hooks.
- Gate relay behind a constructor flag so current behavior stays unchanged.

5. Add tests.
- relational bundle extraction and encoding
- field message queue isolation
- orchestrator relay of structural messages when enabled

6. Run targeted verification.
- baseline structured/hierarchical/field communication tests
- new tests for relational messaging

## Acceptance Checks

- `LevelBundle` path remains backward compatible.
- Structural message queue is isolated from pattern broadcast queue.
- Agents can receive another agent’s structural message when enabled.
- Existing communication tests still pass unchanged.
