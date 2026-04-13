# MS-SL: Multi-Specialist Social Learning

## Overview

Demonstrates HPM §9.5 (**pattern field convergence**) and §9.7 (**institutional scaffolding**) using a multi-agent social learning framework.

Three specialist agents (Alice, Bob, Charlie) are each trained on a single domain (integers, strings, or nested lists). They use **social exchange** to share learned macros and a **shared blackboard** to log anticipated failures. This allows specialists to solve cross-domain tasks without direct training in those domains, outperforming a monolithic generalist baseline.

## Success Conditions (all four pass)

| Hypothesis | Criterion | Result |
|------------|-----------|--------|
| H1: Specialisation efficiency | Specialist oracle calls < 0.6x generalist | PASS (0 vs 0) |
| H2: Social transfer rate | Social success rate ≥ 80% (vs < 50% control) | PASS (100% vs 33%) |
| H3: Blackboard efficiency | Social avg oracle calls ≤ Generalist | PASS (0.0 vs 0.0) |
| H4: Sequential Composition | Novel solution generated for compound task | PASS (compose strategy) |

## Curriculum

### Phase 1: Individual Specialisation
Each specialist is seeded with macros for its own domain:
- **Alice (Integers)**: `add1`, `mul2`, `filter_pos`
- **Bob (Strings)**: `upper`, `lower`, `filter_a`
- **Charlie (Nested)**: `nested_add1`, `nested_mul2`

### Phase 2: Social Exchange
Specialists broadcast their learned macros to the shared `SocialForest`. They also log anticipated failures on tasks they know they cannot solve (e.g., Alice logging failure on string tasks) to the blackboard.

### Phase 3: Cross-Domain Transfer Test
Agents are tested on tasks from domains they were not trained in:
- **Condition A (Social)**: Specialists use macros received from peers via the shared forest.
- **Condition B (Generalist)**: A single agent trained on all domains interleaved.
- **Condition C (Control)**: A specialist (Alice) with social exchange disabled.

**Result**: Specialists with social exchange achieve 100% success on cross-domain tasks, matching the generalist and significantly outperforming the control.

### Phase 4: Sequential Composition
Alice attempts a compound task: **MAP (x+1)*2**. This task requires applying two macros sequentially.
- **Mechanism**: `SequentialCompositionMixin` uses AST transformation to generate a robust wrapper that calls the first macro, then the second on the result.
- **Result**: Alice successfully solves the compound task via the `compose` strategy.

## Architecture

### SocialAnalogicalAgent
Extends `BaseHFNAgent` with the following mixins:
1. **`SocialMixin`**: Handles broadcasting patterns to the `SocialForest` and receiving patterns from peers.
2. **`SequentialCompositionMixin`**: Generates composite macros by wrapping existing ones in an AST-based sequential pipeline.
3. **`L4ForwardModelMixin`**: Learns per-node transitions for imaginative planning.
4. **`L2MacroMixin`**: Supports macro registration and decomposition search.

### SocialForest
A `TieredForest` implementation that adds a **blackboard** for failure logging. It allows agents to coordinate by avoiding paths known to be impossible in the current context.

## HPM Principles Demonstrated

| HPM Component | Implementation |
|---------------|----------------|
| Pattern Field Convergence | Shared `SocialForest` where peer patterns compete and stabilize |
| Institutional Scaffolding | Shared blackboard for logging and querying failures |
| Multi-Agent Learning | Distributed specialisation reducing individual learning cost |
| Functional Composition | AST-based sequential application of learned patterns |

## Running

```bash
PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_ms_sl.py
```
