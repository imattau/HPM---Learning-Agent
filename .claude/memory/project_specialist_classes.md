---
name: specialist_class_mapping
description: Mapping of conceptual specialist roles to actual class names in the codebase
type: project
---

The HPM codebase uses different names for what the design docs call "specialist roles":

| Design Role | Actual Class | File |
|-------------|--------------|------|
| Librarian | StructuralLawMonitor | `hpm/monitor/structural_law.py` |
| Innovator | RecombinationStrategist | `hpm/monitor/recombination_strategist.py` |
| Translator | SubstrateBridge | `hpm/substrate/bridge.py` |
| Forecaster | PredictiveSynthesisAgent | `hpm/monitor/predictive_synthesis.py` |
| Actor | DecisionalActor | `hpm/agents/actor.py` |

**Why:** Phase 3 of the Contextual Pattern Store should delegate to these existing classes, NOT create new Librarian/Forecaster classes.

**How to apply:** Before implementing any "specialist role" from a design doc, check this mapping first.
