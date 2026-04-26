# HPM AI AGI Roadmap Implementation Plan

> **For agentic workers:** implement one milestone at a time. Keep the HPM alignment constraints active while progressing.

**Goal:** extend HPM AI from a narrow text stack into a general predictive-control system without introducing a hard controller or a separate memory engine.

**Architecture:** reuse the existing `l1-l5` stack, the multi-polygraph episodic store, the reasoner, and the learned meta-policy. Add one new environment/task simulation per milestone, plus a held-out transfer benchmark for each.

---

## Milestone 1: Ground the stack in environment state

**Deliverable:** an interactive non-text environment simulation with action/outcome episodes stored in the polygraph.

- [ ] Define the episode schema for state, action, outcome, stage, and policy.
- [ ] Add a simple environment simulation that emits observations and accepts actions.
- [ ] Record environment episodes through the existing polygraph path.
- [ ] Add a held-out validation run that compares before/after reload behavior.
- [ ] Add tests that prove the polygraph retrieval changes with repeated environment families.

**Exit criteria**
- The stack learns something outside text.
- The learned state survives reload.
- Retrieval is still soft and resonance-based.

---

## Milestone 2: Add tool and action use

**Deliverable:** a planning benchmark where the reasoner chooses and sequences external actions.

- [ ] Define a small tool/action interface.
- [ ] Teach the reasoner to score candidate action sequences.
- [ ] Let the meta-policy choose among action families using polygraph priors.
- [ ] Add a simulation that requires multi-step action use to succeed.
- [ ] Add transfer tests across task variants and action families.

**Exit criteria**
- Action choice improves with experience.
- Prior episodes bias planning without hard rules.
- Reload preserves the learned control policy.

---

## Milestone 3: Continual learning and self-curriculum

**Deliverable:** a curriculum benchmark where the stack selects useful next tasks and avoids catastrophic collapse.

- [ ] Add task-family metadata to episodes.
- [ ] Make the meta-policy rank tasks by expected transfer gain.
- [ ] Add a curriculum simulation that cycles through task families.
- [ ] Measure held-out transfer, not just in-domain accuracy.
- [ ] Decay bootstrap influence until learned selection dominates.

**Exit criteria**
- The system can choose what to practice next.
- New tasks improve transfer rather than displacing old skills.
- The policy remains stable across reload and phase changes.

---

## HPM Alignment Checks

Before each merge:

- [ ] No hard-coded controller was introduced.
- [ ] No dedicated memory engine was added.
- [ ] Bootstrap remains a temporary prior only.
- [ ] The new capability is validated on held-out data.
- [ ] The learned policy survives bundle reload.

If any check fails, stop and reduce scope.
