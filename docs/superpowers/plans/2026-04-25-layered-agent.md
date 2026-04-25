# Layered Agent (L1+L2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LayeredAgent` wrapping L1 (obs_dim=5, char classes) and L2 (obs_dim=95, actual chars) so simulations produce decoded character output.

**Architecture:** `LayeredAgent` holds two `HPMAgent` instances; `perceive(raw_char_id)` feeds both every step; `generate(steps)` samples from L2 and decodes to actual characters. `full_simulation.py` and `experiment_generative_output.py` are updated to use it. Checkpoints use `PatternSerializer` (existing codebase tool).

**Tech Stack:** Python 3.10+, numpy, hpm_ai_v4 (HPMAgent, HierarchicalPattern, FlatPattern, CharClassAdapter, PatternSerializer)

---

## File Structure

- **Create:** `hpm_ai_v4/simulations/layered_agent.py` — `LayeredAgent` class + `_init_equal_weights` helper
- **Create:** `hpm_ai_v4/tests/test_layered_agent.py` — unit tests
- **Modify:** `hpm_ai_v4/simulations/full_simulation.py` — replace direct HPMAgent with LayeredAgent
- **Modify:** `hpm_ai_v4/simulations/experiment_generative_output.py` — use LayeredAgent, decode L2 to chars

---

### Task 1: LayeredAgent core

**Files:**
- Create: `hpm_ai_v4/simulations/layered_agent.py`
- Create: `hpm_ai_v4/tests/test_layered_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# hpm_ai_v4/tests/test_layered_agent.py
from hpm_ai_v4.simulations.layered_agent import LayeredAgent

def test_layered_agent_perceive_runs():
    agent = LayeredAgent(num_workers=1)
    for i in range(20):
        agent.perceive(i % 95)

def test_layered_agent_obs_dims():
    agent = LayeredAgent(num_workers=1)
    assert agent.l1.obs_dim == 5
    assert agent.l2.obs_dim == 95

def test_layered_agent_equal_weights():
    agent = LayeredAgent(num_workers=1)
    assert max(p.weight for p in agent.l1.patterns) < 0.5
    assert max(p.weight for p in agent.l2.patterns) < 0.5

def test_layered_agent_generate_returns_string():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    assert isinstance(result, str)
    assert len(result) <= 20

def test_layered_agent_generate_printable():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    assert all(32 <= ord(ch) <= 126 for ch in result)

def test_predict_next_chars_returns_list():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    preds = agent.predict_next_chars(list(range(10)), top_k=5)
    assert len(preds) <= 5
    assert all(isinstance(ch, str) and isinstance(prob, float) for ch, prob in preds)
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_layered_agent.py -v
```
Expected: `ImportError` — layered_agent doesn't exist yet.

- [ ] **Step 3: Implement LayeredAgent**

```python
# hpm_ai_v4/simulations/layered_agent.py
"""LayeredAgent: L1 (char classes, obs_dim=5) + L2 (raw chars, obs_dim=95)."""
from typing import List, Tuple
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern


def _init_equal_weights(agent: HPMAgent, hier_k: int, obs_dim: int) -> None:
    """Replace agent.patterns with equal-weight hier+flat population."""
    agent.patterns = []
    for i in range(4):
        p = HierarchicalPattern(i, latent_dim=hier_k, obs_dim=obs_dim)
        p.weight = 0.15
        agent.patterns.append(p)
    for i in range(4, 6):
        p = FlatPattern(i, obs_dim=obs_dim)
        p.weight = 0.10
        agent.patterns.append(p)


class LayeredAgent:
    """Two-level HPM agent: L1 learns char-class structure, L2 learns actual chars."""

    def __init__(self, num_workers: int = 1):
        self._adapter = CharClassAdapter()
        self.l1 = HPMAgent(obs_dim=5, num_initial_patterns=4, num_workers=num_workers)
        self.l2 = HPMAgent(obs_dim=95, num_initial_patterns=4, num_workers=num_workers)
        _init_equal_weights(self.l1, hier_k=2, obs_dim=5)
        _init_equal_weights(self.l2, hier_k=4, obs_dim=95)

    def perceive(self, raw_char_id: int) -> None:
        """Feed one character to both levels."""
        class_id = self._adapter.encode(raw_char_id)
        self.l1.perceive_and_learn(class_id)
        self.l2.perceive_and_learn(raw_char_id)

    def generate(self, steps: int = 80) -> str:
        """Sample from L2 and decode to printable characters."""
        future = self.l2.reasoner.simulate_future(steps=steps, top_k=3)
        return "".join(chr(v + 32) for v in future if 0 <= v <= 94)

    def predict_next_chars(self, context_raw: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Top-k next character predictions from L2."""
        relevant = self.l2.reasoner.get_relevant_patterns(context_raw, top_k=top_k)
        if not relevant:
            return []
        dist = self.l2.reasoner.compose_predictions(relevant, context_raw)
        top = np.argsort(dist)[::-1][:top_k]
        return [(chr(i + 32), float(dist[i])) for i in top]

    def l1_metrics(self) -> dict:
        """Summary metrics for L1 population."""
        top3 = sorted(self.l1.patterns, key=lambda p: -p.weight)[:3]
        mi = float(np.mean([p.compression() for p in top3])) if top3 else 0.0
        return {
            'pop_size': len(self.l1.patterns),
            'mi': mi,
            'stage': self.l1.development.level,
            'best_weight': max(p.weight for p in self.l1.patterns) if self.l1.patterns else 0.0,
        }

    def l2_metrics(self, recent_raw: List[int]) -> dict:
        """Prediction accuracy of L2 over recent char buffer."""
        correct = 0
        total = max(1, len(recent_raw) - 1)
        for i in range(len(recent_raw) - 1):
            ctx = recent_raw[max(0, i - 20):i]
            actual = recent_raw[i + 1]
            relevant = self.l2.reasoner.get_relevant_patterns(ctx, top_k=3)
            if relevant:
                dist = self.l2.reasoner.compose_predictions(relevant, ctx)
                if int(np.argmax(dist)) == actual:
                    correct += 1
        return {
            'accuracy': correct / total,
            'pop_size': len(self.l2.patterns),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest hpm_ai_v4/tests/test_layered_agent.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v4/simulations/layered_agent.py hpm_ai_v4/tests/test_layered_agent.py
git commit -m "feat: LayeredAgent with L1 (obs_dim=5) + L2 (obs_dim=95) and generate()"
```

---

### Task 2: Wire LayeredAgent into full_simulation.py

**Files:**
- Modify: `hpm_ai_v4/simulations/full_simulation.py`
- Modify: `hpm_ai_v4/tests/test_full_simulation.py`

- [ ] **Step 1: Write failing test**

```python
# append to hpm_ai_v4/tests/test_full_simulation.py
def test_run_simulation_has_l2_accuracy(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox " * 30)
    history = run_simulation(
        corpus_path=str(corpus),
        total_steps=300,
        log_every=100,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    assert 'l2_accuracy' in history[-1]
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py::test_run_simulation_has_l2_accuracy -v
```
Expected: FAIL — `l2_accuracy` not in snapshot.

- [ ] **Step 3: Add LayeredAgent import to full_simulation.py**

At the top of `hpm_ai_v4/simulations/full_simulation.py`, add:

```python
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
```

Remove these imports (no longer needed directly in run_simulation):
```python
from hpm_ai_v4.io.adapters import CharClassAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
```

- [ ] **Step 4: Replace agent construction in run_simulation**

Remove the block that builds HPMAgent with equal weights and replace with:

```python
    layered = LayeredAgent(num_workers=num_workers)

    if library_path and os.path.exists(library_path):
        from hpm_ai_v4.tools.serializer import PatternSerializer
        layered.l1.patterns = PatternSerializer.load(library_path + ".l1.pkl")
        layered.l2.patterns = PatternSerializer.load(library_path + ".l2.pkl")
        print(f"Loaded library from {library_path}")
```

- [ ] **Step 5: Replace per-step encoding in the loop**

Replace:
```python
        raw_id = next(stream_iter)
        char_id = adapter.encode(raw_id)
        accuracy_buffer.append(char_id)
        ...
        agent.perceive_and_learn(char_id)
```

With:
```python
        raw_id = next(stream_iter)
        accuracy_buffer.append(raw_id)
        if len(accuracy_buffer) > log_every + 21:
            accuracy_buffer = accuracy_buffer[-(log_every + 21):]
        layered.perceive(raw_id)
```

- [ ] **Step 6: Update _metrics_snapshot signature and body**

```python
def _metrics_snapshot(layered: 'LayeredAgent', recent_chars: List[int], step: int) -> Dict[str, Any]:
    snap: Dict[str, Any] = {'step': step}

    # L1 metrics
    m1 = layered.l1_metrics()
    snap['compression_mi'] = m1['mi']
    snap['pop_size'] = m1['pop_size']
    snap['best_weight'] = m1['best_weight']
    snap['dev_stage'] = m1['stage']
    snap['best_loss'] = float(min(p.running_loss for p in layered.l1.patterns)) if layered.l1.patterns else 0.0

    # L1 accuracy (class level)
    adapter = layered._adapter
    correct = 0
    total = max(1, len(recent_chars) - 1)
    for i in range(len(recent_chars) - 1):
        ctx_cls = [adapter.encode(v) for v in recent_chars[max(0, i - 20):i]]
        actual_cls = adapter.encode(recent_chars[i + 1])
        relevant = layered.l1.reasoner.get_relevant_patterns(ctx_cls, top_k=3)
        if relevant:
            dist = layered.l1.reasoner.compose_predictions(relevant, ctx_cls)
            if int(np.argmax(dist)) == actual_cls:
                correct += 1
    snap['accuracy'] = correct / total

    # L2 accuracy (raw char, sampled over last 200 for speed)
    sample = recent_chars[-200:] if len(recent_chars) > 200 else recent_chars
    m2 = layered.l2_metrics(sample)
    snap['l2_accuracy'] = m2['accuracy']
    snap['l2_pop_size'] = m2['pop_size']
    snap['word_completion'] = None
    return snap
```

- [ ] **Step 7: Update log line**

```python
            print(
                f"[step {step:6d}] "
                f"L1 acc={snap['accuracy']:.3f} mi={snap['compression_mi']:.3f} "
                f"pop={snap['pop_size']} stage={snap['dev_stage']} | "
                f"L2 acc={snap['l2_accuracy']:.3f} pop={snap['l2_pop_size']}"
            )
```

- [ ] **Step 8: Update checkpoint saving to use PatternSerializer**

```python
        if step % 10_000 == 0 and step > 0:
            from hpm_ai_v4.tools.serializer import PatternSerializer
            base = os.path.join(checkpoint_dir, f"checkpoint_{step}")
            PatternSerializer.save(layered.l1.patterns, base + ".l1.pkl")
            PatternSerializer.save(layered.l2.patterns, base + ".l2.pkl")
            print(f"  [checkpoint saved: {base}.l1.pkl + .l2.pkl]")
```

And final save:
```python
    from hpm_ai_v4.tools.serializer import PatternSerializer
    base = os.path.join(checkpoint_dir, "final_library")
    PatternSerializer.save(layered.l1.patterns, base + ".l1.pkl")
    PatternSerializer.save(layered.l2.patterns, base + ".l2.pkl")
    print(f"Final library saved: {base}.l1.pkl + .l2.pkl")
```

- [ ] **Step 9: Run all tests**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v
```
Expected: all tests PASS.

- [ ] **Step 10: Smoke test**

```bash
python3 -m hpm_ai_v4.simulations.full_simulation \
    --corpus hpm_ai_v4/simulations/data/wiki_sample.txt \
    --steps 2000 --log-every 500 --workers 1
```
Expected: log lines show both `L1 acc=` and `L2 acc=` fields, no errors.

- [ ] **Step 11: Commit**

```bash
git add hpm_ai_v4/simulations/full_simulation.py hpm_ai_v4/tests/test_full_simulation.py
git commit -m "feat: full_simulation uses LayeredAgent, adds L2 accuracy to metrics"
```

---

### Task 3: Update experiment_generative_output.py

**Files:**
- Modify: `hpm_ai_v4/simulations/experiment_generative_output.py`

- [ ] **Step 1: Replace entire file content**

```python
#!/usr/bin/env python3
"""Generative Output Experiment — uses LayeredAgent for actual character output."""
import argparse
import numpy as np
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.full_simulation import WikipediaStream

CLASS_NAMES = ['letter', 'digit', 'space', 'punct', 'newline']


def run_generative_demo(corpus_path: str, steps: int = 10000):
    print(f"\n[output] Generative demo: {corpus_path!r} ({steps} steps)")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)
    tokens = [next(stream_iter) for _ in range(steps + 100)]

    layered = LayeredAgent(num_workers=1)

    print(f"[learn] Training on {steps} tokens...")
    for i in range(steps):
        layered.perceive(tokens[i])
        if (i + 1) % 2000 == 0:
            print(f"  Step {i+1}...")

    print("\n" + "=" * 50)
    print("DEMONSTRATING OUTPUT CAPABILITIES")
    print("=" * 50)

    # 1. L2 Character Prediction
    context_raw = tokens[steps - 20:steps]
    context_str = "".join(chr(v + 32) for v in context_raw if 0 <= v <= 94)
    print(f"\n[1. Prediction] Context: '{context_str}'")
    preds = layered.predict_next_chars(context_raw, top_k=5)
    print("  Next character predictions (L2):")
    for ch, prob in preds:
        print(f"    {repr(ch)}: {prob*100:.1f}%")

    # 2. L2 Character Generation
    print(f"\n[2. Generation] L2 generates 80 characters:")
    generated = layered.generate(steps=80)
    print(f"  '{generated}'")
    words = [w for w in generated.split() if len(w) >= 2]
    print(f"  Words (len>=2): {words[:10]}")

    # 3. L1 Pattern Explanation
    print(f"\n[3. L1 Patterns] Dominant char-class patterns:")
    top3_l1 = sorted(layered.l1.patterns, key=lambda p: -p.weight)[:3]
    for i, p in enumerate(top3_l1):
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_cls = CLASS_NAMES[min(top_obs, 4)]
        print(f"  L1 Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts '{top_cls}'  MI={p.compression():.3f}")

    # 4. L2 Pattern Explanation
    print(f"\n[4. L2 Patterns] Dominant char-level patterns:")
    top3_l2 = sorted(layered.l2.patterns, key=lambda p: -p.weight)[:3]
    for i, p in enumerate(top3_l2):
        top_obs = int(np.argmax(p.B[np.argmax(p.pi)]))
        top_ch = repr(chr(top_obs + 32)) if 0 <= top_obs <= 94 else '?'
        print(f"  L2 Pattern {i+1} (weight={p.weight:.3f} K={p.latent_dim}): "
              f"predicts {top_ch}  MI={p.compression():.3f}")

    # 5. Planning (find space after "the")
    prefix_raw = [ord(ch) - 32 for ch in "the"]
    goal = ord(' ') - 32  # space = 0
    print(f"\n[5. Planning] From 'the' find space (horizon=6, rollouts=30):")
    layered.l2.obs_buffer = list(prefix_raw)
    plan = layered.l2.reasoner.plan(goal_state=goal, horizon=6, num_rollouts=30,
                                     require_valid_words=False, require_grammatical=False)
    plan_str = "".join(chr(v + 32) for v in plan if 0 <= v <= 94)
    reached = (plan[-1] == goal) if plan else False
    print(f"  Plan: 'the{plan_str}'  reached_space={'YES' if reached else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative Output Experiment")
    parser.add_argument('--corpus', default="hpm_ai_v4/simulations/data/wiki_sample.txt")
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()
    run_generative_demo(args.corpus, steps=args.steps)
```

- [ ] **Step 2: Run smoke test**

```bash
python3 -m hpm_ai_v4.simulations.experiment_generative_output \
    --corpus hpm_ai_v4/simulations/data/wiki_sample.txt --steps 5000
```
Expected: output shows actual character predictions and generated string. L2 patterns show K>1 patterns with MI>0.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v4/simulations/experiment_generative_output.py
git commit -m "feat: generative demo uses LayeredAgent, L2 decodes to actual characters"
```

---

## Self-Review

**Spec coverage:**
- ✅ `LayeredAgent` with L1 (obs_dim=5, K=2) + L2 (obs_dim=95, K=4) — Task 1
- ✅ `_init_equal_weights` helper — Task 1
- ✅ `perceive(raw_char_id)` feeds both levels — Task 1
- ✅ `generate(steps) -> str` decodes L2 output — Task 1
- ✅ `predict_next_chars` returns actual chars with probabilities — Task 1
- ✅ `l1_metrics()` and `l2_metrics()` — Task 1
- ✅ `full_simulation.py` uses LayeredAgent — Task 2
- ✅ Metrics snapshot includes `l2_accuracy` — Task 2
- ✅ Checkpoint saves both L1 and L2 via PatternSerializer — Task 2
- ✅ `experiment_generative_output.py` uses LayeredAgent — Task 3
- ✅ Generation decodes to actual chars — Task 3
- ✅ Planning uses L2 agent — Task 3

**No placeholders. Type consistency verified.**
