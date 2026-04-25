# Layered Agent (L1+L2) Design

**Date:** 2026-04-25

## Goal

Add a second HPMAgent (L2, obs_dim=95) running in parallel with the existing L1 agent (obs_dim=5) so that simulations produce actual decoded character output rather than class-label sequences.

---

## Architecture

Two `HPMAgent` instances run simultaneously on every character from the stream:

### L1 Agent (existing)
- `CharClassAdapter` encodes raw char IDs (0–94) → 5 char classes
- `obs_dim=5`, `latent_dim=2`, equal initial weights
- Learns fast class-level structure (letter/digit/space/punct/newline transitions)
- Already passing benchmarks: accuracy >76%, MI >0.20 at 50k steps

### L2 Agent (new)
- Receives raw char IDs directly (0–94, no adapter)
- `obs_dim=95`, `latent_dim=4` (higher K for richer char-level patterns), equal initial weights
- 4 HierarchicalPattern (K=4) + 2 FlatPattern, all weight=1/6
- Learns actual character sequences — which letters follow which
- L1's current latent state (`l1_agent.patterns[best].get_top_state(buf)`) is used as a hint to `get_relevant_patterns` context — soft gating, not hard routing
- Decodes predictions: `chr(obs_id + 32)` → actual character

### Data Flow (per step)
```
raw_char_id (0–94)
    ├── CharClassAdapter.encode() → class_id (0–4) → L1.perceive_and_learn()
    └── directly → L2.perceive_and_learn()

For generation:
    L1.get_top_state() → context hint → L2.reasoner.get_relevant_patterns()
    L2.reasoner.simulate_future(steps) → [obs_ids] → "".join(chr(id+32))
```

---

## New File: `hpm_ai_v4/simulations/layered_agent.py`

Single class `LayeredAgent` (~70 lines):

```python
class LayeredAgent:
    def __init__(self, num_workers=1):
        self.adapter = CharClassAdapter()
        self.l1 = HPMAgent(obs_dim=5, num_initial_patterns=4, num_workers=num_workers)
        self.l2 = HPMAgent(obs_dim=95, num_initial_patterns=4, num_workers=num_workers)
        # Equal initial weights for both
        _init_equal_weights(self.l1, hier_k=2, obs_dim=5)
        _init_equal_weights(self.l2, hier_k=4, obs_dim=95)

    def perceive(self, raw_char_id: int):
        class_id = self.adapter.encode(raw_char_id)
        self.l1.perceive_and_learn(class_id)
        self.l2.perceive_and_learn(raw_char_id)

    def generate(self, steps: int = 80) -> str:
        future = self.l2.reasoner.simulate_future(steps=steps, top_k=3)
        return "".join(chr(v + 32) for v in future if 0 <= v <= 94)

    def predict_next_chars(self, context_raw: list, top_k: int = 5):
        relevant = self.l2.reasoner.get_relevant_patterns(context_raw, top_k=top_k)
        dist = self.l2.reasoner.compose_predictions(relevant, context_raw)
        top = np.argsort(dist)[::-1][:top_k]
        return [(chr(i + 32), float(dist[i])) for i in top]

    def l1_metrics(self) -> dict:
        top3 = sorted(self.l1.patterns, key=lambda p: -p.weight)[:3]
        return {
            'pop_size': len(self.l1.patterns),
            'mi': float(np.mean([p.compression() for p in top3])),
            'stage': self.l1.development.level,
        }

    def l2_metrics(self, recent_raw: list) -> dict:
        correct = sum(
            1 for i in range(len(recent_raw) - 1)
            if int(np.argmax(
                self.l2.reasoner.compose_predictions(
                    self.l2.reasoner.get_relevant_patterns(recent_raw[max(0,i-20):i], top_k=3),
                    recent_raw[max(0,i-20):i]
                )
            )) == recent_raw[i + 1]
        )
        return {
            'accuracy': correct / max(1, len(recent_raw) - 1),
            'pop_size': len(self.l2.patterns),
        }
```

`_init_equal_weights(agent, hier_k, obs_dim)` replaces agent.patterns with 4 HierarchicalPattern(latent_dim=hier_k) at weight=0.15 and 2 FlatPattern at weight=0.10.

---

## Changes to Existing Files

### `full_simulation.py`
- Replace `HPMAgent` direct usage with `LayeredAgent`
- `run_simulation` calls `layered.perceive(raw_id)` each step (no adapter call in sim loop)
- Metrics snapshot calls both `layered.l1_metrics()` and `layered.l2_metrics(recent_raw)`
- Log line adds `l2_acc=` field
- Checkpoint saves both `layered.l1.patterns` and `layered.l2.patterns`

### `experiment_generative_output.py`
- Replace separate L1 agent with `LayeredAgent`
- Section 1 (Prediction): use `layered.predict_next_chars(context_raw)` — shows actual chars with %
- Section 2 (Simulation): use `layered.generate(80)` — shows actual decoded string
- Section 3 (Explanation): show L2 top patterns (K, MI, top predicted char)
- Section 4 (Planning): use L2 agent for char-level planning toward goal char

---

## Success Criteria

After 20k steps on wiki_sample.txt:

| Metric | Target |
|--------|--------|
| L1 accuracy (class) | >70% |
| L1 MI | >0.10 |
| L2 accuracy (raw char) | >10% (baseline 1/95 ≈ 1%) |
| `generate(80)` contains common words | "the", "of", "in", "and" present |
| No crashes on 20k step run | ✅ |
