# Full HPM AI Simulation Design

**Date:** 2026-04-25

## Goal

A single simulation that exercises every HPM v4 feature: `HPMAgent` (Reasoner, `observe_outcome`, lexical reward, DevelopmentalStage, ExternalSubstrate gossip, ParallelPatternPool, replicator dynamics, recombination), dictionary-constrained planning, pattern library load/save, and quantitative benchmarking — all on a real Wikipedia character stream.

---

## Environment

`WikipediaStream` reads a plain-text Wikipedia corpus file, maps each character (ASCII 32–126 → 0–94, newline → 94) to a char ID, and loops on exhaustion. Emits one `int` per step.

Vocab size: **95 symbols** (obs_dim=95).

---

## Agent Configuration

```python
HPMAgent(
    obs_dim=95,
    num_initial_patterns=6,
    num_workers=N,          # CLI arg, default 1
    dictionary=NLTKWordList(),  # optional, requires nltk words corpus
    grammar=None,           # not used in this simulation
)
```

If `library_path` is provided, call `agent.load_library(library_path, reset_weights=True)` before the run. This replaces the initial random patterns with pre-trained ones.

All HPM features activate automatically through `HPMAgent`:
- `perceive_and_learn` → parallel EM, replicator, recombination
- `observe_outcome` → per-pattern prediction error + lexical reward at word boundaries
- `DevelopmentalStage` → modulates evaluator weights as population complexity grows
- `ExternalSubstrate` → gossip every 20 steps (broadcast + inject)
- `ParallelPatternPool` → configurable workers

---

## Simulation Loop

```
for step in range(total_steps):
    char_id = next(stream)
    agent.perceive_and_learn(char_id)
    record char_id for accuracy tracking

    if step % log_every == 0:
        snapshot = _metrics_snapshot(agent, recent_chars, step)
        print snapshot
        append snapshot to history

    if step % 10_000 == 0 and step > 0:
        PatternSerializer.save(agent.patterns, f"checkpoint_{step}.pkl")

_benchmark_report(history)
PatternSerializer.save(agent.patterns, "final_library.pkl")
```

---

## Metrics Snapshot (every `log_every` steps)

Computed in `_metrics_snapshot(agent, recent_chars, step)`:

### Prediction Accuracy
For each of the last `log_every` chars (stored in a rolling buffer), compute:
```python
relevant = agent.reasoner.get_relevant_patterns(context, top_k=3)
pred = agent.reasoner.compose_predictions(relevant, context)
correct += (argmax(pred) == actual)
```
Report: `accuracy = correct / log_every`

### Word Completion Rate
Extract 10 word prefixes of length 2–4 from the recent buffer (chars between spaces). For each prefix, encode as char IDs, call `agent.reasoner.simulate_future(steps=8)`, decode result, check `dictionary.contains(completed_word)`. Report: `word_completion = hits / 10`. Skip if no dictionary.

### Compression MI
Average `p.compression()` over the top-3 patterns by weight. Reports how much transition structure has been learned.

### Population Stats
- `pop_size`: number of active patterns
- `best_weight`: max pattern weight
- `dev_stage`: `agent.development.level`
- `best_loss`: min `p.running_loss` across patterns

---

## Benchmark Report

After the run, `_benchmark_report(history)` prints a pass/fail table:

| Metric | Target | Final | Pass? |
|--------|--------|-------|-------|
| Prediction accuracy | > 0.50 | — | — |
| Word completion rate | > 0.30 | — | — |
| Compression MI | > 0.20 | — | — |
| Pop survived (not collapsed to 1) | pop_size > 1 | — | — |

"Final" is the last snapshot value. The report also prints the trajectory (metric over time) so degradation is visible.

---

## Files

| Action | Path |
|--------|------|
| Create | `hpm_ai_v4/simulations/full_simulation.py` |

No other files created or modified. All dependencies (`HPMAgent`, `Reasoner`, `SimpleWordList`, `PatternSerializer`, `ParallelPatternPool`) already exist.

---

## Entry Point

```bash
python -m hpm_ai_v4.simulations.full_simulation \
    --corpus data/wikipedia.txt \
    --steps 100000 \
    --log-every 1000 \
    --workers 4 \
    --dict            # flag: enable NLTKWordList (no path needed) \
    --library checkpoints/pretrained.pkl   # optional
```

---

## Success Criteria

- Runs to completion without error on a 100k-char corpus
- All four benchmark metrics logged at every checkpoint
- Prediction accuracy rises above random (1/95 ≈ 1%) — target >50%
- Word completion >30% by step 50k (with dictionary)
- Compression MI >0.2 by step 20k
- Pattern library checkpoint written every 10k steps
- Final library saved at end of run
