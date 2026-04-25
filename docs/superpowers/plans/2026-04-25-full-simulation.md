# Full HPM AI Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `hpm_ai_v4/simulations/full_simulation.py` — a single simulation that exercises every HPM v4 feature on a Wikipedia character stream, with quantitative benchmarking.

**Architecture:** `HPMAgent(obs_dim=95)` is the core unit; all features (Reasoner, observe_outcome, lexical reward, DevelopmentalStage, ExternalSubstrate, ParallelPatternPool, replicator dynamics, recombination) activate automatically through `perceive_and_learn`. A `WikipediaStream` feeds raw char IDs (0–94). Metrics are snapshotted every `log_every` steps; pattern library checkpoints saved every 10k steps.

**Tech Stack:** Python 3.10+, numpy, hpm_ai_v4 package (HPMAgent, NLTKWordList, PatternSerializer, ParallelPatternPool), argparse, nltk (words corpus)

---

## File Structure

- **Create:** `hpm_ai_v4/simulations/full_simulation.py` — main simulation (~220 lines)
- **Create:** `hpm_ai_v4/tests/test_full_simulation.py` — unit tests for WikipediaStream, _metrics_snapshot, _benchmark_report

---

### Task 1: WikipediaStream + argparse skeleton

**Files:**
- Create: `hpm_ai_v4/simulations/full_simulation.py`
- Create: `hpm_ai_v4/tests/test_full_simulation.py`

- [ ] **Step 1: Write failing tests for WikipediaStream**

```python
# hpm_ai_v4/tests/test_full_simulation.py
import tempfile, os, pytest
from hpm_ai_v4.simulations.full_simulation import WikipediaStream

def _write_corpus(text):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    f.write(text)
    f.close()
    return f.name

def test_wikipedia_stream_yields_ints():
    path = _write_corpus("hello world\n")
    stream = WikipediaStream(path)
    ids = [next(iter(stream)) for _ in range(5)]
    os.unlink(path)
    assert all(isinstance(i, int) for i in ids)

def test_wikipedia_stream_range():
    path = _write_corpus("az AZ 09 !~\n")
    stream = WikipediaStream(path)
    ids = list(next(iter(stream)) for _ in range(20) if True)
    # collect all from one pass
    ids = []
    for i, v in enumerate(stream):
        ids.append(v)
        if i >= 30: break
    os.unlink(path)
    assert all(0 <= v <= 94 for v in ids)

def test_wikipedia_stream_loops():
    path = _write_corpus("ab")
    stream = WikipediaStream(path)
    ids = []
    for i, v in enumerate(stream):
        ids.append(v)
        if i >= 5: break
    os.unlink(path)
    assert len(ids) == 6  # loops back
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — full_simulation doesn't exist yet.

- [ ] **Step 3: Implement WikipediaStream + argparse skeleton**

```python
# hpm_ai_v4/simulations/full_simulation.py
"""Full HPM AI simulation: all v4 features on a Wikipedia character stream."""
import argparse
import os
from typing import Iterator, List, Optional, Dict, Any

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.serializer import PatternSerializer


class WikipediaStream:
    """Yields char IDs (0–94) from a text file, looping on exhaustion."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self) -> Iterator[int]:
        while True:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for ch in f.read():
                    if ch == '\n':
                        yield 94
                    else:
                        code = ord(ch)
                        if 32 <= code <= 126:
                            yield code - 32


def _parse_args():
    p = argparse.ArgumentParser(description="Full HPM AI simulation")
    p.add_argument('--corpus', required=True, help='Path to plain-text corpus file')
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--log-every', type=int, default=1_000)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dict', action='store_true', help='Enable NLTKWordList dictionary')
    p.add_argument('--library', default=None, help='Path to pre-built pattern library (.pkl)')
    p.add_argument('--checkpoint-dir', default='.', help='Directory for checkpoint files')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print(f"corpus={args.corpus} steps={args.steps} workers={args.workers}")
```

- [ ] **Step 4: Run tests**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v
```
Expected: all 3 WikipediaStream tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v4/simulations/full_simulation.py hpm_ai_v4/tests/test_full_simulation.py
git commit -m "feat: WikipediaStream + argparse skeleton for full simulation"
```

---

### Task 2: _metrics_snapshot

**Files:**
- Modify: `hpm_ai_v4/simulations/full_simulation.py`
- Modify: `hpm_ai_v4/tests/test_full_simulation.py`

- [ ] **Step 1: Write failing tests**

```python
# append to hpm_ai_v4/tests/test_full_simulation.py
from hpm_ai_v4.simulations.full_simulation import _metrics_snapshot
from hpm_ai_v4.agents.agent import HPMAgent

def _make_agent():
    agent = HPMAgent(obs_dim=95, num_initial_patterns=3, num_workers=1)
    # Feed some observations so buffer is not empty
    for i in range(50):
        agent.perceive_and_learn(i % 95)
    return agent

def test_metrics_snapshot_returns_dict():
    agent = _make_agent()
    recent = list(range(50))
    snap = _metrics_snapshot(agent, recent, step=50)
    assert 'accuracy' in snap
    assert 'compression_mi' in snap
    assert 'pop_size' in snap
    assert 'best_weight' in snap
    assert 'dev_stage' in snap
    assert 'best_loss' in snap

def test_metrics_snapshot_accuracy_range():
    agent = _make_agent()
    recent = list(range(100))
    snap = _metrics_snapshot(agent, recent, step=100)
    assert 0.0 <= snap['accuracy'] <= 1.0

def test_metrics_snapshot_no_dict():
    agent = _make_agent()
    recent = list(range(50))
    snap = _metrics_snapshot(agent, recent, step=50)
    assert snap.get('word_completion') is None  # no dictionary attached
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py::test_metrics_snapshot_returns_dict -v
```
Expected: `ImportError` — `_metrics_snapshot` not defined yet.

- [ ] **Step 3: Implement _metrics_snapshot**

Add after `WikipediaStream` class in `full_simulation.py`:

```python
def _metrics_snapshot(agent: HPMAgent, recent_chars: List[int], step: int) -> Dict[str, Any]:
    """Compute metrics over the recent character buffer."""
    snap: Dict[str, Any] = {'step': step}

    # --- Prediction accuracy ---
    correct = 0
    total = max(1, len(recent_chars) - 1)
    for i in range(len(recent_chars) - 1):
        context = recent_chars[max(0, i - 20):i]
        actual = recent_chars[i + 1]
        relevant = agent.reasoner.get_relevant_patterns(context, top_k=3)
        if relevant:
            pred_dist = agent.reasoner.compose_predictions(relevant, context)
            if int(np.argmax(pred_dist)) == actual:
                correct += 1
    snap['accuracy'] = correct / total

    # --- Compression MI ---
    top3 = sorted(agent.patterns, key=lambda p: p.weight, reverse=True)[:3]
    snap['compression_mi'] = float(np.mean([p.compression() for p in top3])) if top3 else 0.0

    # --- Population stats ---
    snap['pop_size'] = len(agent.patterns)
    snap['best_weight'] = float(max(p.weight for p in agent.patterns)) if agent.patterns else 0.0
    snap['dev_stage'] = agent.development.level
    snap['best_loss'] = float(min(p.running_loss for p in agent.patterns)) if agent.patterns else 0.0

    # --- Word completion (only if dictionary attached) ---
    if agent.dictionary:
        hits = 0
        # Extract up to 10 word prefixes of length 2-4 from recent chars
        prefixes = _extract_prefixes(recent_chars, n=10)
        for prefix_ids in prefixes:
            future = agent.reasoner.simulate_future(steps=8, top_k=3)
            word_ids = prefix_ids + future
            # Decode: char_id + 32 = ASCII
            word = ''.join(chr(v + 32) for v in word_ids if 0 <= v <= 62).strip()
            word = word.split()[0] if ' ' in word else word
            if word and agent.dictionary.contains(word.lower()):
                hits += 1
        snap['word_completion'] = hits / max(1, len(prefixes))
    else:
        snap['word_completion'] = None

    return snap


def _extract_prefixes(chars: List[int], n: int = 10) -> List[List[int]]:
    """Extract up to n word prefixes (length 2-4) from char ID sequence."""
    # Space = char_id 0 (ASCII 32 - 32 = 0)
    space_id = 0
    prefixes = []
    i = 0
    while i < len(chars) and len(prefixes) < n:
        if chars[i] == space_id and i + 1 < len(chars):
            # Start of a word — take 2-4 chars
            end = min(i + 5, len(chars))
            word_chars = [c for c in chars[i+1:end] if c != space_id]
            if 2 <= len(word_chars) <= 4:
                prefixes.append(word_chars)
        i += 1
    return prefixes
```

- [ ] **Step 4: Run tests**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v -k "metrics"
```
Expected: all 3 metrics tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v4/simulations/full_simulation.py hpm_ai_v4/tests/test_full_simulation.py
git commit -m "feat: _metrics_snapshot with accuracy, compression MI, word completion"
```

---

### Task 3: _benchmark_report

**Files:**
- Modify: `hpm_ai_v4/simulations/full_simulation.py`
- Modify: `hpm_ai_v4/tests/test_full_simulation.py`

- [ ] **Step 1: Write failing tests**

```python
# append to hpm_ai_v4/tests/test_full_simulation.py
from hpm_ai_v4.simulations.full_simulation import _benchmark_report
import io, sys

def test_benchmark_report_outputs_table(capsys):
    history = [
        {'step': 0,     'accuracy': 0.01, 'compression_mi': 0.0,  'pop_size': 6, 'word_completion': None},
        {'step': 50000, 'accuracy': 0.45, 'compression_mi': 0.15, 'pop_size': 4, 'word_completion': 0.25},
        {'step': 99000, 'accuracy': 0.55, 'compression_mi': 0.22, 'pop_size': 5, 'word_completion': 0.35},
    ]
    _benchmark_report(history)
    out = capsys.readouterr().out
    assert 'accuracy' in out.lower()
    assert 'PASS' in out or 'FAIL' in out

def test_benchmark_report_pass_fail():
    history = [
        {'step': 99000, 'accuracy': 0.60, 'compression_mi': 0.25, 'pop_size': 5, 'word_completion': 0.40},
    ]
    import io, sys
    captured = io.StringIO()
    sys.stdout = captured
    _benchmark_report(history)
    sys.stdout = sys.__stdout__
    out = captured.getvalue()
    assert 'PASS' in out

def test_benchmark_report_fail():
    history = [
        {'step': 99000, 'accuracy': 0.02, 'compression_mi': 0.01, 'pop_size': 1, 'word_completion': 0.05},
    ]
    import io, sys
    captured = io.StringIO()
    sys.stdout = captured
    _benchmark_report(history)
    sys.stdout = sys.__stdout__
    out = captured.getvalue()
    assert 'FAIL' in out
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py::test_benchmark_report_outputs_table -v
```
Expected: `ImportError` — `_benchmark_report` not defined.

- [ ] **Step 3: Implement _benchmark_report**

```python
def _benchmark_report(history: List[Dict[str, Any]]) -> None:
    """Print pass/fail benchmark summary from metric history."""
    if not history:
        print("No metrics recorded.")
        return

    final = history[-1]
    targets = [
        ('Prediction accuracy',  'accuracy',        0.50),
        ('Compression MI',       'compression_mi',  0.20),
        ('Population survived',  'pop_size',        2),     # > 1 pattern
    ]
    if final.get('word_completion') is not None:
        targets.append(('Word completion', 'word_completion', 0.30))

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"{'Metric':<28} {'Target':>8} {'Final':>8} {'':>6}")
    print("-" * 60)
    all_pass = True
    for label, key, target in targets:
        val = final.get(key, 0.0)
        passed = val > target
        if not passed:
            all_pass = False
        mark = 'PASS' if passed else 'FAIL'
        print(f"{label:<28} {target:>8.2f} {val:>8.2f} {mark:>6}")

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print()

    # Trajectory
    print("Accuracy trajectory:")
    for snap in history[::max(1, len(history)//10)]:
        bar = '#' * int(snap.get('accuracy', 0) * 40)
        print(f"  step {snap['step']:6d}: {snap.get('accuracy', 0):.3f} {bar}")
    print()
```

- [ ] **Step 4: Run tests**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v -k "benchmark"
```
Expected: all 3 benchmark tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v4/simulations/full_simulation.py hpm_ai_v4/tests/test_full_simulation.py
git commit -m "feat: _benchmark_report with pass/fail table and accuracy trajectory"
```

---

### Task 4: run_simulation — main loop

**Files:**
- Modify: `hpm_ai_v4/simulations/full_simulation.py`
- Modify: `hpm_ai_v4/tests/test_full_simulation.py`

- [ ] **Step 1: Write failing tests**

```python
# append to hpm_ai_v4/tests/test_full_simulation.py
from hpm_ai_v4.simulations.full_simulation import run_simulation

def test_run_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)
    history = run_simulation(
        corpus_path=str(corpus),
        total_steps=200,
        log_every=100,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    assert len(history) >= 2
    assert 'accuracy' in history[-1]
    assert 'compression_mi' in history[-1]

def test_run_simulation_saves_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world " * 100)
    run_simulation(
        corpus_path=str(corpus),
        total_steps=500,
        log_every=200,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    # Should save final_library.pkl
    assert (tmp_path / "final_library.pkl").exists()
```

- [ ] **Step 2: Run to verify fail**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py::test_run_simulation_short -v
```
Expected: `ImportError` — `run_simulation` not defined.

- [ ] **Step 3: Implement run_simulation**

```python
def run_simulation(
    corpus_path: str,
    total_steps: int = 100_000,
    log_every: int = 1_000,
    num_workers: int = 1,
    use_dict: bool = False,
    library_path: Optional[str] = None,
    checkpoint_dir: str = '.',
) -> List[Dict[str, Any]]:
    """Run the full HPM AI simulation. Returns metric history."""

    dictionary = NLTKWordList() if use_dict else None

    agent = HPMAgent(
        obs_dim=95,
        num_initial_patterns=6,
        num_workers=num_workers,
        dictionary=dictionary,
    )

    if library_path and os.path.exists(library_path):
        n = agent.load_library(library_path, reset_weights=True)
        print(f"Loaded {n} patterns from {library_path}")

    stream = WikipediaStream(corpus_path)
    stream_iter = iter(stream)

    history: List[Dict[str, Any]] = []
    recent_chars: List[int] = []
    accuracy_buffer: List[int] = []  # actual chars for accuracy computation

    print(f"Starting simulation: steps={total_steps} log_every={log_every} "
          f"workers={num_workers} dict={use_dict}")

    for step in range(total_steps):
        char_id = next(stream_iter)
        accuracy_buffer.append(char_id)
        if len(accuracy_buffer) > log_every + 21:
            accuracy_buffer = accuracy_buffer[-(log_every + 21):]

        agent.perceive_and_learn(char_id)

        recent_chars.append(char_id)
        if len(recent_chars) > log_every:
            recent_chars = recent_chars[-log_every:]

        if step % log_every == 0 and step > 0:
            snap = _metrics_snapshot(agent, list(accuracy_buffer), step)
            history.append(snap)
            wc = f"{snap['word_completion']:.3f}" if snap['word_completion'] is not None else 'n/a'
            print(
                f"[step {step:6d}] acc={snap['accuracy']:.3f} "
                f"mi={snap['compression_mi']:.3f} "
                f"wc={wc} "
                f"pop={snap['pop_size']} "
                f"stage={snap['dev_stage']} "
                f"loss={snap['best_loss']:.3f}"
            )

        if step % 10_000 == 0 and step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
            PatternSerializer.save(agent.patterns, ckpt_path)
            print(f"  [checkpoint saved: {ckpt_path}]")

    final_path = os.path.join(checkpoint_dir, "final_library.pkl")
    PatternSerializer.save(agent.patterns, final_path)
    print(f"Final library saved: {final_path}")

    _benchmark_report(history)
    return history
```

- [ ] **Step 4: Wire up __main__**

Replace the existing `if __name__ == '__main__':` block:

```python
if __name__ == '__main__':
    args = _parse_args()
    run_simulation(
        corpus_path=args.corpus,
        total_steps=args.steps,
        log_every=args.log_every,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        checkpoint_dir=args.checkpoint_dir,
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest hpm_ai_v4/tests/test_full_simulation.py -v
```
Expected: all tests PASS (including `test_run_simulation_short` and `test_run_simulation_saves_checkpoint`).

- [ ] **Step 6: Smoke test end-to-end**

```bash
# Requires a corpus file. Use the build_library corpus if available, or generate one:
python -c "print('the quick brown fox ' * 500)" > /tmp/test_corpus.txt
python -m hpm_ai_v4.simulations.full_simulation \
    --corpus /tmp/test_corpus.txt \
    --steps 2000 \
    --log-every 500 \
    --workers 1
```
Expected: 4 metric snapshots printed, benchmark report at end, `final_library.pkl` written.

- [ ] **Step 7: Commit**

```bash
git add hpm_ai_v4/simulations/full_simulation.py hpm_ai_v4/tests/test_full_simulation.py
git commit -m "feat: run_simulation main loop with checkpointing and benchmark report"
```

---

## Self-Review

**Spec coverage check:**
- ✅ WikipediaStream (char IDs 0–94, loops) — Task 1
- ✅ HPMAgent(obs_dim=95, num_workers, dictionary) — Task 4
- ✅ library_path load via agent.load_library — Task 4
- ✅ perceive_and_learn every step (triggers observe_outcome, EM, replicator, DevelopmentalStage, gossip) — Task 4
- ✅ Prediction accuracy metric — Task 2
- ✅ Word completion metric — Task 2
- ✅ Compression MI metric — Task 2
- ✅ Population stats (pop_size, best_weight, dev_stage, best_loss) — Task 2
- ✅ Checkpoint every 10k steps — Task 4
- ✅ Final library save — Task 4
- ✅ Benchmark report pass/fail table — Task 3
- ✅ Accuracy trajectory — Task 3
- ✅ argparse CLI entry point — Task 1 + Task 4

**No placeholders found.**

**Type consistency:** `_metrics_snapshot` returns `Dict[str, Any]` throughout. `run_simulation` returns `List[Dict[str, Any]]`. `_benchmark_report` consumes `List[Dict[str, Any]]`. All consistent.
