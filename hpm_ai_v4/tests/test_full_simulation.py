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
    assert len(ids) == 6  # loops back (0, 1, 0, 1, 0, 1)

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
