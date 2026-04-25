# hpm_ai_v4/tests/test_full_simulation.py
import tempfile, os, pytest
import numpy as np
from hpm_ai_v4.simulations.full_simulation import WikipediaStream, _metrics_snapshot, _benchmark_report, run_simulation
from hpm_ai_v4.simulations.layered_agent import LayeredAgent

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

def _make_agent():
    agent = LayeredAgent(num_workers=1)
    # Feed some observations so buffer is not empty
    for i in range(50):
        agent.perceive(i % 95)
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
    assert 'l2_accuracy' in snap

def test_metrics_snapshot_accuracy_range():
    agent = _make_agent()
    recent = list(range(100))
    snap = _metrics_snapshot(agent, recent, step=100)
    assert 0.0 <= snap['accuracy'] <= 1.0
    assert 0.0 <= snap['l2_accuracy'] <= 1.0

def test_metrics_snapshot_no_dict():
    agent = _make_agent()
    recent = list(range(50))
    snap = _metrics_snapshot(agent, recent, step=50)
    assert snap.get('word_completion') is None  # no dictionary attached

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

def test_run_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)
    history = run_simulation(
        corpus_path=str(corpus),
        total_steps=201,
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
    # Should save final_library.l1.pkl + .l2.pkl
    assert (tmp_path / "final_library.l1.pkl").exists()
    assert (tmp_path / "final_library.l2.pkl").exists()

def test_run_simulation_has_l2_accuracy(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox " * 30)
    history = run_simulation(
        corpus_path=str(corpus),
        total_steps=301,
        log_every=100,
        num_workers=1,
        use_dict=False,
        library_path=None,
        checkpoint_dir=str(tmp_path),
    )
    assert 'l2_accuracy' in history[-1]
