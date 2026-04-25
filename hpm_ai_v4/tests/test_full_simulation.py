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
