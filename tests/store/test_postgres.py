import os
import json
import pytest
import numpy as np

dsn = os.environ.get("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not dsn, reason="TEST_POSTGRES_DSN not set")


@pytest.fixture
def store():
    from hpm.store.postgres import PostgreSQLStore
    s = PostgreSQLStore(dsn)
    yield s
    # Teardown: drop patterns table for isolation
    with s._conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS patterns")
    s._conn.commit()
    s.close()


def _pattern():
    from hpm.patterns.gaussian import GaussianPattern
    return GaussianPattern(np.zeros(2), np.eye(2))


def test_save_and_query(store):
    p = _pattern()
    store.save(p, 0.8, 'agent_a')
    records = store.query('agent_a')
    assert len(records) == 1
    loaded, weight = records[0]
    assert loaded.id == p.id
    assert abs(weight - 0.8) < 1e-9


def test_query_returns_only_agent_patterns(store):
    pa = _pattern()
    pb = _pattern()
    store.save(pa, 0.5, 'agent_a')
    store.save(pb, 0.5, 'agent_b')
    assert len(store.query('agent_a')) == 1
    assert len(store.query('agent_b')) == 1


def test_delete(store):
    p = _pattern()
    store.save(p, 1.0, 'agent_a')
    store.delete(p.id)
    assert store.query('agent_a') == []


def test_update_weight(store):
    p = _pattern()
    store.save(p, 1.0, 'agent_a')
    store.update_weight(p.id, 0.3)
    _, w = store.query('agent_a')[0]
    assert abs(w - 0.3) < 1e-9


def test_update_weight_missing_id_silent_noop(store):
    store.update_weight('nonexistent-id', 0.5)  # should not raise


def test_save_overwrites_existing_id(store):
    p = _pattern()
    store.save(p, 0.5, 'agent_a')
    store.save(p, 0.9, 'agent_a')
    records = store.query('agent_a')
    assert len(records) == 1
    assert abs(records[0][1] - 0.9) < 1e-9


def test_query_all(store):
    pa = _pattern()
    pb = _pattern()
    store.save(pa, 0.6, 'agent_a')
    store.save(pb, 0.4, 'agent_b')
    all_records = store.query_all()
    assert len(all_records) == 2


def test_source_id_round_trips(store):
    from hpm.patterns.gaussian import GaussianPattern
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='origin-uuid')
    store.save(p, 1.0, 'agent_a')
    loaded, _ = store.query('agent_a')[0]
    assert loaded.source_id == 'origin-uuid'
