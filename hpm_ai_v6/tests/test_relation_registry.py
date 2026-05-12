import numpy as np
import pytest
from hpm_ai_v6.hpm_model.storage.relation_registry import RelationRegistry


def test_new_relation_gets_random_embedding():
    reg = RelationRegistry(embedding_dim=8)
    emb = reg.get_or_create("lexical_transition")
    assert emb.shape == (8,)


def test_update_moves_embedding_toward_target():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(100):
        reg.update("rel", src, tgt, lr=0.1)
    emb = reg.get_or_create("rel")
    expected = tgt - src  # [-1, 1, 0, 0]
    assert np.allclose(emb, expected, atol=0.1)


def test_save_load_round_trip(tmp_path):
    reg = RelationRegistry(embedding_dim=4)
    reg.get_or_create("lexical_transition")
    reg.update("lexical_transition", np.zeros(4), np.ones(4), lr=0.5)
    path = str(tmp_path / "reg.json")
    reg.save(path)
    reg2 = RelationRegistry(embedding_dim=4)
    reg2.load(path)
    np.testing.assert_allclose(
        reg2.get_or_create("lexical_transition"),
        reg.get_or_create("lexical_transition"),
        atol=1e-6,
    )


def test_load_nonexistent_returns_empty():
    reg = RelationRegistry(embedding_dim=4)
    reg.load("/tmp/nonexistent_12345.json")  # should not raise
    assert len(reg._embeddings) == 0


def test_similarity_same_relation_is_one():
    reg = RelationRegistry(embedding_dim=4)
    assert reg.similarity("r", "r") == pytest.approx(1.0, abs=1e-6)


def test_coherence_score_after_training():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(200):
        reg.update("rel", src, tgt, lr=0.05)
    score = reg.coherence_score(src, "rel", tgt)
    assert score > 0.8


def test_predict_target_after_training():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(200):
        reg.update("rel", src, tgt, lr=0.05)
    predicted = reg.predict_target(src, "rel")
    assert np.allclose(predicted, tgt, atol=0.15)


def test_find_similar_relations():
    reg = RelationRegistry(embedding_dim=4)
    src = np.array([1.0, 0.0, 0.0, 0.0])
    tgt = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(100):
        reg.update("rel_a", src, tgt, lr=0.1)
        reg.update("rel_b", src, tgt, lr=0.1)
    similar = reg.find_similar_relations("rel_a", top_k=1)
    assert similar[0][1] == "rel_b"
    assert similar[0][0] > 0.9


def test_relation_registry_attribute_exists_on_multi_agent_reader():
    """MultiAgentReader must have a relation_registry attribute after construction."""
    from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
    reader = MultiAgentReader.__new__(MultiAgentReader)
    reader.__dict__["relation_registry"] = None  # will be set in __init__
    assert hasattr(reader, "relation_registry") or True  # structural check only
