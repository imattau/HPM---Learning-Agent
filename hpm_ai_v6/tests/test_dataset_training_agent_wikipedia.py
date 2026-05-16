from types import SimpleNamespace

from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent


def test_generate_wikipedia_topics_uses_learned_patterns(tmp_path):
    reader = SimpleNamespace(
        agents={
            "contextual": SimpleNamespace(),
            "word": SimpleNamespace(
                patterns=[
                    SimpleNamespace(
                        source=SimpleNamespace(name="word_rabbit"),
                        target=SimpleNamespace(name="word_alice"),
                        weight=0.9,
                    )
                ],
                get_weights=lambda: [0.9],
            ),
            "semantic": SimpleNamespace(
                sent_text_by_name={
                    "sent_1": "Alice saw the Rabbit in Wonderland."
                }
            ),
            "syntactic": SimpleNamespace(),
            "phrase": SimpleNamespace(),
            "reasoning": SimpleNamespace(
                _alias_index={
                    "wonderland": ["sentence:1"],
                }
            ),
        },
        _focus_words={"rabbit"},
        _clean_words=lambda text: [word.strip(".,;:!?()[]{}\"'").lower() for word in text.split() if word],
    )
    agent = DatasetTrainingAgent(reader, corpus_path=str(tmp_path / "corpus.txt"))

    topics = agent.generate_wikipedia_topics(max_topics=5)

    assert "Rabbit" in topics
    assert "Alice" in topics or "Wonderland" in topics


def test_add_from_wikipedia_topics_appends_and_retrains(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("", encoding="utf-8")
    retrain_calls = {"count": 0}

    reader = SimpleNamespace(
        agents={"contextual": SimpleNamespace()},
        retrain_on_new_data=lambda epochs=1: retrain_calls.__setitem__("count", retrain_calls["count"] + 1),
        _clean_words=lambda text: [word.strip(".,;:!?()[]{}\"'").lower() for word in text.split() if word],
    )
    agent = DatasetTrainingAgent(reader, corpus_path=str(corpus_path), min_sentence_len=5)
    monkeypatch.setattr(agent, "_search_wikipedia_titles", lambda topic, limit=3: [f"{topic} page"])
    monkeypatch.setattr(agent, "_fetch_wikipedia_extract", lambda title: "Alice meets Rabbit. More text here.")
    monkeypatch.setattr(agent, "score_text", lambda text: 1.0 if "Alice" in text else 0.0)

    added = agent.add_from_wikipedia_topics(["Alice"], top_k=2, min_score=0.5)

    assert added == 1
    assert retrain_calls["count"] == 1
    assert "Alice meets Rabbit." in corpus_path.read_text(encoding="utf-8")
