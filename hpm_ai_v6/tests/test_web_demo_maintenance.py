from types import SimpleNamespace
import os

from hpm_ai_v6.web import web_demo
from hpm_ai_v6.web.web_demo import (
    _update_gutenberg_cycle_state,
    _update_wikipedia_cycle_state,
    gutenberg_cycle_state,
    wikipedia_cycle_state,
    wikipedia_topics_state,
)


def test_maintenance_done_updates_gutenberg_cycle_status_message():
    original = dict(gutenberg_cycle_state)
    try:
        gutenberg_cycle_state.update(
            {
                "active": True,
                "message": "Running.",
                "book_id": 11,
                "chapter": None,
                "phase": "book_done",
                "books_processed": 2,
                "chapter_added": 0,
            }
        )

        _update_gutenberg_cycle_state(
            {
                "type": "maintenance_done",
                "book_id": 11,
                "report": {
                    "_summary": {
                        "loaded": 17,
                        "improved_agents": ["word", "contextual"],
                    }
                },
            }
        )

        assert "book 11" in gutenberg_cycle_state["message"].lower()
        assert "loaded 17 patterns" in gutenberg_cycle_state["message"].lower()
        assert "word" in gutenberg_cycle_state["message"].lower()
    finally:
        gutenberg_cycle_state.clear()
        gutenberg_cycle_state.update(original)


def test_api_reason_returns_trace_payload():
    original_reader = web_demo.reader
    try:
        web_demo.reader = SimpleNamespace(
            reasoning_agent=SimpleNamespace(
                reason_with_trace=lambda question: {
                    "question": question,
                    "intent": "path",
                    "mode": "connection",
                    "method": "auto",
                    "answer": "stub answer",
                    "explanation_method": "abductive_explain",
                }
            ),
            reason=lambda question: "fallback answer",
        )

        client = web_demo.app.test_client()
        response = client.post("/api/reason", json={"question": "How does alice connect?"})
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["reasoning"] == "stub answer"
        assert payload["trace"]["intent"] == "path"
        assert payload["trace"]["explanation_method"] == "abductive_explain"
    finally:
        web_demo.reader = original_reader


def test_api_reasoning_index_status_returns_state():
    original_reader = web_demo.reader
    try:
        web_demo.reader = SimpleNamespace(
            reasoning_agent=SimpleNamespace(
                index_status=lambda: {
                    "state": "indexing",
                    "dirty": True,
                    "last_refresh_started_at": 123.0,
                    "last_refresh_finished_at": None,
                }
            )
        )

        client = web_demo.app.test_client()
        response = client.get("/api/reasoning_index_status")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["state"] == "indexing"
        assert payload["dirty"] is True
    finally:
        web_demo.reader = original_reader


def test_api_wikipedia_topics_returns_generated_topics(monkeypatch):
    original_agent = web_demo.dataset_agent
    original_state = dict(wikipedia_topics_state)
    try:
        web_demo.dataset_agent = SimpleNamespace(
            generate_wikipedia_topics=lambda max_topics=8: ["Alice", "Rabbit", "Wonderland"]
        )

        client = web_demo.app.test_client()
        response = client.get("/api/wikipedia_topics")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["topics"] == ["Alice", "Rabbit", "Wonderland"]
        assert wikipedia_topics_state["topics_text"] == "Alice\nRabbit\nWonderland"
    finally:
        web_demo.dataset_agent = original_agent
        wikipedia_topics_state.clear()
        wikipedia_topics_state.update(original_state)


def test_index_post_generates_wikipedia_topics(monkeypatch):
    original_reader = web_demo.reader
    original_dataset_agent = web_demo.dataset_agent
    original_state = dict(wikipedia_topics_state)
    try:
        web_demo.reader = SimpleNamespace(
            generate=lambda seed, max_length=50, temperature=0.0: "stub",
            reason=lambda question: "stub",
            reasoning_agent=SimpleNamespace(reason_with_trace=lambda question: {"answer": "stub"}),
        )
        web_demo.dataset_agent = SimpleNamespace(
            generate_wikipedia_topics=lambda max_topics=8: ["Alice", "Rabbit"]
        )

        client = web_demo.app.test_client()
        response = client.post(
            "/",
            data={
                "action": "generate_wikipedia_topics",
                "wikipedia_topics": "",
                "wikipedia_top_k": "12",
                "wikipedia_min_score": "0.05",
            },
        )
        html = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "Alice" in html
        assert "Rabbit" in html
        assert wikipedia_topics_state["topics_text"] == "Alice\nRabbit"
    finally:
        web_demo.reader = original_reader
        web_demo.dataset_agent = original_dataset_agent
        wikipedia_topics_state.clear()
        wikipedia_topics_state.update(original_state)


def test_index_renders_wikipedia_button_wiring():
    original_reader = web_demo.reader
    try:
        web_demo.reader = SimpleNamespace(
            generate=lambda seed, max_length=50, temperature=0.0: "stub",
            reason=lambda question: "stub",
            reasoning_agent=SimpleNamespace(reason_with_trace=lambda question: {"answer": "stub"}),
        )

        client = web_demo.app.test_client()
        response = client.get("/")
        html = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "return showSub('train', 'links', this, event)" in html
        assert "return showSub('train', 'gutenberg', this, event)" in html
        assert "return showSub('train', 'wikipedia', this, event)" in html
        assert "submitWikipediaAction('generate_wikipedia_topics'" in html
        assert "submitWikipediaAction('start_wikipedia_cycle'" in html
        assert "submitWikipediaAction('stop_wikipedia_cycle'" in html
        assert "this.form.action.value='generate_wikipedia_topics'" not in html
    finally:
        web_demo.reader = original_reader


def test_api_wikipedia_cycle_status_returns_state():
    original_state = dict(wikipedia_cycle_state)
    try:
        wikipedia_cycle_state.update(
            {
                "active": True,
                "message": "Running Wikipedia cycle.",
                "topic": "Alice",
                "page_title": "Alice's Adventures in Wonderland",
                "phase": "page_start",
                "topics_processed": 3,
                "pages_added": 12,
            }
        )

        client = web_demo.app.test_client()
        response = client.get("/api/wikipedia_cycle_status")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["active"] is True
        assert payload["topic"] == "Alice"
        assert payload["pages_added"] == 12
    finally:
        wikipedia_cycle_state.clear()
        wikipedia_cycle_state.update(original_state)


def test_index_renders_gutenberg_polling_gate():
    original_state = dict(gutenberg_cycle_state)
    original_reader = web_demo.reader
    try:
        web_demo.reader = SimpleNamespace(
            generate=lambda seed, max_length=50, temperature=0.0: "stub",
            reason=lambda question: "stub",
            reasoning_agent=SimpleNamespace(reason_with_trace=lambda question: {"answer": "stub"}),
        )
        gutenberg_cycle_state.update(
            {
                "active": False,
                "message": "Idle.",
                "book_id": None,
                "chapter": None,
                "phase": "idle",
                "books_processed": 0,
                "chapter_added": 0,
            }
        )

        client = web_demo.app.test_client()
        response = client.get("/")
        html = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "if (false) {" in html
        assert "setInterval(refreshCycleStatus, 4000)" not in html
        assert "reasoning-index-badge" in html
        assert "Wikipedia" in html
        assert "wikipedia_topics" in html
        assert "/api/wikipedia_topics" in html
        assert "/api/wikipedia_cycle_status" in html
    finally:
        web_demo.reader = original_reader
        gutenberg_cycle_state.clear()
        gutenberg_cycle_state.update(original_state)


def test_update_wikipedia_cycle_state_tracks_progress():
    original_state = dict(wikipedia_cycle_state)
    try:
        wikipedia_cycle_state.update(
            {
                "active": True,
                "message": "Running.",
                "topic": None,
                "page_title": None,
                "phase": "idle",
                "topics_processed": 0,
                "pages_added": 0,
            }
        )

        _update_wikipedia_cycle_state(
            {
                "type": "page_done",
                "topic": "Alice",
                "page_title": "Alice's Adventures in Wonderland",
                "added": 4,
            }
        )

        assert wikipedia_cycle_state["topic"] == "Alice"
        assert wikipedia_cycle_state["page_title"] == "Alice's Adventures in Wonderland"
        assert wikipedia_cycle_state["pages_added"] == 4
    finally:
        wikipedia_cycle_state.clear()
        wikipedia_cycle_state.update(original_state)


def test_build_reader_skips_startup_retrain_by_default(monkeypatch):
    calls = {"train": 0}

    class DummyReader:
        def __init__(self, corpus_path, warm_start=True):
            self.corpus_path = corpus_path
            self.warm_start = warm_start

        def train(self, *args, **kwargs):
            calls["train"] += 1

    monkeypatch.setattr(web_demo, "MultiAgentReader", DummyReader)
    monkeypatch.setattr(web_demo, "_corpus_path", lambda: "/tmp/corpus.txt")
    monkeypatch.delenv("HPM_WEB_DEMO_RETRAIN", raising=False)

    reader = web_demo._build_reader()

    assert reader.corpus_path == "/tmp/corpus.txt"
    assert reader.warm_start is True
    assert calls["train"] == 0


def test_build_reader_can_opt_in_to_startup_retrain(monkeypatch):
    calls = {"train": 0}

    class DummyReader:
        def __init__(self, corpus_path, warm_start=True):
            self.corpus_path = corpus_path
            self.warm_start = warm_start

        def train(self, *args, **kwargs):
            calls["train"] += 1

    monkeypatch.setattr(web_demo, "MultiAgentReader", DummyReader)
    monkeypatch.setattr(web_demo, "_corpus_path", lambda: "/tmp/corpus.txt")
    monkeypatch.setenv("HPM_WEB_DEMO_RETRAIN", "1")

    reader = web_demo._build_reader()

    assert reader.corpus_path == "/tmp/corpus.txt"
    assert calls["train"] == 1


def test_build_reader_hydrates_generation_vocab():
    reader = web_demo._build_reader()
    generated = reader.generate("Alice was", max_length=8, temperature=0.0)

    assert len(getattr(reader.word_agent, "word_cells", {})) > 0
    assert generated.lower() != "alice was"
    assert generated.lower().startswith("alice was ")
