from __future__ import annotations

from types import SimpleNamespace

from hpm_ai_v4.io.adapters import WordAdapter
from hpm_ai_v4.simulations import wiki_self_study as wss
from hpm_ai_v4.tools.ingest import TextIngestGate


def test_text_ingest_gate_reuses_canonical_chunk_signature():
    gate = TextIngestGate(adapter=WordAdapter(max_vocab_size=64, lowercase=True))

    assert gate.register_text("Artificial intelligence") is True
    assert gate.register_text("artificial intelligence") is False
    assert gate.register_text("Artificial   intelligence") is False
    assert gate.snapshot()["duplicate_texts"] == 2


def test_text_ingest_gate_round_trips_signature_snapshot(tmp_path):
    gate = TextIngestGate(adapter=WordAdapter(max_vocab_size=64, lowercase=True))
    gate.register_text("Alpha beta gamma.")
    gate.register_text("One two three.")

    path = tmp_path / "ingest.json"
    path.write_text(gate.to_json(), encoding="utf-8")

    loaded = TextIngestGate.load_snapshot_from_path(str(path), adapter=WordAdapter(max_vocab_size=64, lowercase=True))

    assert loaded.register_text("Alpha beta gamma.") is False
    assert loaded.register_text("One two three.") is False
    assert loaded.snapshot()["seen_text_signatures"] == 2


def test_text_ingest_gate_tracks_window_reuse():
    gate = TextIngestGate(adapter=WordAdapter(max_vocab_size=64, lowercase=True))

    assert gate.register_window("Alpha beta gamma.") is True
    assert gate.register_window("Alpha beta gamma.") is False
    assert gate.window_status("Alpha beta delta.") == "near"
    assert gate.snapshot()["duplicate_windows"] == 1


def test_self_study_agent_skips_duplicate_chunks_before_training(monkeypatch, tmp_path):
    class FakePattern:
        def __init__(self, latent_dim=2):
            self.latent_dim = latent_dim

    class FakeReasoner:
        def get_relevant_patterns(self, context, top_k=3):
            return []

    class FakeAgent:
        def __init__(self):
            self.patterns = [FakePattern(), FakePattern()]
            self.reasoner = FakeReasoner()
            self.observed = []
            self._adapter = WordAdapter(max_vocab_size=64, lowercase=True)

        def observe_text(self, text, **kwargs):
            self.observed.append(text)
            return {"target_chars": len(text)}

    class FakeFetcher:
        def __init__(self):
            self.calls = []
            shared = "Alpha beta gamma. Delta epsilon zeta."
            self.pages = {
                "Seed_Topic": wss.WikiPage(title="Seed_Topic", text=shared, links=["Linked_Page"], source_url=""),
                "Linked_Page": wss.WikiPage(
                    title="Linked_Page",
                    text="Alpha beta delta. Delta epsilon eta.",
                    links=[],
                    source_url="",
                ),
            }

        def canonicalize_title(self, title):
            return wss.WikiFetcher.canonicalize_title(title)

        def fetch(self, title, max_links=200):
            self.calls.append(title)
            return self.pages[title]

    saved_paths = []
    monkeypatch.setattr(wss.PatternSerializer, "save", lambda patterns, path: saved_paths.append(path))

    study = wss.SelfStudyAgent(
        seed_topics=["Seed Topic"],
        output_library=str(tmp_path / "wiki_self_study.pkl"),
        steps_per_chunk=5,
        max_pages=2,
        target_patterns=10,
        max_links_per_page=10,
        chunk_char_budget=120,
        surface_mode="word",
        agent=FakeAgent(),
        fetcher=FakeFetcher(),
    )

    result = study.study()

    assert result.pages_read == 2
    assert study.fetcher.calls == ["Seed_Topic", "Linked_Page"]
    assert len(study.agent.observed) >= 1
    assert study._ingest_gate.snapshot()["near_duplicate_windows"] >= 1
    assert saved_paths[-1].endswith("wiki_self_study.pkl")
