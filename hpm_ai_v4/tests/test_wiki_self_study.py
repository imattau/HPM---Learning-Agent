from __future__ import annotations

from types import SimpleNamespace

from hpm_ai_v4.simulations import wiki_self_study as wss


def test_wiki_fetcher_canonicalizes_and_filters_mainspace_links(monkeypatch):
    fetcher = wss.WikiFetcher(api_url="https://example.org/w/api.php")

    payloads = iter(
        [
            {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Artificial intelligence",
                            "extract": "AI is a field.",
                            "links": [
                                {"title": "Machine learning"},
                                {"title": "File:Ignored"},
                                {"title": "Category:Ignored"},
                            ],
                        }
                    }
                },
                "continue": {"plcontinue": "next"},
            },
            {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Artificial intelligence",
                            "extract": "AI is a field.",
                            "links": [
                                {"title": "Machine learning"},
                                {"title": "Knowledge representation"},
                            ],
                        }
                    }
                }
            },
        ]
    )

    monkeypatch.setattr(fetcher, "_request_json", lambda params: next(payloads))

    page = fetcher.fetch("Artificial intelligence", max_links=10)

    assert page.title == "Artificial_intelligence"
    assert page.text == "AI is a field."
    assert page.links == ["Machine learning", "Knowledge representation"]


def test_curiosity_scheduler_scores_unknown_topics_from_epistemic_gap(monkeypatch):
    class FakeReasoner:
        def get_relevant_patterns(self, context, top_k=3):
            return [object()]

    agent = SimpleNamespace(
        reasoner=FakeReasoner(),
        _surface_ids_from_text=lambda text: [1, 2, 3],
    )
    scheduler = wss.CuriosityScheduler(agent)
    monkeypatch.setattr(wss, "epistemic_score", lambda pattern: 0.25)

    scheduler.page_bonus["Test_Page"] = 0.1
    score = scheduler.score_link("Test Page")

    assert 0.8 < score < 1.0


def test_self_study_agent_reads_pages_and_follows_links(monkeypatch, tmp_path):
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

        def _surface_ids_from_text(self, text):
            return [max(0, min(94, ord(ch) - 32)) for ch in text if 32 <= ord(ch) <= 126]

        def observe_text(self, text, **kwargs):
            self.observed.append(text)
            return {"target_chars": len(text)}

    class FakeFetcher:
        def __init__(self):
            self.calls = []
            self.pages = {
                "Seed_Topic": wss.WikiPage(
                    title="Seed_Topic",
                    text="Seed topic page. It links onward.",
                    links=["Linked_Page"],
                    source_url="",
                ),
                "Linked_Page": wss.WikiPage(
                    title="Linked_Page",
                    text="Linked page text.",
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
        steps_per_chunk=2,
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
    assert result.patterns_saved == 2
    assert study.fetcher.calls == ["Seed_Topic", "Linked_Page"]
    assert len(study.agent.observed) >= 2
    assert saved_paths[-1].endswith("wiki_self_study.pkl")
