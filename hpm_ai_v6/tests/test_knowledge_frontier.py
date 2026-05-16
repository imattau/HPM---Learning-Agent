"""Unit tests for KnowledgeFrontier class."""
import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch


def _make_reader(edge_counts: dict):
    """Build a mock reader whose agents return controlled edge counts."""
    reader = MagicMock()
    agent = MagicMock()
    # iter_index_payloads yields dicts with "name" key
    payloads = [{"name": term} for term, count in edge_counts.items() for _ in range(count)]
    agent.pattern_pager.iter_index_payloads.return_value = iter(payloads)
    reader.agents = {"agent0": agent}
    return reader


class TestKnowledgeFrontierInit:
    def test_default_state(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        assert kf.known_seeds == set()
        assert kf.frontier == []
        assert kf.exhausted == set()
        assert kf.hop_depth == 1

    def test_load_missing_path_returns_default(self, tmp_path):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier.load(str(tmp_path / "nonexistent.json"))
        assert kf.hop_depth == 1
        assert kf.known_seeds == set()

    def test_save_and_load_roundtrip(self, tmp_path):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        path = str(tmp_path / "kf.json")
        kf = KnowledgeFrontier()
        kf.known_seeds = {"dog", "cat"}
        kf.frontier = ["animal", "mammal"]
        kf.exhausted = {"pet"}
        kf.hop_depth = 3
        kf.save(path)
        kf2 = KnowledgeFrontier.load(path)
        assert kf2.known_seeds == {"dog", "cat"}
        assert sorted(kf2.frontier) == ["animal", "mammal"]
        assert kf2.exhausted == {"pet"}
        assert kf2.hop_depth == 3


class TestIncrementHop:
    def test_increments(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.increment_hop()
        assert kf.hop_depth == 2

    def test_capped_at_5(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.hop_depth = 5
        kf.increment_hop()
        assert kf.hop_depth == 5


class TestEdgeDensity:
    def test_counts_matching_payloads(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = _make_reader({"dog": 3, "cat": 1, "animal": 0})
        kf = KnowledgeFrontier()
        count = kf._edge_density("dog", reader)
        assert count == 3

    def test_zero_for_unknown_term(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = _make_reader({"dog": 2})
        kf = KnowledgeFrontier()
        assert kf._edge_density("zebra", reader) == 0

    def test_skips_agents_without_pager(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        reader = MagicMock()
        agent = MagicMock()
        agent.pattern_pager = None
        reader.agents = {"a": agent}
        kf = KnowledgeFrontier()
        assert kf._edge_density("anything", reader) == 0


class TestWordnetCandidates:
    def test_returns_set_of_strings(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        candidates = kf._wordnet_candidates("dog", hop_depth=1)
        assert isinstance(candidates, set)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, str)

    def test_hop2_returns_more_than_hop1(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        hop1 = kf._wordnet_candidates("dog", hop_depth=1)
        hop2 = kf._wordnet_candidates("dog", hop_depth=2)
        assert len(hop2) >= len(hop1)

    def test_filters_pos_prefixes(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        candidates = kf._wordnet_candidates("dog", hop_depth=2)
        for c in candidates:
            assert not c.startswith("pos_")
            assert not c.startswith("word_")
            assert len(c) > 1


class TestAddLearnedSeeds:
    def test_new_seeds_added_to_known(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value={"mammal", "canine", "pet"}):
            kf.add_learned_seeds(["dog"], reader)
        assert "dog" in kf.known_seeds

    def test_already_known_seeds_not_re_expanded(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.known_seeds = {"dog"}
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value=set()) as mock_wn:
            kf.add_learned_seeds(["dog"], reader)
        mock_wn.assert_not_called()

    def test_candidates_added_to_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        reader = _make_reader({})
        candidates = {"mammal", "canine", "pet", "hound", "carnivore", "animal", "canis", "wolf"}
        with patch.object(kf, "_wordnet_candidates", return_value=candidates):
            kf.add_learned_seeds(["dog"], reader)
        assert len(kf.frontier) <= 8
        assert all(c in candidates for c in kf.frontier)

    def test_exhausted_terms_excluded_from_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.exhausted = {"mammal"}
        reader = _make_reader({})
        with patch.object(kf, "_wordnet_candidates", return_value={"mammal", "canine"}):
            kf.add_learned_seeds(["dog"], reader)
        assert "mammal" not in kf.frontier


class TestNextTopics:
    def test_returns_n_topics(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["alpha", "beta", "gamma", "delta", "epsilon"]
        reader = _make_reader({})
        topics = kf.next_topics(reader, n=3)
        assert len(topics) == 3

    def test_returned_topics_removed_from_frontier(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["alpha", "beta", "gamma"]
        reader = _make_reader({})
        topics = kf.next_topics(reader, n=2)
        for t in topics:
            assert t not in kf.frontier

    def test_returns_fewer_if_frontier_small(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["only"]
        reader = _make_reader({})
        topics = kf.next_topics(reader, n=4)
        assert len(topics) == 1

    def test_prefers_sparser_topics(self):
        from hpm_ai_v6.cli.quiz_cli import KnowledgeFrontier
        kf = KnowledgeFrontier()
        kf.frontier = ["dense", "sparse"]
        reader = MagicMock()
        agent = MagicMock()

        def side_effect():
            return iter([{"name": "dense"}] * 10)

        agent.pattern_pager.iter_index_payloads.side_effect = side_effect
        reader.agents = {"a": agent}
        topics = kf.next_topics(reader, n=1)
        assert topics == ["sparse"]
