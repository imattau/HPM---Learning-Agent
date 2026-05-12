# hpm_ai_v6/tests/test_syntactic_rule_agent.py
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from hpm_ai_v6.agents.syntactic_rule_agent import SyntacticRuleAgent
from hpm_ai_v6.hpm_model.core.cell import Cell


def make_mock_nlp(sentences_to_pos):
    """Returns a mock spaCy nlp whose .pipe() yields pre-specified token/POS pairs.
    sentences_to_pos: List[List[Tuple[str, str]]] — (text, pos_) per sentence.
    """
    def nlp_pipe(texts, **kwargs):
        for token_list in sentences_to_pos:
            doc = MagicMock()
            doc.__iter__ = MagicMock(return_value=iter([
                MagicMock(text=t, pos_=p) for t, p in token_list
            ]))
            yield doc
    mock = MagicMock()
    mock.pipe = nlp_pipe
    return mock


SIMPLE_NLP = make_mock_nlp([
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
])


class TestSyntacticRuleAgentInterface:
    def test_has_patterns_list(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent.patterns, list)

    def test_get_weights_returns_list(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent.get_weights(), list)

    def test_paging_lookup_returns_dict(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        assert isinstance(agent._paging_lookup(), dict)

    def test_weights_parallel_to_patterns(self):
        agent = SyntacticRuleAgent(nlp=SIMPLE_NLP)
        agent.learn_from_corpus(["the cat sat", "the dog ran"])
        assert len(agent.get_weights()) == len(agent.patterns)


class TestCorpusLearning:
    def setup_method(self):
        self.nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
        ])
        self.agent = SyntacticRuleAgent(min_prob=0.1, nlp=self.nlp)
        self.agent.learn_from_corpus(["the cat sat", "the dog ran"])

    def test_produces_dim3_cells(self):
        assert all(p.dim == 3 for p in self.agent.patterns)

    def test_cells_have_subgraph_derivation_rule_type(self):
        for p in self.agent.patterns:
            assert p.metadata.get("rule_type") == "subgraph_derivation"

    def test_cells_have_two_antecedent_edges(self):
        for p in self.agent.patterns:
            assert "antecedent_edges" in p.metadata
            assert len(p.metadata["antecedent_edges"]) == 2

    def test_det_noun_verb_chain_exists(self):
        names = {p.name for p in self.agent.patterns}
        assert "syn_rule_DET_NOUN_VERB" in names

    def test_weights_are_probabilities(self):
        for w in self.agent.get_weights():
            assert 0.0 <= w <= 1.0

    def test_high_prob_transition_has_high_weight(self):
        det_noun_verb = next(
            p for p in self.agent.patterns if p.name == "syn_rule_DET_NOUN_VERB"
        )
        # DET->NOUN is 100%, NOUN->VERB is 100% -> joint ~1.0
        assert det_noun_verb.weight > 0.5

    def test_cell_has_source_and_target(self):
        for p in self.agent.patterns:
            assert p.source is not None
            assert p.target is not None

    def test_embedding_is_difference_of_pos_edge_embeddings(self):
        for p in self.agent.patterns:
            expected = p.target.as_numpy() - p.source.as_numpy()
            np.testing.assert_allclose(p.as_numpy(), expected, atol=1e-6)

    def test_antecedent_relations_use_pos_prefix(self):
        for p in self.agent.patterns:
            for edge_spec in p.metadata["antecedent_edges"]:
                assert edge_spec["relation"].startswith("pos_")


class TestTagEdges:
    def setup_method(self):
        self.nlp = make_mock_nlp([
            [("cat", "NOUN"), ("sat", "VERB")],
        ])
        self.agent = SyntacticRuleAgent(nlp=self.nlp)
        self.agent.learn_from_corpus(["cat sat"])

    def test_tag_edges_runs_without_error(self):
        from hpm_ai_v6.agents.reasoning_agent import EdgeRecord
        src = Cell(name="word_cat", dim=0, embedding=np.zeros(16))
        tgt = Cell(name="word_sat", dim=0, embedding=np.ones(16))
        edge = Cell(name="w_cat->sat", dim=1, embedding=np.ones(16),
                    source=src, target=tgt)
        record = EdgeRecord(
            pattern=edge, source=src, target=tgt,
            score=0.5, raw_weight=0.5, agent_name="word",
            source_key="word:cat", target_key="word:sat",
            relation="lexical_transition",
        )
        # EdgeRecord is frozen — tag_edges is advisory only; must not raise
        self.agent.tag_edges({"word:cat": [record]})

    def test_paging_lookup_contains_pos_cells(self):
        lookup = self.agent._paging_lookup()
        assert any("pos_" in k for k in lookup)

    def test_get_pos_returns_tag_for_known_word(self):
        assert self.agent.get_pos("cat") == "NOUN"

    def test_get_pos_returns_none_for_unknown_word(self):
        assert self.agent.get_pos("xyzzy") is None


class TestMinProbThreshold:
    def test_high_min_prob_excludes_low_prob_transitions(self):
        # With min_prob=0.99, each bigram must be >= 0.99 to qualify.
        # DET->NOUN=1.0 (qualifies), NOUN->VERB=0.5, NOUN->ADJ=0.5 (both excluded < 0.99)
        # No qualifying trigram because NOUN's outgoing transitions are split 50/50.
        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("cat", "NOUN"), ("big", "ADJ")],
        ])
        agent = SyntacticRuleAgent(min_prob=0.99, nlp=nlp)
        agent.learn_from_corpus(["the cat sat", "the cat big"])
        assert len(agent.patterns) == 0

    def test_empty_corpus_produces_empty_patterns(self):
        nlp = make_mock_nlp([])
        agent = SyntacticRuleAgent(nlp=nlp)
        agent.learn_from_corpus([])
        assert agent.patterns == []
        assert agent.get_weights() == []

    def test_learn_from_corpus_is_idempotent_on_second_call(self):
        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
        ])
        agent = SyntacticRuleAgent(min_prob=0.1, nlp=nlp)
        agent.learn_from_corpus(["the cat sat"])
        count_first = len(agent.patterns)
        agent.learn_from_corpus(["the cat sat"])
        assert len(agent.patterns) == count_first  # reset, not accumulate


class TestReasoningAgentIntegration:
    def test_syntactic_rules_appear_in_forward_rule_patterns(self):
        from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent

        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
            [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
        ])
        syn_agent = SyntacticRuleAgent(min_prob=0.1, nlp=nlp)
        syn_agent.learn_from_corpus(["the cat sat", "the dog ran"])

        reader = SimpleNamespace(agents={"syntactic": syn_agent})
        ra = ReasoningAgent(reader=reader)
        ra.refresh()

        assert len(ra._forward_rule_patterns) > 0
        rule_names = {cell.name for _, cell in ra._forward_rule_patterns}
        assert "syn_rule_DET_NOUN_VERB" in rule_names

    def test_forward_rules_sorted_by_weight_descending(self):
        from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent

        nlp = make_mock_nlp([
            [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
        ])
        syn_agent = SyntacticRuleAgent(min_prob=0.01, nlp=nlp)
        syn_agent.learn_from_corpus(["the cat sat"])

        reader = SimpleNamespace(agents={"syntactic": syn_agent})
        ra = ReasoningAgent(reader=reader)
        ra.refresh()

        weights = [w for w, _ in ra._forward_rule_patterns]
        assert weights == sorted(weights, reverse=True)


class TestWiringIntegration:
    """Tests for wiring SyntacticRuleAgent into MultiAgentReader and ReasoningAgent."""

    def test_train_sequence_calls_syntactic_learn_from_corpus(self):
        """MultiAgentReader.train_sequence should call syntactic agent's learn_from_corpus."""
        from unittest.mock import MagicMock, patch
        import tempfile, os

        # Create a minimal corpus file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("cat sat\n")
            corpus_path = f.name

        try:
            nlp = make_mock_nlp([
                [("cat", "NOUN"), ("sat", "VERB")],
            ])
            syn_agent = SyntacticRuleAgent(min_prob=0.01, nlp=nlp)

            from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
            reader = MultiAgentReader(corpus_path=corpus_path, warm_start=False)
            reader.agents["syntactic"] = syn_agent

            # Call train_sequence — should invoke learn_from_corpus on syn_agent
            reader.train_sequence(["cat sat"])

            # After training, syntactic agent should have POS info
            assert syn_agent.get_pos("cat") == "NOUN"
        finally:
            os.unlink(corpus_path)

    def test_reasoning_agent_enriches_word_edges_with_pos_relation(self):
        """After refresh(), word agent edges for known-POS words have relation='pos_<TAG>'."""
        import numpy as np
        from types import SimpleNamespace
        from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent, EdgeRecord
        from hpm_ai_v6.hpm_model.core.cell import Cell

        alpha = Cell(name="word_cat", dim=0, embedding=[1, 0, 0])
        beta = Cell(name="word_sat", dim=0, embedding=[0, 1, 0])
        edge = Cell(
            name="w_cat->sat", dim=1,
            embedding=np.array(beta.as_numpy()) - np.array(alpha.as_numpy()),
            source=alpha, target=beta,
        )

        class StubWordAgent:
            patterns = [edge]
            def get_weights(self): return [0.5]
            def _paging_lookup(self): return {"word_cat": alpha, "word_sat": beta}

        nlp = make_mock_nlp([[("cat", "NOUN"), ("sat", "VERB")]])
        syn_agent = SyntacticRuleAgent(min_prob=0.01, nlp=nlp)
        syn_agent.learn_from_corpus(["cat sat"])

        reader = SimpleNamespace(agents={
            "word": StubWordAgent(),
            "syntactic": syn_agent,
        })
        ra = ReasoningAgent(reader)
        ra.refresh()

        # Find the edge from cat -> sat
        cat_key = ra._cell_key(alpha)
        records = ra._edge_index.get(cat_key, [])
        cat_sat = next((r for r in records if r.target.name == "word_sat"), None)
        assert cat_sat is not None, "Expected edge from word_cat to word_sat"
        assert cat_sat.relation == "pos_NOUN", f"Expected pos_NOUN, got {cat_sat.relation}"
