from types import SimpleNamespace

from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.semantic_agent import SemanticAgent
from hpm_ai_v6.agents.causal_agent import CausalRule
from hpm_ai_v6.hpm_model.core.cell import Cell


class StubAgent:
    def __init__(self, patterns=None, weights=None, lookup=None, sent_text_by_name=None):
        self.patterns = patterns or []
        self._weights = weights or []
        self._lookup = lookup or {}
        self.sent_text_by_name = sent_text_by_name or {}

    def get_weights(self):
        return list(self._weights)

    def _paging_lookup(self):
        return dict(self._lookup)


def make_edge(name, source, target):
    return Cell(
        name=name,
        dim=1,
        embedding=target.as_numpy() - source.as_numpy(),
        source=source,
        target=target,
    )


def make_analogy(name, source_pattern, target_pattern, weight=1.0):
    return Cell(
        name=name,
        dim=2,
        embedding=target_pattern.as_numpy() - source_pattern.as_numpy(),
        source=source_pattern,
        target=target_pattern,
        weight=weight,
    )


def make_rule(name, source_pattern, target_pattern, weight=1.0):
    return Cell(
        name=name,
        dim=3,
        embedding=target_pattern.as_numpy() - source_pattern.as_numpy(),
        source=source_pattern,
        target=target_pattern,
        weight=weight,
    )


def make_rule_with_metadata(name, source_pattern, target_pattern, metadata, weight=1.0):
    return Cell(
        name=name,
        dim=3,
        embedding=target_pattern.as_numpy() - source_pattern.as_numpy(),
        source=source_pattern,
        target=target_pattern,
        weight=weight,
        metadata=metadata,
    )


def test_semantic_agent_uses_stable_hashed_sentence_ids():
    agent = SemanticAgent()
    sentence_a = "A long sentence prefix that should no longer collide with another."
    sentence_b = "A long sentence prefix that should no longer collide with a different ending."

    cell_a = agent._get_or_create_sent_cell(sentence_a)
    cell_b = agent._get_or_create_sent_cell(sentence_b)

    assert cell_a is not None
    assert cell_b is not None
    assert cell_a.name != cell_b.name
    assert agent.sent_text_by_name[cell_a.name] == sentence_a
    assert agent.sent_text_by_name[cell_b.name] == sentence_b


def test_reasoning_prefers_calibrated_multi_hop_path_over_raw_heavy_dead_end():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    hole = Cell(name="word_hole", dim=0, embedding=[0.0, 0.0, 1.0])
    book = Cell(name="word_book", dim=0, embedding=[1.0, 1.0, 0.0])

    patterns = [
        make_edge("w_alice->book", alice, book),
        make_edge("w_alice->rabbit", alice, rabbit),
        make_edge("w_rabbit->hole", rabbit, hole),
    ]
    word_agent = StubAgent(
        patterns=patterns,
        weights=[0.99, 0.70, 0.65],
        lookup={cell.name: cell for cell in [alice, rabbit, hole, book]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    agent = ReasoningAgent(reader, beam_width=3, max_depth=3)
    answer = agent.reason("How does Alice connect to hole?")

    assert "alice -> rabbit -> hole" in answer.lower()
    assert "combined calibrated score" in answer
    assert "lexical_transition/word" in answer


def test_reasoning_resolves_sentence_cells_by_content_tokens():
    sentence = "Alice follows the rabbit into the deep tunnel."
    sent_cell = Cell(name="sent_deadbeefcafe", dim=0, embedding=[0.2, 0.2, 0.2])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    pattern = make_edge("sem_sent->rabbit", sent_cell, rabbit)

    semantic_agent = StubAgent(
        patterns=[pattern],
        weights=[0.8],
        lookup={sent_cell.name: sent_cell},
        sent_text_by_name={sent_cell.name: sentence},
    )
    word_agent = StubAgent(
        patterns=[],
        weights=[],
        lookup={rabbit.name: rabbit},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": semantic_agent,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    agent = ReasoningAgent(reader)
    answer = agent.reason("How does Alice connect to rabbit?")

    assert "alice follows the rabbit into the deep tunnel" in answer.lower()


def test_reasoning_prefers_structured_causal_explanation_for_why_question():
    cause = Cell(name="cause_rabbit->suddenly@2", dim=0, embedding=[1.0, 2.0, 1.0])
    effect = Cell(name="effect_rabbit->semantic", dim=0, embedding=[1.0, 1.0, 1.0])
    causal_rule = CausalRule(
        name="causal_rabbit_in_semantic",
        intervention="replace 'rabbit' at pos 2",
        effect_magnitude=0.42,
        agent_impacted="semantic",
        source=cause,
        target=effect,
        metadata={
            "original_word": "rabbit",
            "counterfactual_word": "suddenly",
            "position": 2,
            "agent_impacted": "semantic",
            "effect_label": "surprise in semantic",
        },
    )
    causal_agent = StubAgent(
        patterns=[causal_rule],
        weights=[0.9],
        lookup={cause.name: cause, effect.name: effect},
    )
    reader = SimpleNamespace(
        agents={
            "word": None,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
        }
    )

    agent = ReasoningAgent(reader)
    answer = agent.reason("Why is there surprise in semantic after rabbit?")

    assert "strongest causal explanation" in answer.lower()
    assert "replace 'rabbit' at pos 2" in answer
    assert "impacted agent=semantic" in answer
    assert "effect=0.4200" in answer


def test_reasoning_lists_causal_outgoing_transitions():
    cause = Cell(name="cause_alice->suddenly@1", dim=0, embedding=[1.0, 1.0, 1.0])
    effect = Cell(name="effect_alice->word", dim=0, embedding=[1.0, 2.0, 1.0])
    causal_rule = CausalRule(
        name="causal_alice_in_word",
        intervention="replace 'alice' at pos 1",
        effect_magnitude=0.31,
        agent_impacted="word",
        source=cause,
        target=effect,
        metadata={"original_word": "alice", "agent_impacted": "word"},
    )
    causal_agent = StubAgent(
        patterns=[causal_rule],
        weights=[0.7],
        lookup={cause.name: cause, effect.name: effect},
    )
    reader = SimpleNamespace(
        agents={
            "word": None,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
        }
    )

    agent = ReasoningAgent(reader)
    answer = agent.reason("How does alice connect?")

    assert "alice to suddenly at 1" in answer.lower()
    assert "surprise in word after alice" in answer.lower()
    assert "causal" in answer.lower()


def test_reasoning_builds_mixed_causal_and_learned_explanation_chain():
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    hole = Cell(name="word_hole", dim=0, embedding=[0.0, 0.0, 1.0])
    word_pattern = make_edge("w_rabbit->hole", rabbit, hole)
    word_agent = StubAgent(
        patterns=[word_pattern],
        weights=[0.8],
        lookup={rabbit.name: rabbit, hole.name: hole},
    )

    cause = Cell(name="cause_rabbit->suddenly@2", dim=0, embedding=[1.0, 2.0, 1.0])
    effect = Cell(name="effect_rabbit->semantic", dim=0, embedding=[1.0, 1.0, 1.0])
    causal_rule = CausalRule(
        name="causal_rabbit_in_semantic",
        intervention="replace 'rabbit' at pos 2",
        effect_magnitude=0.42,
        agent_impacted="semantic",
        source=cause,
        target=effect,
        metadata={
            "original_word": "rabbit",
            "counterfactual_word": "suddenly",
            "position": 2,
            "agent_impacted": "semantic",
            "effect_label": "surprise in semantic",
        },
    )
    causal_agent = StubAgent(
        patterns=[causal_rule],
        weights=[0.9],
        lookup={cause.name: cause, effect.name: effect},
    )

    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
        }
    )

    agent = ReasoningAgent(reader, beam_width=4, max_depth=4)
    answer = agent.reason("Why does surprise in semantic after rabbit connect to hole?")

    assert "strongest causal explanation" in answer.lower()
    assert "downstream chain:" in answer.lower()
    assert "surprise in semantic after rabbit -> rabbit -> hole" in answer.lower()


def test_reason_with_trace_returns_structured_path_trace():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    pattern = make_edge("w_alice->rabbit", alice, rabbit)
    word_agent = StubAgent(
        patterns=[pattern],
        weights=[0.8],
        lookup={alice.name: alice, rabbit.name: rabbit},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader).reason_with_trace("How does alice connect to rabbit?")

    assert trace["intent"] == "path"
    assert trace["mode"] == "connection"
    assert trace["anchors"]["start"]["label"] == "alice"
    assert trace["anchors"]["goal"]["label"] == "rabbit"
    assert trace["chosen_path"]["nodes"][0]["label"] == "alice"
    assert trace["chosen_path"]["nodes"][-1]["label"] == "rabbit"
    assert trace["chosen_path"]["steps"][0]["relation"] == "lexical_transition"


def test_reason_with_trace_returns_structured_mixed_explanation_trace():
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    hole = Cell(name="word_hole", dim=0, embedding=[0.0, 0.0, 1.0])
    word_pattern = make_edge("w_rabbit->hole", rabbit, hole)
    word_agent = StubAgent(
        patterns=[word_pattern],
        weights=[0.8],
        lookup={rabbit.name: rabbit, hole.name: hole},
    )
    cause = Cell(name="cause_rabbit->suddenly@2", dim=0, embedding=[1.0, 2.0, 1.0])
    effect = Cell(name="effect_rabbit->semantic", dim=0, embedding=[1.0, 1.0, 1.0])
    causal_rule = CausalRule(
        name="causal_rabbit_in_semantic",
        intervention="replace 'rabbit' at pos 2",
        effect_magnitude=0.42,
        agent_impacted="semantic",
        source=cause,
        target=effect,
        metadata={"original_word": "rabbit", "agent_impacted": "semantic"},
    )
    causal_agent = StubAgent(
        patterns=[causal_rule],
        weights=[0.9],
        lookup={cause.name: cause, effect.name: effect},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
        }
    )

    trace = ReasoningAgent(reader, beam_width=4, max_depth=4).reason_with_trace(
        "Why does surprise in semantic after rabbit connect to hole?"
    )

    assert trace["mode"] == "explanation"
    assert trace["anchors"]["effect"]["label"] == "surprise in semantic after rabbit"
    assert trace["anchors"]["goal"]["label"] == "hole"
    assert trace["chosen_path"]["nodes"][0]["label"] == "rabbit to suddenly at 2"
    assert trace["chosen_path"]["nodes"][-1]["label"] == "hole"
    assert any(step["relation"] == "causal_reentry" for step in trace["chosen_path"]["steps"])
    assert any(step["relation"] == "lexical_transition" for step in trace["chosen_path"]["steps"])


def test_reasoning_uses_semantic_fallback_when_terms_do_not_resolve_lexically():
    sentence = "The red queen studies impossible geometry."
    sent_cell = Cell(name="sent_semantic_a", dim=0, embedding=[0.9, 0.1, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 1.0, 0.0])
    semantic_pattern = make_edge("sem_queen", sent_cell, queen)

    class SemanticStub(StubAgent):
        def _get_or_create_sent_cell(self, text):
            if "monarch" in text.lower():
                return Cell(name="sent_query", dim=0, embedding=[0.88, 0.12, 0.0])
            return sent_cell

    semantic_agent = SemanticStub(
        patterns=[semantic_pattern],
        weights=[0.8],
        lookup={sent_cell.name: sent_cell},
        sent_text_by_name={sent_cell.name: sentence},
    )
    reader = SimpleNamespace(
        agents={
            "word": None,
            "contextual": None,
            "semantic": semantic_agent,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader).reason_with_trace("How does monarch connect?")

    assert trace["anchors"]["start"]["label"] == sentence
    assert trace["anchors"]["goal"]["label"] == "queen"
    assert trace["chosen_path"]["steps"][0]["relation"] == "semantic_transition"


def test_beam_search_avoids_revisiting_worse_cycle_paths():
    a = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    b = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    c = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])
    d = Cell(name="word_delta", dim=0, embedding=[1.0, 1.0, 0.0])
    patterns = [
        make_edge("w_a->b", a, b),
        make_edge("w_b->a", b, a),
        make_edge("w_b->c", b, c),
        make_edge("w_c->d", c, d),
        make_edge("w_a->d", a, d),
    ]
    word_agent = StubAgent(
        patterns=patterns,
        weights=[0.9, 0.85, 0.8, 0.75, 0.2],
        lookup={cell.name: cell for cell in [a, b, c, d]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=4, max_depth=5).reason_with_trace("How does alpha connect to delta?")

    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alpha", "beta", "gamma", "delta"]


def test_reasoning_prefers_explicit_dim2_analogy_patterns():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 0.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 1.0, 0.0])

    source_pattern = make_edge("w_alice->rabbit", alice, rabbit)
    target_pattern = make_edge("w_queen->crown", queen, crown)
    analogy_pattern = make_analogy("analogy_royal", source_pattern, target_pattern, weight=0.95)

    word_agent = StubAgent(
        patterns=[source_pattern, target_pattern, analogy_pattern],
        weights=[0.6, 0.55, 0.95],
        lookup={cell.name: cell for cell in [alice, rabbit, queen, crown]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    answer = ReasoningAgent(reader).reason("What is analogous to alice rabbit?")

    assert "strongest explicit learned analogies" in answer.lower()
    assert "w_queen->crown via analogy_royal" in answer


def test_reason_with_trace_reports_explicit_dim2_analogy_evidence():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 0.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 1.0, 0.0])

    source_pattern = make_edge("w_alice->rabbit", alice, rabbit)
    target_pattern = make_edge("w_queen->crown", queen, crown)
    analogy_pattern = make_analogy("analogy_royal", source_pattern, target_pattern, weight=0.95)

    word_agent = StubAgent(
        patterns=[source_pattern, target_pattern, analogy_pattern],
        weights=[0.6, 0.55, 0.95],
        lookup={cell.name: cell for cell in [alice, rabbit, queen, crown]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader).reason_with_trace("What is analogous to alice rabbit?")

    assert trace["intent"] == "analogy"
    assert trace["evidence"][0]["pattern"] == "analogy_royal"
    assert trace["evidence"][0]["target_pattern"] == "w_queen->crown"


def test_reasoning_prefers_explicit_dim3_rule_over_base_dim2_analogy():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 0.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 1.0, 0.0])
    king = Cell(name="word_king", dim=0, embedding=[0.5, 0.5, 1.0])
    throne = Cell(name="word_throne", dim=0, embedding=[1.0, 0.5, 0.5])

    source_pattern = make_edge("w_alice->rabbit", alice, rabbit)
    base_target = make_edge("w_queen->crown", queen, crown)
    refined_target = make_edge("w_king->throne", king, throne)
    analogy_pattern = make_analogy("analogy_royal", source_pattern, base_target, weight=0.9)
    rule_pattern = make_rule("rule_refine_royal", analogy_pattern, make_analogy("analogy_refined", base_target, refined_target, weight=0.88), weight=0.93)

    word_agent = StubAgent(
        patterns=[source_pattern, base_target, refined_target, analogy_pattern, rule_pattern, rule_pattern.target],
        weights=[0.6, 0.55, 0.53, 0.9, 0.93, 0.88],
        lookup={cell.name: cell for cell in [alice, rabbit, queen, crown, king, throne]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    answer = ReasoningAgent(reader).reason("What is analogous to alice rabbit?")

    assert "strongest compositional learned rules" in answer.lower()
    assert "analogy_refined via rule_refine_royal" in answer


def test_reason_with_trace_reports_dim3_rule_evidence():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 0.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 1.0, 0.0])
    king = Cell(name="word_king", dim=0, embedding=[0.5, 0.5, 1.0])
    throne = Cell(name="word_throne", dim=0, embedding=[1.0, 0.5, 0.5])

    source_pattern = make_edge("w_alice->rabbit", alice, rabbit)
    base_target = make_edge("w_queen->crown", queen, crown)
    refined_target = make_edge("w_king->throne", king, throne)
    analogy_pattern = make_analogy("analogy_royal", source_pattern, base_target, weight=0.9)
    refined_analogy = make_analogy("analogy_refined", base_target, refined_target, weight=0.88)
    rule_pattern = make_rule("rule_refine_royal", analogy_pattern, refined_analogy, weight=0.93)

    word_agent = StubAgent(
        patterns=[source_pattern, base_target, refined_target, analogy_pattern, refined_analogy, rule_pattern],
        weights=[0.6, 0.55, 0.53, 0.9, 0.88, 0.93],
        lookup={cell.name: cell for cell in [alice, rabbit, queen, crown, king, throne]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader).reason_with_trace("What is analogous to alice rabbit?")

    assert trace["intent"] == "analogy"
    assert trace["evidence"][0]["kind"] == "dim3_rule"
    assert trace["evidence"][0]["pattern"] == "rule_refine_royal"
    assert trace["evidence"][0]["source_pattern"] == "analogy_royal"


def test_beam_search_can_use_transient_rule_application_edge():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    queen = Cell(name="word_queen", dim=0, embedding=[0.0, 0.0, 1.0])
    crown = Cell(name="word_crown", dim=0, embedding=[1.0, 1.0, 0.0])
    king = Cell(name="word_king", dim=0, embedding=[0.5, 0.5, 1.0])
    throne = Cell(name="word_throne", dim=0, embedding=[1.0, 0.5, 0.5])

    source_pattern = make_edge("w_alice->rabbit", alice, rabbit)
    base_target = make_edge("w_queen->crown", queen, crown)
    refined_target = make_edge("w_king->throne", king, throne)
    analogy_pattern = make_analogy("analogy_royal", source_pattern, base_target, weight=0.9)
    refined_analogy = make_analogy("analogy_refined", base_target, refined_target, weight=0.88)
    rule_pattern = make_rule("rule_refine_royal", analogy_pattern, refined_analogy, weight=0.93)

    word_agent = StubAgent(
        patterns=[source_pattern, base_target, refined_target, analogy_pattern, refined_analogy, rule_pattern],
        weights=[0.6, 0.55, 0.53, 0.9, 0.88, 0.93],
        lookup={cell.name: cell for cell in [alice, rabbit, queen, crown, king, throne]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace("How does alice connect to throne?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alice", "throne"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "rule_application"
    assert trace["chosen_path"]["steps"][0]["agent"] == "reasoning"


def test_beam_search_can_use_forward_chained_transitivity_edge():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    bg = make_edge("w_beta->gamma", beta, gamma)
    trans_rule = Cell(name="transitivity_rule", dim=3, embedding=[0.1, 0.1, 0.1], weight=0.95)

    word_agent = StubAgent(
        patterns=[ab, bg, trans_rule],
        weights=[0.8, 0.75, 0.95],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace("How does alpha connect to gamma?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alpha", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "forward_chain"


def test_beam_search_can_use_multi_round_forward_chaining():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])
    delta = Cell(name="word_delta", dim=0, embedding=[1.0, 1.0, 0.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    bg = make_edge("w_beta->gamma", beta, gamma)
    gd = make_edge("w_gamma->delta", gamma, delta)
    trans_rule = Cell(name="transitivity_rule", dim=3, embedding=[0.1, 0.1, 0.1], weight=0.95)

    word_agent = StubAgent(
        patterns=[ab, bg, gd, trans_rule],
        weights=[0.8, 0.75, 0.7, 0.95],
        lookup={cell.name: cell for cell in [alpha, beta, gamma, delta]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=4).reason_with_trace("How does alpha connect to delta?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alpha", "delta"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "forward_chain"


def test_forward_chaining_supports_generalized_rule_endpoint_schema():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])
    delta = Cell(name="word_delta", dim=0, embedding=[1.0, 1.0, 0.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    gd = make_edge("w_gamma->delta", gamma, delta)
    helper_src = make_analogy("analogy_src", ab, gd, weight=0.7)
    helper_tgt = make_analogy("analogy_tgt", gd, ab, weight=0.65)
    generalized_rule = make_rule_with_metadata(
        "bridge_rule",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "edge_derivation",
            "pair_mode": "any_reachable",
            "derive_source": "source.source",
            "derive_target": "target.source",
        },
        weight=0.9,
    )

    word_agent = StubAgent(
        patterns=[ab, gd, helper_src, helper_tgt, generalized_rule],
        weights=[0.8, 0.75, 0.7, 0.65, 0.9],
        lookup={cell.name: cell for cell in [alpha, beta, gamma, delta]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace("How does alpha connect to gamma?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alpha", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "forward_chain"


def test_forward_chaining_supports_shared_source_subgraph_rules():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    ag = make_edge("w_alpha->gamma", alpha, gamma)
    helper_src = make_analogy("analogy_src", ab, ag, weight=0.7)
    helper_tgt = make_analogy("analogy_tgt", ag, ab, weight=0.65)
    subgraph_rule = make_rule_with_metadata(
        "shared_source_rule",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "subgraph_derivation",
            "pair_mode": "any_reachable",
            "antecedent_edges": [
                {"source_var": "x", "target_var": "y"},
                {"source_var": "x", "target_var": "z"},
            ],
            "consequent": {
                "source_var": "y",
                "target_var": "z",
            },
        },
        weight=0.92,
    )

    word_agent = StubAgent(
        patterns=[ab, ag, helper_src, helper_tgt, subgraph_rule],
        weights=[0.8, 0.77, 0.7, 0.65, 0.92],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace("How does beta connect to gamma?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["beta", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "subgraph_rule"


def test_forward_chaining_supports_shared_target_subgraph_rules():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ba = make_edge("w_beta->alpha", beta, alpha)
    ga = make_edge("w_gamma->alpha", gamma, alpha)
    helper_src = make_analogy("analogy_src_target", ba, ga, weight=0.7)
    helper_tgt = make_analogy("analogy_tgt_target", ga, ba, weight=0.65)
    subgraph_rule = make_rule_with_metadata(
        "shared_target_rule",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "subgraph_derivation",
            "pair_mode": "any_reachable",
            "antecedent_edges": [
                {"source_var": "y", "target_var": "x"},
                {"source_var": "z", "target_var": "x"},
            ],
            "consequent": {
                "source_var": "y",
                "target_var": "z",
            },
        },
        weight=0.91,
    )

    word_agent = StubAgent(
        patterns=[ba, ga, helper_src, helper_tgt, subgraph_rule],
        weights=[0.8, 0.78, 0.7, 0.65, 0.91],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace("How does beta connect to gamma?")

    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["beta", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "subgraph_rule"


def test_backward_chaining_proves_transitive_goal_directedly():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    bg = make_edge("w_beta->gamma", beta, gamma)
    trans_rule = Cell(name="transitivity_rule", dim=3, embedding=[0.1, 0.1, 0.1], weight=0.95)

    word_agent = StubAgent(
        patterns=[ab, bg, trans_rule],
        weights=[0.8, 0.75, 0.95],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace(
        "How does alpha connect to gamma?",
        method="backward",
    )

    assert trace["method"] == "backward"
    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["alpha", "beta", "gamma"]
    assert [step["relation"] for step in trace["chosen_path"]["steps"]] == [
        "lexical_transition",
        "lexical_transition",
    ]


def test_backward_chaining_works_for_explanation_downstream_paths():
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    hole = Cell(name="word_hole", dim=0, embedding=[0.0, 0.0, 1.0])
    burrow = Cell(name="word_burrow", dim=0, embedding=[1.0, 0.0, 1.0])
    rh = make_edge("w_rabbit->hole", rabbit, hole)
    hb = make_edge("w_hole->burrow", hole, burrow)
    trans_rule = Cell(name="transitivity_rule", dim=3, embedding=[0.1, 0.1, 0.1], weight=0.95)
    word_agent = StubAgent(
        patterns=[rh, hb, trans_rule],
        weights=[0.8, 0.76, 0.95],
        lookup={cell.name: cell for cell in [rabbit, hole, burrow]},
    )

    cause = Cell(name="cause_rabbit->suddenly@2", dim=0, embedding=[1.0, 2.0, 1.0])
    effect = Cell(name="effect_rabbit->semantic", dim=0, embedding=[1.0, 1.0, 1.0])
    causal_rule = CausalRule(
        name="causal_rabbit_in_semantic",
        intervention="replace 'rabbit' at pos 2",
        effect_magnitude=0.42,
        agent_impacted="semantic",
        source=cause,
        target=effect,
        metadata={"original_word": "rabbit", "agent_impacted": "semantic"},
    )
    causal_agent = StubAgent(
        patterns=[causal_rule],
        weights=[0.9],
        lookup={cause.name: cause, effect.name: effect},
    )

    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": causal_agent,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=4).reason_with_trace(
        "Why does surprise in semantic after rabbit connect to burrow?",
        method="backward",
    )

    assert trace["method"] == "backward"
    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels[0] == "rabbit to suddenly at 2"
    assert labels[-1] == "burrow"


def test_backward_chaining_supports_shared_source_subgraph_rules():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ab = make_edge("w_alpha->beta", alpha, beta)
    ag = make_edge("w_alpha->gamma", alpha, gamma)
    helper_src = make_analogy("analogy_src_backward", ab, ag, weight=0.7)
    helper_tgt = make_analogy("analogy_tgt_backward", ag, ab, weight=0.65)
    subgraph_rule = make_rule_with_metadata(
        "shared_source_rule_backward",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "subgraph_derivation",
            "pair_mode": "any_reachable",
            "antecedent_edges": [
                {"source_var": "x", "target_var": "y"},
                {"source_var": "x", "target_var": "z"},
            ],
            "consequent": {
                "source_var": "y",
                "target_var": "z",
            },
        },
        weight=0.92,
    )

    word_agent = StubAgent(
        patterns=[ab, ag, helper_src, helper_tgt, subgraph_rule],
        weights=[0.8, 0.77, 0.7, 0.65, 0.92],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace(
        "How does beta connect to gamma?",
        method="backward",
    )

    assert trace["method"] == "backward"
    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["beta", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "subgraph_rule"


def test_backward_chaining_supports_shared_target_subgraph_rules():
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    gamma = Cell(name="word_gamma", dim=0, embedding=[0.0, 0.0, 1.0])

    ba = make_edge("w_beta->alpha", beta, alpha)
    ga = make_edge("w_gamma->alpha", gamma, alpha)
    helper_src = make_analogy("analogy_src_target_backward", ba, ga, weight=0.7)
    helper_tgt = make_analogy("analogy_tgt_target_backward", ga, ba, weight=0.65)
    subgraph_rule = make_rule_with_metadata(
        "shared_target_rule_backward",
        helper_src,
        helper_tgt,
        metadata={
            "rule_type": "subgraph_derivation",
            "pair_mode": "any_reachable",
            "antecedent_edges": [
                {"source_var": "y", "target_var": "x"},
                {"source_var": "z", "target_var": "x"},
            ],
            "consequent": {
                "source_var": "y",
                "target_var": "z",
            },
        },
        weight=0.91,
    )

    word_agent = StubAgent(
        patterns=[ba, ga, helper_src, helper_tgt, subgraph_rule],
        weights=[0.8, 0.78, 0.7, 0.65, 0.91],
        lookup={cell.name: cell for cell in [alpha, beta, gamma]},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    trace = ReasoningAgent(reader, beam_width=5, max_depth=3).reason_with_trace(
        "How does beta connect to gamma?",
        method="backward",
    )

    assert trace["method"] == "backward"
    assert trace["chosen_path"] is not None
    labels = [node["label"] for node in trace["chosen_path"]["nodes"]]
    assert labels == ["beta", "gamma"]
    assert trace["chosen_path"]["steps"][0]["relation"] == "subgraph_rule"


def test_refresh_only_runs_once_across_repeated_queries():
    alice = Cell(name="word_alice", dim=0, embedding=[1.0, 0.0, 0.0])
    rabbit = Cell(name="word_rabbit", dim=0, embedding=[0.0, 1.0, 0.0])
    pattern = make_edge("w_alice->rabbit", alice, rabbit)
    word_agent = StubAgent(
        patterns=[pattern],
        weights=[0.8],
        lookup={alice.name: alice, rabbit.name: rabbit},
    )
    reader = SimpleNamespace(
        agents={
            "word": word_agent,
            "contextual": None,
            "semantic": None,
            "phrase": None,
            "char": None,
            "causal": None,
        }
    )

    agent = ReasoningAgent(reader)
    assert agent._dirty is True

    agent.reason("How does alice connect to rabbit?")
    assert agent._dirty is False
    index_id_after_first = id(agent._edge_index)

    agent.reason("How does alice connect to rabbit?")
    assert id(agent._edge_index) == index_id_after_first

    agent.invalidate()
    assert agent._dirty is True
    agent.reason("How does alice connect to rabbit?")
    assert agent._dirty is False
    assert id(agent._edge_index) != index_id_after_first


def test_parse_question_explanation_detected_mid_sentence():
    from types import SimpleNamespace
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    result = agent._parse_question("Can you explain why alice connects to rabbit?")
    assert result["type"] == "path"
    assert result["mode"] == "explanation"


def test_parse_question_analogy_detected_by_synonym():
    from types import SimpleNamespace
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    result = agent._parse_question("Compare alice and rabbit.")
    assert result["type"] == "analogy"


def test_parse_question_connection_detected_mid_sentence():
    from types import SimpleNamespace
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    result = agent._parse_question("Tell me how alice relates to rabbit.")
    assert result["type"] == "path"
    assert result["mode"] == "connection"


def test_parse_question_explanation_takes_priority_over_connection():
    from types import SimpleNamespace
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    result = agent._parse_question("How do you explain why alice connects?")
    assert result["mode"] == "explanation"


# ── Phase 2: Cross-Agent Evidence Synthesis ──────────────────────────────────

def _make_parallel_path_reader(word_score: float, phrase_score: float):
    """Two agents each providing a direct edge from 'alpha' to 'beta'.

    Each agent also carries a high-weight decoy edge to a dead-end node so
    that calibration produces a non-trivial spread (the target edge gets a
    percentile < 1.0), making noisy-OR synthesis meaningfully > individual scores.
    """
    alpha = Cell(name="word_alpha", dim=0, embedding=[1.0, 0.0, 0.0])
    beta = Cell(name="word_beta", dim=0, embedding=[0.0, 1.0, 0.0])
    dead_word = Cell(name="word_dead_w", dim=0, embedding=[1.0, 1.0, 0.0])
    dead_phrase = Cell(name="word_dead_p", dim=0, embedding=[1.0, 0.0, 1.0])

    edge_word = Cell(
        name="w_alpha->beta_word",
        dim=1,
        embedding=beta.as_numpy() - alpha.as_numpy(),
        source=alpha,
        target=beta,
    )
    decoy_word = Cell(
        name="w_alpha->dead_word",
        dim=1,
        embedding=dead_word.as_numpy() - alpha.as_numpy(),
        source=alpha,
        target=dead_word,
    )
    edge_phrase = Cell(
        name="w_alpha->beta_phrase",
        dim=1,
        embedding=beta.as_numpy() - alpha.as_numpy(),
        source=alpha,
        target=beta,
    )
    decoy_phrase = Cell(
        name="w_alpha->dead_phrase",
        dim=1,
        embedding=dead_phrase.as_numpy() - alpha.as_numpy(),
        source=alpha,
        target=dead_phrase,
    )

    # Decoy weight is higher so target edge calibrates below 1.0
    word_agent = StubAgent(
        patterns=[edge_word, decoy_word],
        weights=[word_score, 0.99],
        lookup={"word_alpha": alpha, "word_beta": beta, "word_dead_w": dead_word},
    )
    phrase_agent = StubAgent(
        patterns=[edge_phrase, decoy_phrase],
        weights=[phrase_score, 0.99],
        lookup={"word_alpha": alpha, "word_beta": beta, "word_dead_p": dead_phrase},
    )
    return SimpleNamespace(
        agents={
            "word": word_agent,
            "phrase": phrase_agent,
            "contextual": None,
            "semantic": None,
            "char": None,
            "causal": None,
        }
    )


def test_noisy_or_score_unit():
    """_noisy_or_score([0.5, 0.4]) == 1 - (1-0.5)*(1-0.4) == 0.7"""
    from types import SimpleNamespace
    import pytest
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    result = agent._noisy_or_score([0.5, 0.4])
    assert result == pytest.approx(0.7, abs=1e-9)


def test_noisy_or_score_single_path_is_identity():
    """With one path score, noisy-OR equals that score."""
    from types import SimpleNamespace
    import pytest
    reader = SimpleNamespace(agents={})
    agent = ReasoningAgent(reader)
    assert agent._noisy_or_score([0.6]) == pytest.approx(0.6, abs=1e-9)


def test_two_weak_parallel_paths_populate_candidate_paths():
    """Both paths from word and phrase agents must appear in candidate_paths."""
    import pytest
    reader = _make_parallel_path_reader(word_score=0.5, phrase_score=0.4)
    agent = ReasoningAgent(reader, beam_width=5, max_depth=3)
    trace = agent.reason_with_trace("how does alpha connect to beta")
    assert len(trace["candidate_paths"]) >= 2


def test_two_weak_parallel_paths_outscore_single_strong_path():
    """Synthesized noisy-OR score must exceed every individual path's score."""
    reader_two = _make_parallel_path_reader(word_score=0.5, phrase_score=0.4)
    agent_two = ReasoningAgent(reader_two, beam_width=5, max_depth=3)
    trace_two = agent_two.reason_with_trace("how does alpha connect to beta")

    assert len(trace_two["candidate_paths"]) >= 2, "Expected multiple candidate paths"
    individual_scores = [p["combined_score"] for p in trace_two["candidate_paths"]]
    synthesized = trace_two["chosen_path"]["combined_score"]
    assert synthesized > max(individual_scores), (
        f"Synthesized {synthesized:.4f} must exceed best individual {max(individual_scores):.4f}"
    )


def test_reasoning_agent_stores_relation_registry_from_reader():
    """ReasoningAgent must read relation_registry from reader if present."""
    from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
    from hpm_ai_v6.hpm_model.storage.relation_registry import RelationRegistry
    from types import SimpleNamespace

    reg = RelationRegistry(embedding_dim=4)
    reader = SimpleNamespace(
        agents={
            "word": None, "syntactic": None, "phrase": None,
            "contextual": None, "semantic": None, "char": None, "causal": None,
        },
        relation_registry=reg,
    )
    ra = ReasoningAgent(reader, beam_width=3, max_depth=2)
    assert ra._relation_registry is reg
