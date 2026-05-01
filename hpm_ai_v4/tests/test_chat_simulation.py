import os
import numpy as np

from hpm_ai_v4.simulations.chat_simulation import BasicChatSession, ReverseChatSession, run_basic_chat_simulation, run_reverse_chat_simulation, run_binding_evaluator_benchmark, run_binding_width_sweep_benchmark, run_relational_emergence_benchmark, run_relational_emergence_sweep_benchmark, run_relational_component_sweep_benchmark, _resolve_chat_library_path
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.simulations.build_relational_corpus import build_relational_corpus
from hpm_ai_v4.simulations.build_relational_mixed_corpus import build_relational_mixed_corpus
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.text_signals import TextSignalPack, TextSignalExtractor


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


class _StubGrammar:
    def __init__(self):
        self._pos = {
            "it": "PR",
            "by": "IN",
            "gave": "VB",
            "gives": "VB",
        }
        self._lemma = {
            "gave": "give",
            "gives": "give",
        }

    def get_pos(self, word: str) -> str:
        return self._pos.get(word.lower(), "NN")

    def normalize_lemma(self, word: str) -> str:
        return self._lemma.get(word.lower(), word.lower())


def test_basic_chat_session_turn_updates_history():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(
        agent,
        history_window=3,
        response_steps=24,
        use_constraints=False,
    )
    result = session.chat_turn("Hello there.")

    assert isinstance(result.response_text, str)
    assert len(result.response_text) > 0
    assert "L3:" not in result.response_text
    assert len(session.history) == 2
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"
    assert "User:" in result.prompt_text
    assert "Assistant:" in result.prompt_text
    assert "User:" in session.transcript()
    assert "Assistant:" in session.transcript()


def test_basic_chat_session_chat_method_matches_turn_result():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    result = session.chat_turn("What can you do?")
    reply = session.chat("Give me a short answer.")

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert reply == session.history[-1].text
    assert result.response_text != ""


def test_basic_chat_session_uses_content_seed_and_no_unsupervised_policy_update(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = {}

    def fake_generate_text(**kwargs):
        captured["generate_text"] = kwargs
        return "fine reply."

    def fake_generate_chars(**kwargs):
        captured["generate_chars"] = kwargs
        return "alternate reply."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_chars)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    result = session.chat_turn("Hello there.")

    assert result.response_text in {
        "fine reply.",
        "alternate reply.",
        "What can I do for you?",
        "How can I help?",
        "Hello.",
        "Hi there.",
    }
    assert "generate_text" in captured
    assert "generate_chars" in captured
    assert "User:" not in captured["generate_text"]["seed_text"]
    assert "Assistant:" not in captured["generate_text"]["seed_text"]
    assert captured["generate_text"]["update_policy"] is False
    assert captured["generate_chars"]["update_policy"] is False


def test_basic_chat_session_uses_supplied_grammar_for_pos_and_lemma():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(
        agent,
        history_window=2,
        response_steps=16,
        use_constraints=False,
        grammar=_StubGrammar(),
    )

    assert session._is_pronoun_token("it")
    assert session._is_preposition_token("by")
    assert session._grammar_lemma("gave") == "give"
    assert session._semantic_tokens(["it", "gave", "by", "book"]) == ["gave", "book"]


def test_basic_chat_session_uses_text_signal_ranking(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    monkeypatch.setattr(agent, "_text_plausibility", lambda text: 0.5)

    class FakeSignals:
        def analyze(self, text, **kwargs):
            return TextSignalPack(
                structure_score=0.6,
                repeat_score=0.1 if "useful" in text else 0.8,
                commonness_score=0.7,
                sentence_confidence=0.9,
                target_alignment=0.0,
                source="fake",
            )

    session = BasicChatSession(
        agent,
        history_window=2,
        response_steps=16,
        use_constraints=False,
        text_signals=FakeSignals(),
    )

    chosen = session._choose_chat_candidate(
        ["bad bad bad.", "useful reply."],
        user_text="Hello there.",
    )

    assert chosen == "useful reply."


def test_basic_chat_session_uses_multiple_candidates_and_penalizes_echo(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    calls = {"generate_text": 0, "generate_chars": 0}

    def fake_generate_text(**kwargs):
        calls["generate_text"] += 1
        if calls["generate_text"] == 1:
            return "It will help us to relax."
        return "What do you mean?"

    def fake_generate_chars(**kwargs):
        calls["generate_chars"] += 1
        return "What do you mean?"

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_chars)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    session.history.extend(
        [
            type("T", (), {"role": "assistant", "text": "What do you mean?"})(),
            type("T", (), {"role": "assistant", "text": "It will help us to relax."})(),
        ]
    )

    response = session._generate_response("Assistant:", user_text="Why is conversation hard?")

    assert calls["generate_text"] >= 2
    assert calls["generate_chars"] >= 2
    assert response in {
        "What do you mean?",
        "It will help us to relax.",
        "Let me know what you need.",
        "I can help with that.",
        "Here is a short answer.",
        "The main idea is simple.",
        "What would you like next?",
    }
    assert response != "What do you mean?" or calls["generate_text"] == 1


def test_basic_chat_session_passes_dialogue_act_to_policy(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = {}

    def fake_generate_text(**kwargs):
        captured["generate_text"] = kwargs
        return "I can help with that."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    session.chat_turn("Hello there.")

    assert captured["generate_text"]["context_features"]["dialogue_act"] == "greeting"


def test_basic_chat_session_sentence_features_prefer_single_clear_sentence(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    monkeypatch.setattr(agent, "_text_plausibility", lambda text: 0.5)

    class FakeSignals:
        def analyze(self, text, **kwargs):
            return TextSignalPack(
                structure_score=0.5,
                repeat_score=0.0,
                commonness_score=0.5,
                sentence_confidence=0.8,
                target_alignment=0.0,
                source="fake",
            )

    session = BasicChatSession(
        agent,
        history_window=2,
        response_steps=16,
        use_constraints=False,
        text_signals=FakeSignals(),
    )

    single_sentence = "What do you mean?"
    two_sentences = "What do you mean? Please explain."
    chosen = session._choose_chat_candidate(
        [two_sentences, single_sentence],
        user_text="Can you clarify that?",
        dialogue_act="question",
    )

    assert chosen == single_sentence


def test_basic_chat_session_can_disable_sentence_features(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(
        agent,
        history_window=2,
        response_steps=16,
        use_constraints=False,
        use_sentence_features=False,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("sentence scoring should be disabled")

    monkeypatch.setattr(session, "_sentence_score", fail_if_called)
    score = session._chat_response_score("A clean reply.", "Hello there.", dialogue_act="greeting")

    assert isinstance(score, float)


def test_basic_chat_session_target_sentence_features_prefer_matching_shape(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    monkeypatch.setattr(agent, "_text_plausibility", lambda text: 0.5)

    class FakeSignals:
        def analyze(self, text, **kwargs):
            return TextSignalPack(
                structure_score=0.5,
                repeat_score=0.0,
                commonness_score=0.5,
                sentence_confidence=0.9 if text.count("?") == 1 else 0.4,
                target_alignment=0.0,
                source="fake",
            )

    session = BasicChatSession(
        agent,
        history_window=2,
        response_steps=16,
        use_constraints=False,
        text_signals=FakeSignals(),
        use_sentence_features=True,
    )

    matching = "What do you mean?"
    mismatching = "What do you mean? Please explain."
    chosen = session._choose_chat_candidate(
        [mismatching, matching],
        user_text="Can you clarify that?",
        dialogue_act="question",
        target_sentence_features={
            "sentence_count": 1,
            "dominant_sentence_type": "question",
            "sentence_confidence": 0.9,
        },
    )

    assert chosen == matching


def test_reverse_chat_session_asks_questions(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = {}

    def fake_generate_text(**kwargs):
        captured["generate_text"] = kwargs
        return "What do you mean"

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)

    session = ReverseChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    opening = session.ask()
    result = session.answer_turn("I need help with planning.")

    assert opening.endswith("?")
    assert result.response_text.endswith("?")
    assert captured["generate_text"]["context_features"]["conversation_mode"] == "reverse"
    assert captured["generate_text"]["context_features"]["desired_question"] is True


def test_basic_chat_session_discourse_state_persists_across_pronouns(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = []

    def fake_generate_text(**kwargs):
        captured.append(kwargs)
        return "It sat down."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    first = session.chat_turn("The cat chased the dog.")
    first_topic = session.discourse_state.topic
    first_confidence = session.discourse_state.topic_confidence
    second = session.chat_turn("It sat down.")

    assert first_topic != "unknown"
    assert "Focus:" in first.prompt_text
    assert second.prompt_text.count("Focus:") == 1
    assert session.discourse_state.topic == first_topic
    assert session.discourse_state.topic_confidence >= first_confidence
    assert session.discourse_state.active_entities
    assert any(call["context_features"]["discourse_topic"] == first_topic for call in captured)
    assert any("discourse_summary" in call["context_features"] for call in captured)


def test_reverse_chat_session_discourse_state_persists(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = []

    def fake_generate_text(**kwargs):
        captured.append(kwargs)
        return "What should I ask next?"

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = ReverseChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    opening = session.ask()
    first_topic = session.discourse_state.topic
    answer = session.answer_turn("The cat is on the mat.")

    assert opening.endswith("?")
    assert answer.response_text.endswith("?")
    assert first_topic != "unknown"
    assert "Focus:" in answer.prompt_text
    assert session.discourse_state.topic != "unknown"
    assert session.discourse_state.topic_confidence > 0.0
    assert session.discourse_state.active_entities
    assert any(call["context_features"]["discourse_topic"] != "unknown" for call in captured)
    assert any(call["context_features"]["discourse_summary"] for call in captured)
    assert any(call["context_features"]["conversation_mode"] == "reverse" for call in captured)


def test_basic_chat_session_bounded_history_compresses_old_turns(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It sits there."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    for idx in range(10):
        session.chat_turn(f"The cat {idx} chased the dog.")

    assert len(session.history) <= session.history_cap
    assert session.discourse_state.archived_entities
    assert session.discourse_state.archived_summary
    assert "archive=" in session.discourse_state.discourse_summary


def test_basic_chat_session_rewrites_relation_from_prompt(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    captured = {}

    def fake_generate_text(**kwargs):
        captured["generate_text"] = kwargs
        return "The cat sat."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")

    assert session.relational_state.subject == "cat"
    assert session.relational_state.proposition
    assert "Relation:" in captured["generate_text"]["seed_text"]
    assert captured["generate_text"]["context_features"]["relational_subject"] == "cat"
    assert captured["generate_text"]["context_features"]["relational_confidence"] > 0.0


def test_basic_chat_session_relational_state_distinguishes_active_and_passive_voice():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)

    session._update_relational_state("The dog chased the cat.", role="user", dialogue_act="default")
    active = session.relational_state.to_dict()

    session.relational_state.reset()
    session._update_relational_state("The cat was chased by the dog.", role="user", dialogue_act="default")
    passive = session.relational_state.to_dict()

    assert active["subject"] == "dog"
    assert active["object"] == "cat"
    assert active["voice"] == "active"
    assert passive["subject"] == "dog"
    assert passive["object"] == "cat"
    assert passive["voice"] == "passive"
    assert passive["role_bindings"]["grammatical_subject"] == "cat"
    assert passive["role_bindings"]["agent"] == "dog"


def test_basic_chat_session_clause_stack_preserves_nested_frames():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)

    session._update_relational_state("The cat that the dog chased sat on the mat.", role="user", dialogue_act="default")

    stack = session.relational_state.to_dict()["binding_stack"]
    subjects = [frame.get("subject") for frame in stack]

    assert len(stack) >= 2
    assert "cat" in subjects
    assert "dog" in subjects
    assert session.relational_state.role_bindings.get("main_subject") == "cat"


def test_basic_chat_session_query_chain_resolves_two_step_property():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat sat on the mat.")
    session.chat_turn("The mat is red.")

    result = session.query("What colour is the thing the cat sat on?")

    assert result["source"] == "chain_resolved"
    assert result["hops"] == 2
    assert result["answer"] == "red"


def test_basic_chat_session_query_chain_generalizes_property_hop():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat sat on the mat.")
    session.chat_turn("The mat is round.")

    result = session.query("What shape is the thing the cat sat on?")

    assert result["source"] == "chain_resolved"
    assert result["hops"] == 2
    assert result["answer"] == "round"
    assert result["property_hint"] == "shape"


def test_basic_chat_session_query_chain_supports_three_hops():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The dog chased the cat.")
    session.chat_turn("The cat sat on the mat.")
    session.chat_turn("The mat was red.")

    result = session.query("What colour is the thing sat on by the animal that the dog chased?")

    assert result["source"] == "chain_resolved"
    assert result["hops"] == 3
    assert result["answer"] == "red"
    assert result["pivot_source"] in {"by_chain", "clause", "relation"}


def test_basic_chat_session_world_model_tracks_transfer_possession():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.observe_text("John gave Mary the book.", role="user", feedback_mode="target")

    mary_record = session.discourse_state.entity_registry["mary"]
    john_record = session.discourse_state.entity_registry["john"]

    assert "book" in mary_record.get("possessions", [])
    assert "book" not in john_record.get("possessions", [])

    assert session.query("What does Mary have?")["answer"] == "book"
    assert session.query("Does Mary have the book?")["answer"] == "yes"
    assert session.query("Who has the book?")["answer"] == "mary"


def test_basic_chat_session_world_model_tracks_break_state():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.observe_text("The vase broke.", role="user", feedback_mode="target")

    vase_record = session.discourse_state.entity_registry["vase"]
    assert vase_record.get("states", {}).get("condition") == "broken"


def test_basic_chat_session_query_uses_entity_registry_after_pruning(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")
    for idx in range(8):
        session.chat_turn(f"The robot {idx} noted the signal.")

    result = session.query("Who chased the dog?")

    assert len(session.history) <= session.history_cap
    assert "cat" in session.discourse_state.entity_registry
    assert session.discourse_state.entity_registry["cat"]["last_predicate"] == "chased"
    assert result["answer"] == "cat"
    assert result["source"] in {"registry_subject", "relational_subject"}
    assert result["confidence"] > 0.0
    assert result["predicted_entity"]
    assert isinstance(result["prediction_matched"], bool)


def test_basic_chat_session_query_resolves_pronouns_from_registry(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")
    for idx in range(8):
        session.chat_turn(f"The robot {idx} noted the signal.")

    result = session.query("What about it?")

    assert result["source"] == "coreference"
    assert result["answer"] in session.discourse_state.entity_registry
    assert result["answer"] != "unknown"
    assert "predicted_entity" in result


def test_basic_chat_session_pending_feedback_reaches_next_sentence_learning(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)

    current_l3 = int(agent.l3_soft_state())
    session.discourse_state.entity_registry["cat"] = {
        "entity": "cat",
        "first_seen_turn": 0,
        "last_turn": 0,
        "last_role": "subject",
        "last_dialogue_act": "default",
        "last_sentence_type": "declarative",
        "last_subject": "cat",
        "last_predicate": "sat",
        "last_object": "mat",
        "mention_count": 3,
        "salience": 1.0,
        "binding_confidence": 0.9,
        "stability_score": 0.9,
        "prediction_hits": 3,
        "prediction_misses": 0,
        "last_position": 0,
        "dominant_latent_state": current_l3,
        "latent_states": [current_l3],
    }
    session.discourse_state.focus_stack = ["cat"]
    session.discourse_state.active_entities = ["cat"]
    session.discourse_state.topic = "cat"
    session.discourse_state.topic_confidence = 0.9

    observed_feedback = []
    original_perceive = agent.perceive

    def wrapped_perceive(raw_char_id, feedback=None):
        observed_feedback.append(dict(feedback or {}))
        return original_perceive(raw_char_id, feedback=feedback)

    monkeypatch.setattr(agent, "perceive", wrapped_perceive)

    stats_one = session.observe_text("The cat sat.", role="user", feedback_mode="target")
    pending = dict(agent._pending_feedback)
    assert float(pending.get("topdown_gate", 0.0)) > 0.0
    assert float(pending.get("reward", 0.0)) > 0.0
    assert stats_one["binding_predictions"] == 1
    assert stats_one["binding_prediction_hits"] + stats_one["binding_prediction_misses"] == 1

    observed_feedback.clear()
    stats_two = session.observe_text("The cat slept.", role="user", feedback_mode="target")

    assert any(float(payload.get("topdown_gate", 0.0)) > 0.0 for payload in observed_feedback)
    assert any(float(payload.get("reward", 0.0)) > 0.0 for payload in observed_feedback)
    assert stats_two["binding_predictions"] == 1
    assert stats_two["binding_prediction_hits"] + stats_two["binding_prediction_misses"] == 1


def test_basic_chat_session_query_handles_did_question_word_order(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The dog chased the cat.")

    result = session.query("Who did the dog chase?")

    assert result["answer"] == "cat"
    assert result["source"] == "registry_object"
    assert result["prediction_matched"] in {True, False}


def test_basic_chat_session_query_handles_plain_who_subject(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")

    result = session.query("Who chased the dog?")

    assert result["answer"] == "cat"
    assert result["source"] == "registry_subject"


def test_basic_chat_session_query_handles_nested_clause_subject(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat that the dog chased sat on the mat.")
    session.chat_turn("The mat was red.")

    result = session.query("Who sat on the mat?")

    assert result["answer"] == "cat"
    assert result["source"] == "registry_subject"


def test_basic_chat_session_query_handles_passive_subject_binding(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The dog was chased by the cat.")

    result = session.query("Who chased the dog?")

    assert result["answer"] == "cat"
    assert result["source"] in {"registry_subject", "relational_subject"}


def test_basic_chat_session_query_handles_what_subject_lookup(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The robot moved toward the mat.")

    result = session.query("What moved toward the mat?")

    assert result["answer"] == "robot"
    assert result["source"] in {"registry_subject", "relational_subject"}


def test_basic_chat_session_query_handles_copula_property_lookup(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The trophy was red.")

    result = session.query("What was red?")

    assert result["answer"] == "trophy"
    assert result["source"] in {"property", "relational_property"}


def test_basic_chat_session_query_handles_negation_with_low_confidence(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")

    result = session.query("What didn't the cat do?")

    assert result["negated"] is True
    assert result["source"] == "negated_query"
    assert result["answer"] == "unknown"
    assert result["confidence"] <= session.relational_state.confidence + 1e-6


def test_binding_confidence_calibrates_from_match_and_mismatch(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def fake_generate_text(**kwargs):
        return "It helps."

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "generate_chars", fake_generate_text)

    session = BasicChatSession(agent, history_window=2, response_steps=16, use_constraints=False)
    session.chat_turn("The cat chased the dog.")

    before = session.relational_state.confidence
    match_feedback = session.record_binding_feedback("cat", "cat")
    after_match = session.relational_state.confidence
    mismatch_feedback = session.record_binding_feedback("cat", "dog")
    after_mismatch = session.relational_state.confidence

    assert match_feedback["matched"] is True
    assert mismatch_feedback["matched"] is False
    assert after_match > before
    assert after_mismatch < after_match
    assert session.discourse_state.entity_registry["cat"]["binding_confidence"] > 0.0
    query_result = session.query("Who chased the dog?")
    assert query_result["predicted_entity"]
    assert isinstance(query_result["prediction_matched"], bool)


def test_basic_chat_session_falls_back_to_dialogue_bank(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)

    def bad_generate_text(**kwargs):
        return "L3: L3: L3: L3:"

    def bad_generate_chars(**kwargs):
        return "L3: L3: L3: L3:"

    monkeypatch.setattr(agent, "generate_text", bad_generate_text)
    monkeypatch.setattr(agent, "generate_chars", bad_generate_chars)

    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    response = session.chat_turn("Hello there.").response_text

    assert response in {"Hello.", "Hi there.", "How can I help?", "What can I do for you?"}


def test_run_binding_evaluator_benchmark_reports_accuracy(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)

    report = run_binding_evaluator_benchmark(
        corpus_path=str(corpus),
        warmup_chars=32,
        history_window=2,
        response_steps=8,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert report["aggregate"]["case_count"] == 3
    assert report["aggregate"]["binding_predictions"] >= 3
    assert report["aggregate"]["binding_prediction_hits"] + report["aggregate"]["binding_prediction_misses"] == report["aggregate"]["binding_predictions"]
    assert 0.0 <= report["aggregate"]["avg_binding_prediction_accuracy"] <= 1.0
    assert 0.0 <= report["aggregate"]["avg_answer_accuracy"] <= 1.0
    assert (tmp_path / "binding_evaluator_benchmark.json").exists()


def test_run_binding_width_sweep_benchmark_reports_comparison(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)

    calls = []

    def fake_run_binding_evaluator_benchmark(*args, **kwargs):
        width = int((kwargs.get("layer_latent_dims") or {}).get("l3", 0))
        calls.append(width)
        return {
            "aggregate": {
                "case_count": 3,
                "avg_answer_accuracy": 0.25 + 0.05 * (width // 4),
                "avg_binding_prediction_accuracy": 0.50 + 0.02 * (width // 4),
                "avg_confidence": 0.40 + 0.01 * (width // 4),
            },
            "cases": [],
        }

    monkeypatch.setattr("hpm_ai_v4.simulations.chat_simulation.run_binding_evaluator_benchmark", fake_run_binding_evaluator_benchmark)

    report = run_binding_width_sweep_benchmark(
        corpus_path=str(corpus),
        widths=[8, 12],
        warmup_chars=32,
        history_window=2,
        response_steps=8,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert calls == [8, 12]
    assert report["baseline_width"] == 8
    assert len(report["arms"]) == 2
    assert report["comparison"][1]["delta_answer_accuracy"] > 0.0
    assert (tmp_path / "binding_width_sweep_benchmark.json").exists()


def test_run_relational_emergence_benchmark_reports_ablation(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "The dog chased the cat. The cat was chased by the dog. "
        "The robot moved toward the mat. The mat was approached by the robot. "
        "The cat that the dog chased sat on the mat. The mat was sat on by the cat that the dog chased. "
        * 4
    )

    class FakeStream:
        def __init__(self, path):
            self.path = path

        def __iter__(self):
            return iter([1] * 256)

    class FakeLayeredAgent:
        def __init__(self, *args, **kwargs):
            self.dictionary = None
            self.grammar = None
            self._last_text = ""
            self._l2_state_history = []

        def _tokenize_words(self, text):
            import re
            return re.findall(r"[A-Za-z']+", text)

        def observe_text(self, text, **kwargs):
            self._last_text = text.lower()
            self._l2_state_history.extend([0] * max(1, len(text)))
            return {
                "binding_predictions": 1,
                "binding_prediction_hits": 1,
                "binding_prediction_misses": 0,
            }

        def load_bundle(self, *args, **kwargs):
            return 0

        def l3_state_distribution(self):
            vec = np.zeros(8, dtype=np.float32)
            idx = 1 if (" was " in f" {self._last_text} " or " by " in f" {self._last_text} ") else 0
            vec[idx] = 1.0
            return vec

        def l3_viterbi_path(self, obs_seq):
            import re
            markers = ["was", "were", "been", "by", "that", "which", "who", "chased", "sat", "gave", "said", "moved", "approached"]
            path = [0] * max(1, len(obs_seq))
            cue_positions = []
            for marker in markers:
                for match in re.finditer(rf"\b{re.escape(marker)}\b", self._last_text):
                    cue_positions.append(match.start())
            cue_positions = sorted({pos for pos in cue_positions if pos < len(path)})
            state = 0
            prev = 0
            for pos in cue_positions:
                state = 1 - state
                for idx in range(prev, pos):
                    if idx < len(path):
                        path[idx] = state
                prev = pos
            for idx in range(prev, len(path)):
                path[idx] = state
            return path

        def l3_soft_state(self):
            return int(np.argmax(self.l3_state_distribution()))

    from pytest import MonkeyPatch
    monkeypatch = MonkeyPatch()
    try:
        monkeypatch.setattr("hpm_ai_v4.simulations.chat_simulation.LayeredAgent", FakeLayeredAgent)
        monkeypatch.setattr("hpm_ai_v4.simulations.chat_simulation.WikipediaStream", FakeStream)

        report = run_relational_emergence_benchmark(
            corpus_path=str(corpus),
            warmup_chars=32,
            history_window=2,
            response_steps=8,
            num_workers=1,
            use_dict=False,
            checkpoint_dir=str(tmp_path),
        )

        assert report["baseline_arm"] == "heuristics_on"
        assert len(report["arms"]) == 2
        assert report["arms"][0]["aggregate"]["case_count"] == 6
        assert report["arms"][1]["use_relational_heuristics"] is False
        assert "active_passive_separation_ratio" in report["arms"][0]["aggregate"]
        assert report["arms"][0]["active_passive_separation"]["avg_between"] >= 0.0
        assert report["corpus_spec"]["latent_probe"]["layer"] == "l3"
        assert "avg_role_segmentation_alignment_ratio" in report["arms"][0]["aggregate"]
        assert report["arms"][0]["aggregate"]["avg_role_segmentation_alignment_ratio"] >= 0.0
        assert len(report["matrix"]) == 12
        assert {row["arm"] for row in report["matrix"]} == {"heuristics_on", "heuristics_off"}
        assert (tmp_path / "relational_emergence_benchmark.json").exists()
    finally:
        monkeypatch.undo()


def test_build_relational_corpus_balances_families(tmp_path):
    corpus_path = tmp_path / "relational.txt"
    result = build_relational_corpus(str(corpus_path), sentence_count=20)

    assert result.corpus_path == str(corpus_path)
    assert result.lines_written == 20
    assert sum(result.family_counts.values()) == 20
    assert corpus_path.exists()
    text = corpus_path.read_text()
    assert "was chased by" in text
    assert "that" in text


def test_build_relational_mixed_corpus_includes_transfer_and_report_forms(tmp_path):
    corpus_path = tmp_path / "relational_mixed.txt"
    result = build_relational_mixed_corpus(str(corpus_path), sentence_count=21)

    assert result.corpus_path == str(corpus_path)
    assert result.lines_written == 21
    assert sum(result.family_counts.values()) == 21
    text = corpus_path.read_text()
    assert "gave the" in text
    assert "said the" in text


def test_run_relational_emergence_sweep_benchmark_reports_scales(tmp_path, monkeypatch):
    calls = []

    def fake_build_relational_corpus(output_path, *, sentence_count=50000, **kwargs):
        path = tmp_path / f"corpus_{sentence_count}.txt"
        path.write_text(f"size {sentence_count}\n")
        return type("R", (), {
            "corpus_path": str(path),
            "lines_written": sentence_count,
            "family_counts": {"active": sentence_count // 2, "passive": sentence_count // 2},
        })()

    def fake_run_relational_emergence_benchmark(*, corpus_path, **kwargs):
        calls.append((corpus_path, kwargs.get("checkpoint_dir", "")))
        return {
            "arms": [
                {"aggregate": {"avg_answer_accuracy": 0.40, "avg_binding_prediction_accuracy": 0.50, "active_passive_separation_ratio": 1.0}},
                {"aggregate": {"avg_answer_accuracy": 0.10, "avg_binding_prediction_accuracy": 0.20, "active_passive_separation_ratio": 0.4}},
            ]
        }

    monkeypatch.setattr("hpm_ai_v4.simulations.build_relational_corpus.build_relational_corpus", fake_build_relational_corpus)
    monkeypatch.setattr("hpm_ai_v4.simulations.build_relational_mixed_corpus.build_relational_mixed_corpus", fake_build_relational_corpus)
    monkeypatch.setattr("hpm_ai_v4.simulations.chat_simulation.run_relational_emergence_benchmark", fake_run_relational_emergence_benchmark)

    report = run_relational_emergence_sweep_benchmark(
        corpus_path=str(tmp_path / "unused.txt"),
        sentence_counts=[10, 20],
        corpus_modes=["focused", "mixed"],
        checkpoint_dir=str(tmp_path),
    )

    assert len(calls) == 4
    assert report["baseline_sentence_count"] == 10
    assert report["baseline_corpus_mode"] == "focused"
    assert [row["sentence_count"] for row in report["comparison"]] == [10, 20, 10, 20]
    assert [row["corpus_mode"] for row in report["comparison"]] == ["focused", "focused", "mixed", "mixed"]
    assert report["comparison"][0]["heuristics_on_answer_accuracy"] == 0.40
    assert (tmp_path / "relational_emergence_sweep_benchmark.json").exists()


def test_run_relational_component_sweep_benchmark_reports_matrix(tmp_path, monkeypatch):
    calls = []

    def fake_build(output_path, *, sentence_count=50000, **kwargs):
        path = tmp_path / f"built_{sentence_count}_{os.path.basename(output_path)}"
        path.write_text(f"size {sentence_count}\n")
        return type("R", (), {
            "corpus_path": str(path),
            "lines_written": sentence_count,
            "family_counts": {"active": sentence_count // 2, "passive": sentence_count // 2},
        })()

    def fake_run_relational_emergence_benchmark(*, corpus_path, session_kwargs=None, **kwargs):
        calls.append((corpus_path, dict(session_kwargs or {})))
        config_name = str((session_kwargs or {}).get("name", "config"))
        score = {
            "full": 0.60,
            "no_passive": 0.55,
            "no_clause": 0.52,
            "no_registry": 0.48,
            "all_off": 0.30,
        }.get(config_name, 0.40)
        return {
            "arms": [
                {"aggregate": {"avg_answer_accuracy": score, "avg_binding_prediction_accuracy": score / 2.0, "active_passive_separation_ratio": score + 0.1}}
            ]
        }

    monkeypatch.setattr("hpm_ai_v4.simulations.build_relational_corpus.build_relational_corpus", fake_build)
    monkeypatch.setattr("hpm_ai_v4.simulations.build_relational_mixed_corpus.build_relational_mixed_corpus", fake_build)
    monkeypatch.setattr("hpm_ai_v4.simulations.chat_simulation.run_relational_emergence_benchmark", fake_run_relational_emergence_benchmark)

    report = run_relational_component_sweep_benchmark(
        corpus_path=str(tmp_path / "unused.txt"),
        sentence_counts=[10],
        corpus_modes=["focused", "mixed"],
        checkpoint_dir=str(tmp_path),
    )

    assert len(calls) == 10
    assert report["baseline_config"] == "full"
    assert report["baseline_corpus_mode"] == "focused"
    assert len(report["matrix"]) == 10
    assert {row["config"] for row in report["matrix"]} == {"full", "no_passive", "no_clause", "no_registry", "all_off"}
    assert {row["corpus_mode"] for row in report["matrix"]} == {"focused", "mixed"}
    assert (tmp_path / "relational_component_sweep_benchmark.json").exists()


def test_chat_response_score_penalizes_recent_echo():
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    session.history.extend(
        [
            type("T", (), {"role": "assistant", "text": "It will help us to relax."})(),
            type("T", (), {"role": "assistant", "text": "What do you mean?"})(),
        ]
    )

    fresh = session._chat_response_score("A useful answer with detail.", "Why is conversation hard?")
    echo = session._chat_response_score("What do you mean?", "Why is conversation hard?")

    assert fresh > echo


def test_text_signal_extractor_penalizes_repetition():
    extractor = TextSignalExtractor(use_spacy=False)
    good = extractor.analyze("The assistant can help with that.", context_texts=["Hello there."])
    bad = extractor.analyze("the the the the the the", context_texts=["Hello there."])

    assert good.combined_score() > bad.combined_score()
    assert 0.0 <= good.structure_score <= 1.0
    assert 0.0 <= bad.repeat_score <= 1.0


def test_run_basic_chat_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)

    history = run_basic_chat_simulation(
        corpus_path=str(corpus),
        prompts=["Hello.", "What can you do?"],
        warmup_chars=64,
        response_steps=20,
        history_window=3,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) == 2
    final = history[-1]
    assert "user_text" in final
    assert "response_text" in final
    assert "prompt_text" in final
    assert isinstance(final["response_text"], str)
    assert len(final["response_text"]) > 0
    assert (tmp_path / "final_chat_library.l1.pkl").exists()
    assert (tmp_path / "final_chat_library.l2.pkl").exists()
    assert (tmp_path / "final_chat_library.l3.pkl").exists()
    assert (tmp_path / "final_chat_library.l4.pkl").exists()
    assert (tmp_path / "final_chat_library.l5.pkl").exists()


def test_run_reverse_chat_simulation_short(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)

    history = run_reverse_chat_simulation(
        corpus_path=str(corpus),
        answers=["I need a plan.", "Keep it brief."],
        warmup_chars=64,
        response_steps=20,
        history_window=3,
        num_workers=1,
        use_dict=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(history) == 3
    assert history[0]["response_text"].endswith("?")
    assert history[-1]["response_text"].endswith("?")
    assert (tmp_path / "final_reverse_chat_library.l1.pkl").exists()
    assert (tmp_path / "final_reverse_chat_library.l2.pkl").exists()
    assert (tmp_path / "final_reverse_chat_library.l3.pkl").exists()
    assert (tmp_path / "final_reverse_chat_library.l4.pkl").exists()
    assert (tmp_path / "final_reverse_chat_library.l5.pkl").exists()


def test_resolve_chat_library_path_prefers_existing_default(tmp_path, monkeypatch):
    base = tmp_path / "daily_dialog_chat_library"
    (tmp_path / "daily_dialog_chat_library.l1.pkl").write_text("stub")
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_REGISTRY_CANDIDATES",
        ["/tmp/does-not-exist"],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_LIBRARY_CANDIDATES",
        [str(base), "/tmp/does-not-exist"],
    )

    resolved = _resolve_chat_library_path()
    assert resolved == str(base)


def test_resolve_chat_library_path_prefers_mixed_library(tmp_path, monkeypatch):
    mixed = tmp_path / "conversational_chat_library"
    daily = tmp_path / "daily_dialog_chat_library"
    (tmp_path / "conversational_chat_library.l1.pkl").write_text("stub")
    (tmp_path / "daily_dialog_chat_library.l1.pkl").write_text("stub")
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_REGISTRY_CANDIDATES",
        ["/tmp/does-not-exist"],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_LIBRARY_CANDIDATES",
        [str(mixed), str(daily)],
    )

    resolved = _resolve_chat_library_path()
    assert resolved == str(mixed)


def test_resolve_chat_library_path_prefers_ultra_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "chat_ultra_bundle"
    mixed = tmp_path / "conversational_chat_library"
    (tmp_path / "chat_ultra_bundle.l1.pkl").write_text("stub")
    (tmp_path / "conversational_chat_library.l1.pkl").write_text("stub")
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_REGISTRY_CANDIDATES",
        ["/tmp/does-not-exist"],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_LIBRARY_CANDIDATES",
        [str(bundle), str(mixed)],
    )

    resolved = _resolve_chat_library_path()
    assert resolved == str(bundle)


def test_resolve_chat_library_path_prefers_registry_view(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    registry = LibraryRegistry(str(registry_path))
    bundle = tmp_path / "chat_ultra_bundle"
    text = tmp_path / "text_seed.pkl"
    (tmp_path / "chat_ultra_bundle.l1.pkl").write_text("stub")
    text.write_text("stub")
    registry.upsert(
        name="chat_ultra_bundle_seed",
        path=str(bundle),
        domain="chat",
        status="promoted",
        bundle_kind="stacked",
        level_contract="l1-l5",
        obs_dims=[5, 10, 10, 32, 64],
        pattern_count=64,
    )
    registry.upsert(
        name="text_seed",
        path=str(text),
        domain="text",
        status="promoted",
        bundle_kind="flat",
        level_contract="l1",
        obs_dims=[5],
        pattern_count=12,
    )

    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_REGISTRY_CANDIDATES",
        [str(registry_path)],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_LIBRARY_CANDIDATES",
        ["/tmp/does-not-exist"],
    )

    resolved = _resolve_chat_library_path()
    assert resolved == str(bundle)


def test_resolve_chat_library_path_prefers_repo_nltk_large_base(tmp_path, monkeypatch):
    repo_base = os.path.join(os.getcwd(), "library_bootstrap", "nltk_large", "nltk_large_nlp_2000")
    assert os.path.exists(repo_base + ".pkl")

    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_REGISTRY_CANDIDATES",
        ["/tmp/does-not-exist"],
    )
    monkeypatch.setattr(
        "hpm_ai_v4.simulations.chat_simulation.CHAT_LIBRARY_CANDIDATES",
        [repo_base, "/tmp/does-not-exist"],
    )

    resolved = _resolve_chat_library_path()
    assert resolved == repo_base + ".pkl"
