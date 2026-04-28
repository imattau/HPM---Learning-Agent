import os

from hpm_ai_v4.simulations.chat_simulation import BasicChatSession, ReverseChatSession, run_basic_chat_simulation, run_reverse_chat_simulation, _resolve_chat_library_path
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.text_signals import TextSignalPack, TextSignalExtractor


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


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
