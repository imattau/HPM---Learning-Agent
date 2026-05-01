# hpm_ai_v4/tests/test_layered_agent.py
import numpy as np
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.agents.meta_decoder_policy import DecoderSpec
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary

def test_layered_agent_perceive_runs():
    agent = LayeredAgent(num_workers=1)
    for i in range(20):
        agent.perceive(i % 95)

def test_layered_agent_perceive_injects_meta_feedback(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    captured = {}

    def fake_control_context(context_obs, top_k=None, feature_pack=None):
        return {
            "mode_prior": {"continue": 0.7, "repair": 0.3},
            "family_prior": {"word": 0.8},
            "stage_prior": {"surface": 1.0},
            "community_strength": 0.6,
            "dominant_mode": "continue",
            "dominant_family": "word",
            "dominant_stage": "surface",
            "summary_count": 2,
        }

    def capture_l1(obs, feedback=None):
        captured["l1"] = dict(feedback or {})

    def capture_l2(obs, feedback=None):
        captured["l2"] = dict(feedback or {})

    def capture_l3(obs, feedback=None):
        captured["l3"] = dict(feedback or {})

    monkeypatch.setattr(agent.l1.reasoner, "control_context", fake_control_context)
    monkeypatch.setattr(agent.l1, "perceive_and_learn", capture_l1)
    monkeypatch.setattr(agent.l2, "perceive_and_learn", capture_l2)
    monkeypatch.setattr(agent.l3, "perceive_and_learn", capture_l3)

    agent.perceive(ord("a") - 32)

    assert captured["l1"]["control_dominant_mode"] == "continue"
    assert captured["l2"]["control_strength"] == 0.6
    assert 0.0 <= captured["l3"]["topdown_gate"] <= 1.0
    assert "meta_structural_score" in captured["l3"]

def test_layered_agent_obs_dims():
    agent = LayeredAgent(num_workers=1)
    assert agent.l1.obs_dim == 95
    assert agent.l2.obs_dim == 16
    assert agent.l3.obs_dim == 16
    assert agent.layer_latent_dims["l2"] == 8
    assert agent.layer_latent_dims["l3"] == 8
    assert agent.l2.patterns[0].latent_dim == 8
    assert agent.l3.patterns[0].latent_dim == 8


def test_layered_agent_allows_custom_l3_latent_width():
    agent = LayeredAgent(num_workers=1, layer_latent_dims={"l3": 12})
    assert agent.layer_latent_dims["l3"] == 12
    assert agent.l3.patterns[0].latent_dim == 12


def test_layered_agent_word_surface_mode():
    agent = LayeredAgent(num_workers=1, surface_mode="word")
    agent.observe_text("hello world", feedback_mode="target")
    assert agent.surface_mode == "word"
    assert agent.l1.obs_dim > 95
    assert agent._surface_history_text()

def test_layered_agent_equal_weights():
    agent = LayeredAgent(num_workers=1)
    # Check that initial weights are as expected (not 1.0)
    assert max(p.weight for p in agent.l1.patterns) < 0.2
    assert max(p.weight for p in agent.l2.patterns) < 0.2
    assert set(agent.decoders.keys()) == {"char", "word", "target", "constrained", "explain"}
    assert agent.decoder_policy is not None

def test_layered_agent_generate_returns_string():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    assert isinstance(result, str)
    assert len(result) > 0
    assert 'L3:' in result

def test_layered_agent_generate_printable():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    result = agent.generate(steps=20)
    # Generation is a readable label sequence.
    assert all(32 <= ord(ch) <= 126 for ch in result)


def test_layered_agent_generate_text_returns_readable_text():
    agent = LayeredAgent(num_workers=1)
    for i in range(200):
        agent.perceive(i % 95)
    result = agent.generate_text(steps=40)
    assert isinstance(result, str)
    assert len(result) > 0
    assert 'L3:' not in result
    assert any(ch.isalpha() for ch in result)


def test_generate_text_uses_meta_policy(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)

    seen = {}

    def select(features, candidates, learn=True):
        seen["features"] = features
        seen["candidates"] = candidates
        return DecoderSpec("word", "decode", True)

    monkeypatch.setattr(agent.decoder_policy, "select", select)
    result = agent.generate_text(steps=10, seed_text="the ")
    assert isinstance(result, str)
    assert "features" in seen
    assert "candidates" in seen
    assert agent._last_decoder_choice == "word"


def test_generate_text_can_skip_policy_learning():
    agent = LayeredAgent(num_workers=1)
    for i in range(80):
        agent.perceive(i % 95)

    before_age = agent.decoder_policy._age
    before_counts = dict(agent.decoder_policy._selection_counts)
    result = agent.generate_text(steps=8, seed_text="the ", target_text="quick brown fox", update_policy=False)
    assert isinstance(result, str)
    assert agent.decoder_policy._age == before_age
    assert dict(agent.decoder_policy._selection_counts) == before_counts


def test_layered_agent_generate_chars_returns_printable_text():
    agent = LayeredAgent(num_workers=1)
    for i in range(200):
        agent.perceive(i % 95)
    result = agent.generate_chars(steps=20, seed_text="abc", mode="target", target_text="def", include_seed=False)
    assert isinstance(result, str)
    assert len(result) > 0
    assert all(32 <= ord(ch) <= 126 for ch in result)

def test_predict_next_chars_returns_list():
    agent = LayeredAgent(num_workers=1)
    for i in range(50):
        agent.perceive(i % 95)
    # use a real context
    context = list(range(10))
    preds = agent.predict_next_chars(context, top_k=5)
    assert len(preds) <= 5
    assert all(isinstance(ch, str) and isinstance(prob, (float, np.float32, np.float64)) for ch, prob in preds)


def test_generate_text_with_validators_and_feedback():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    for i in range(200):
        agent.perceive(i % 95)
    text = agent.generate_text(steps=20, seed_text="the ")
    assert isinstance(text, str)
    assert len(text) > 0
    assert 'L3:' not in text
    assert agent.l1.reasoner.dictionary is dictionary
    assert agent.l1.reasoner.grammar is grammar


def test_observe_text_hybrid_feedback_and_evaluation():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    generated = "the quick brown fox"
    target = "the quick brown fox jumps"
    eval_stats = agent.evaluate_generated_text(generated, target)
    assert 0.0 <= eval_stats["token_agreement"] <= 1.0
    assert 0.0 <= eval_stats["plausibility"] <= 1.0

    stats = agent.observe_text(
        target,
        feedback_mode="hybrid",
        generated_text=generated,
        self_feedback_weight=0.02,
    )
    assert stats["target_chars"] > 0
    assert stats["self_chars"] >= 0
    assert 0.0 <= stats["token_agreement"] <= 1.0


def test_target_conditioned_generation_improves_agreement():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    corpus = "the quick brown fox jumps over the lazy dog. " * 8
    for ch in corpus:
        if ch == '\n':
            raw = 94
        else:
            raw = ord(ch) - 32
        agent.perceive(raw)

    seed = "the quick brown fox "
    target = "jumps over the lazy dog."
    decode = agent.generate_text(steps=8, seed_text=seed, mode="decode", include_seed=False)
    target_gen = agent.generate_text(
        steps=8,
        seed_text=seed,
        target_text=target,
        mode="target",
        include_seed=False,
    )

    decode_score = agent.evaluate_generated_text(decode, target)["token_agreement"]
    target_score = agent.evaluate_generated_text(target_gen, target)["token_agreement"]
    assert target_score >= decode_score


def test_l2_soft_state_returns_valid_symbol():
    agent = LayeredAgent(num_workers=1)
    for i in range(120):
        agent.perceive(i % 95)
    soft_state = agent.l2_soft_state()
    assert 0 <= soft_state < agent.l2.obs_dim
    dist = agent.l2_state_distribution()
    assert len(dist) == agent.l2.obs_dim


def test_l3_state_distribution_returns_normalized_vector():
    agent = LayeredAgent(num_workers=1)
    for i in range(120):
        agent.perceive(i % 95)
    dist = agent.l3_state_distribution()
    assert len(dist) == agent.l3.obs_dim
    assert abs(float(dist.sum()) - 1.0) < 1e-6
    assert 0 <= agent.l3_soft_state() < agent.SOFT_STATE_OBS_DIM


def test_l1_state_distribution_preserves_latent_uncertainty():
    agent = LayeredAgent(num_workers=1)
    for i in range(80):
        agent.perceive(i % 95)
    dist = agent.l1_state_distribution()
    assert len(dist) == 2
    assert abs(dist.sum() - 1.0) < 1e-6
    assert 0 <= agent.l1_soft_state() < agent.l2.obs_dim


def test_plan_text_continuation_returns_printable_text():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    corpus = "the quick brown fox jumps over the lazy dog. " * 6
    for ch in corpus:
        raw = 94 if ch == '\n' else ord(ch) - 32
        agent.perceive(raw)
    planned = agent.plan_text_continuation(
        target_text="jumps over the lazy dog.",
        seed_text="the quick brown fox ",
        horizon=8,
        strategy="beam",
    )
    assert isinstance(planned, str)
    assert len(planned) > 0
    assert all(32 <= ord(ch) <= 126 for ch in planned)
    assert agent.evaluate_generated_text(planned, "jumps over the lazy dog.")["token_agreement"] > 0.0


def test_plan_text_continuation_passes_feature_pack(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    captured = {}

    def fake_plan_sequence(*args, **kwargs):
        captured["kwargs"] = kwargs
        return [1, 1, 1]

    monkeypatch.setattr(agent.l1.reasoner, "plan_sequence", fake_plan_sequence)
    monkeypatch.setattr(agent, "_choose_char_for_class", lambda class_id, prev_char, preferred_char=None: "a")

    planned = agent.plan_text_continuation(
        target_text="abc",
        seed_text="seed",
        horizon=3,
        strategy="beam",
        feature_pack={"mode_prior": {"continue": 1.0}},
    )

    assert isinstance(planned, str)
    assert captured["kwargs"]["feature_pack"] == {"mode_prior": {"continue": 1.0}}


def test_simulate_continuation_ranks_and_restores_state(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    agent._raw_history.extend([1, 2, 3])
    agent._l1_state_history.extend([4, 5])
    agent._l2_state_history.extend([6, 7])
    agent._pending_feedback["reward"] = 0.4

    outputs = iter(["low fit", "high fit", "medium fit", "extra fit"])

    def fake_generate_text(**kwargs):
        agent._raw_history.append(999)
        return next(outputs)

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        agent,
        "_continuation_compression_score",
        lambda text: {"low fit": 0.1, "medium fit": 0.5, "high fit": 0.9, "extra fit": 0.2}[text],
    )

    snapshot = {
        "raw_history": list(agent._raw_history),
        "l1_state_history": list(agent._l1_state_history),
        "l2_state_history": list(agent._l2_state_history),
        "pending_feedback": dict(agent._pending_feedback),
    }

    result = agent.simulate_continuation("seed text", steps=3, candidates=3)

    assert [item["text"] for item in result] == ["high fit", "medium fit", "low fit"]
    assert result[0]["score"] >= result[-1]["score"]
    assert agent._raw_history == snapshot["raw_history"]
    assert agent._l1_state_history == snapshot["l1_state_history"]
    assert agent._l2_state_history == snapshot["l2_state_history"]
    assert agent._pending_feedback == snapshot["pending_feedback"]


def test_simulate_continuation_commit_best_replays_winner(monkeypatch):
    agent = LayeredAgent(num_workers=1)
    outputs = iter(["low fit", "high fit", "medium fit"])
    committed = {}

    def fake_generate_text(**kwargs):
        return next(outputs)

    def fake_observe_text(text, **kwargs):
        committed["text"] = text
        committed["feedback_signal"] = kwargs.get("feedback_signal", {})
        return {"target_chars": len(text)}

    monkeypatch.setattr(agent, "generate_text", fake_generate_text)
    monkeypatch.setattr(agent, "observe_text", fake_observe_text)
    monkeypatch.setattr(
        agent,
        "_continuation_compression_score",
        lambda text: {"low fit": 0.1, "medium fit": 0.5, "high fit": 0.9}[text],
    )

    before = list(agent._raw_history)
    result = agent.simulate_continuation("seed text", steps=3, candidates=3, commit_best=True)

    assert [item["text"] for item in result][0] == "high fit"
    assert committed["text"] == "high fit"
    assert committed["feedback_signal"]["kind"] == "simulation_commit"
    assert agent._raw_history == before


def test_generate_constrained_text_returns_readable_text():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    for i in range(200):
        agent.perceive(i % 95)
    text = agent.generate_constrained_text(
        steps=20,
        seed_text="the ",
        allowed_words={"the", "quick", "brown", "fox"},
    )
    assert isinstance(text, str)
    assert len(text) > 0
    assert all(32 <= ord(ch) <= 126 for ch in text)


def test_generate_constrained_text_with_target_mode():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    for i in range(200):
        agent.perceive(i % 95)
    text = agent.generate_constrained_text(
        steps=12,
        seed_text="the ",
        target_text="quick brown fox",
        mode="target",
        include_seed=False,
        allowed_words={"the", "quick", "brown", "fox"},
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_repair_text_without_constraints_preserves_word_spacing():
    dictionary = NLTKWordList(download=False)
    grammar = HeuristicGrammarLibrary()
    agent = LayeredAgent(num_workers=1, dictionary=dictionary, grammar=grammar)
    corpus = "the quick brown fox jumps over the lazy dog. " * 6
    for ch in corpus:
        raw = 94 if ch == '\n' else ord(ch) - 32
        agent.perceive(raw)

    repaired = agent.repair_text(
        corrupted_text="the quickbrown fox jumps over the lazy dog.",
        target_text="the quick brown fox jumps over the lazy dog.",
        use_constraints=False,
        update_policy=False,
    )

    assert isinstance(repaired, str)
    assert len(repaired) > 0
    assert " " in repaired
    assert "quickbrown" not in repaired.lower()


def test_layered_agent_bundle_persists_reasoner_memory(tmp_path):
    agent = LayeredAgent(num_workers=1, surface_mode="coarse")

    agent.l1.reasoner.record_episode([0, 1, 0, 1], action=1, reward=0.9, tag="train")
    agent.l2.reasoner.record_episode([1, 0, 1], action=0, reward=0.7, tag="train")

    base = tmp_path / "bundle"
    agent.save_bundle(str(base))

    assert (tmp_path / "bundle.reasoner.l1.json").exists()
    assert (tmp_path / "bundle.reasoner.l2.json").exists()

    loaded = LayeredAgent(num_workers=1)
    loaded.load_bundle(str(base))

    assert loaded.l1.reasoner.memory_size == 1
    assert loaded.l1.reasoner.memory[0].action == 1
    assert loaded.l2.reasoner.memory_size == 1
    assert loaded.l2.reasoner.memory[0].action == 0
    assert loaded.surface_mode == "coarse"
    assert loaded.l1.obs_dim == 5


def test_layered_agent_word_bundle_round_trip(tmp_path):
    agent = LayeredAgent(num_workers=1, surface_mode="word")
    agent.observe_text("hello world hello", feedback_mode="target")

    base = tmp_path / "word_bundle"
    agent.save_bundle(str(base))

    loaded = LayeredAgent(num_workers=1)
    loaded.load_bundle(str(base))

    assert loaded.surface_mode == "word"
    assert loaded.l1.obs_dim > 95
