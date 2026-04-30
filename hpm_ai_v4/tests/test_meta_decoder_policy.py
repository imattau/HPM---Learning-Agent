import numpy as np

from hpm_ai_v4.agents.meta_decoder_policy import DecoderSpec, MetaDecoderPolicy


def test_bootstrap_strength_decays_but_never_disappears():
    policy = MetaDecoderPolicy(num_workers=1)
    initial = policy.bootstrap_strength()
    for _ in range(200):
        policy.observe(
            {"recent_agreement": 0.0, "recent_plausibility": 0.0, "structural_score": 0.0},
            DecoderSpec("word", "decode", True),
            {"token_agreement": 0.0, "plausibility": 0.0, "structural_score": 0.0},
        )
    later = policy.bootstrap_strength()
    assert later < initial
    assert later <= 0.1


def test_select_protects_underused_specs():
    policy = MetaDecoderPolicy(num_workers=1)
    policy._distribution = lambda context_obs: np.ones(policy.agent.obs_dim, dtype=np.float32) / float(policy.agent.obs_dim)
    policy._heuristic_bonus = lambda features, spec: 0.0

    used = DecoderSpec("word", "decode", True)
    fresh = DecoderSpec("char", "decode", True)
    policy._selection_counts[used.key()] = 25

    chosen = policy.select({}, [used, fresh])
    assert chosen == fresh


def test_select_responds_to_control_priors():
    policy = MetaDecoderPolicy(num_workers=1)
    policy._age = policy.bootstrap_warmup + 5
    policy._distribution = lambda context_obs: np.ones(policy.agent.obs_dim, dtype=np.float32) / float(policy.agent.obs_dim)
    policy._exploration_bonus = lambda spec: 0.0
    policy._reward_ema.clear()

    word = DecoderSpec("word", "decode", True)
    char = DecoderSpec("char", "decode", True)
    candidates = [word, char]

    word_features = {
        "requested_mode": "decode",
        "target_present": False,
        "validators_present": False,
        "recent_plausibility": 0.0,
        "recent_agreement": 0.0,
        "stage_idx": 0,
        "control_family_prior": {"word": 0.9, "char": 0.1},
        "control_mode_prior": {"decode": 1.0},
        "control_strength": 0.8,
        "control_dominant_family": "word",
        "control_dominant_mode": "decode",
    }
    char_features = dict(word_features)
    char_features["control_family_prior"] = {"word": 0.1, "char": 0.9}
    char_features["control_dominant_family"] = "char"

    chosen_word = policy.select(word_features, candidates)
    chosen_char = policy.select(char_features, candidates)

    assert chosen_word == word
    assert chosen_char == char


def test_state_dict_roundtrip_preserves_spec_mapping():
    policy = MetaDecoderPolicy(num_workers=1)
    first = DecoderSpec("word", "decode", True)
    second = DecoderSpec("char", "target", False)
    policy._spec_code(first)
    policy._spec_code(second)
    policy.observe(
        {"recent_agreement": 0.6, "recent_plausibility": 0.4, "structural_score": 0.2},
        first,
        {"token_agreement": 0.4, "plausibility": 0.5, "structural_score": 0.1},
    )

    clone = MetaDecoderPolicy(num_workers=1)
    clone.load_state_dict(policy.state_dict())

    assert clone._spec_codes == policy._spec_codes
    assert clone._next_spec_code == policy._next_spec_code
    assert clone._selection_counts == policy._selection_counts
    assert clone._reward_ema == policy._reward_ema
    assert clone._metacognitive_reward_ema == policy._metacognitive_reward_ema
    assert clone._context_reward_ema == policy._context_reward_ema
    assert clone._context_reliability == policy._context_reliability


def test_select_uses_metacognitive_reliability():
    policy = MetaDecoderPolicy(num_workers=1)
    policy._distribution = lambda context_obs: np.ones(policy.agent.obs_dim, dtype=np.float32) / float(policy.agent.obs_dim)
    policy._heuristic_bonus = lambda features, spec: 0.0
    policy._control_bonus = lambda features, spec: 0.0
    policy._exploration_bonus = lambda spec: 0.0

    word = DecoderSpec("word", "decode", True)
    char = DecoderSpec("char", "decode", True)
    candidates = [word, char]

    features = {"recent_agreement": 0.0, "recent_plausibility": 0.0, "structural_score": 0.0, "stage_idx": 0}
    policy.observe(features, word, {"token_agreement": 1.0, "plausibility": 1.0, "structural_score": 1.0})

    chosen = policy.select(features, candidates, learn=False)

    assert chosen == word


def test_exploration_bonus_is_higher_early():
    policy = MetaDecoderPolicy(num_workers=1)
    spec = DecoderSpec("word", "decode", True)

    early = policy._exploration_bonus(spec)
    policy._age = 4000
    late = policy._exploration_bonus(spec)

    assert early > late
    assert early > 0.03
