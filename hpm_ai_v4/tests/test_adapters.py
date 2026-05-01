import pytest
from hpm_ai_v4.io.adapters import (
    CharClassAdapter,
    AsciiCharAdapter,
    LearnedSubstrateAdapter,
    CodeDSLAdapter,
    EpisodeBundleAdapter,
    CurriculumAdapter,
    EnvironmentStateAdapter,
    MathTextAdapter,
    WordAdapter,
    SentenceAdapter,
    SympyMathAdapter,
    StructuredTextAdapter,
    ToolActionAdapter,
)

@pytest.fixture
def adapter():
    return CharClassAdapter()

def test_letter_lowercase(adapter):
    # 'a' = ord('a') - 32 = 65
    assert adapter.encode(65) == 0

def test_letter_uppercase(adapter):
    # 'A' = ord('A') - 32 = 33
    assert adapter.encode(33) == 0

def test_digit(adapter):
    # '5' = ord('5') - 32 = 21
    assert adapter.encode(21) == 1

def test_space(adapter):
    # ' ' = ord(' ') - 32 = 0
    assert adapter.encode(0) == 2

def test_newline(adapter):
    # '\n' = ord('\n') - 32 = -22
    assert adapter.encode(-22) == 4

def test_punctuation(adapter):
    # '!' = ord('!') - 32 = 1
    assert adapter.encode(1) == 3

def test_decode_class_letter(adapter):
    assert adapter.decode_class(0) == 'letter'

def test_decode_class_digit(adapter):
    assert adapter.decode_class(1) == 'digit'

def test_decode_class_space(adapter):
    assert adapter.decode_class(2) == 'space'

def test_decode_class_punctuation(adapter):
    assert adapter.decode_class(3) == 'punctuation'

def test_decode_class_newline(adapter):
    assert adapter.decode_class(4) == 'newline'

def test_round_trip_letter(adapter):
    char_id = 65  # 'a'
    assert adapter.decode_class(adapter.encode(char_id)) == 'letter'

def test_round_trip_digit(adapter):
    char_id = 16  # '0'
    assert adapter.decode_class(adapter.encode(char_id)) == 'digit'

def test_obs_dim(adapter):
    assert adapter.obs_dim == 5

def test_decode_invalid_raises(adapter):
    with pytest.raises(ValueError):
        adapter.decode_class(5)

def test_encode_char_letter():
    a = CharClassAdapter()
    assert a.encode_char('a') == 0

def test_encode_char_digit():
    a = CharClassAdapter()
    assert a.encode_char('3') == 1

def test_encode_char_space():
    a = CharClassAdapter()
    assert a.encode_char(' ') == 2

def test_encode_char_punctuation():
    a = CharClassAdapter()
    assert a.encode_char('!') == 3

def test_encode_char_newline():
    a = CharClassAdapter()
    assert a.encode_char('\n') == 4


def test_ascii_char_adapter_preserves_surface_tokens():
    a = AsciiCharAdapter()
    assert a.obs_dim == 95
    assert a.encode_char('a') != a.encode_char('b')
    assert a.encode_char(' ') != a.encode_char('a')
    assert a.encode_char('\n') == 94
    assert a.bucket_for_token(a.encode_char('a')) == "letter"


def test_learned_substrate_adapter_discovers_stable_merges():
    adapter = LearnedSubstrateAdapter(max_merges=8, lowercase=True)
    texts = ["the cat", "the dog", "the bat"]
    latent_paths = [
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
    ]

    learned = adapter.fit_merges(texts, latent_paths, min_support=3, purity_threshold=1.0, max_ngram=3)

    assert learned >= 1
    assert adapter.obs_dim > 95
    tokens = adapter.to_observations("the cat")
    decoded = [adapter.decode_token(tok) for tok in tokens]
    assert "the" in decoded
    assert adapter.encode_token("the") >= 95
    assert "the" in adapter.from_observations(tokens)


def test_word_adapter_round_trip_and_vocab_growth():
    a = WordAdapter(max_vocab_size=64)
    tokens = a.to_observations("Hello world, hello test.")
    assert tokens
    assert a.obs_dim == 64
    assert a.decode_token(tokens[0]) == "hello"
    assert a.from_observations(tokens).startswith("hello world")
    assert a.encode_token("newtoken") != a.encode_token("hello")


def test_word_adapter_semantic_aliases_collapse_synonyms():
    a = WordAdapter(
        max_vocab_size=32,
        canonical_aliases={"feline": "cat", "canine": "dog", "reply": "answer"},
    )
    tokens = a.to_observations("The feline and the canine reply.")
    decoded = [a.decode_token(tok) for tok in tokens]

    assert "cat" in decoded
    assert "dog" in decoded
    assert "answer" in decoded


def test_word_adapter_build_semantic_vocab_orders_common_function_words_first():
    vocab = WordAdapter.build_semantic_vocab(
        [
            "The cat chased the dog.",
            "The cat saw the dog.",
            "The feline saw the canine.",
        ],
        max_vocab_size=32,
        min_freq=1,
        canonical_aliases={"feline": "cat", "canine": "dog"},
    )

    assert vocab["<UNK>"] == 0
    assert vocab["<BOS>"] == 1
    assert vocab["<EOS>"] == 2
    assert vocab["<NL>"] == 3
    assert vocab["<PARA>"] == 4
    assert vocab["the"] < vocab["cat"]
    assert "feline" not in vocab
    assert "canine" not in vocab


def test_word_adapter_vocab_contract_exposes_hpm_guarantees():
    contract = WordAdapter.vocab_contract()
    checklist = WordAdapter.vocab_checklist()

    assert contract["stable_tokenization"] is True
    assert contract["bounded_vocab"] is True
    assert "canonical_aliases" in contract["bundle_keys"]
    assert "<UNK>" in contract["reserved_tokens"]
    assert checklist
    assert any("bounded" in item.lower() for item in checklist)

def test_encode_char_uppercase():
    a = CharClassAdapter()
    assert a.encode_char('Z') == 0


def test_environment_state_adapter_encodes_structured_state():
    adapter = EnvironmentStateAdapter(obs_dim=8)
    tokens = adapter.to_observations({"state": 5, "family": 2, "reward": 1.2})
    assert tokens == [5, 2, 1]


def test_environment_state_adapter_handles_sequences():
    adapter = EnvironmentStateAdapter(obs_dim=8)
    tokens = adapter.to_observations([1, 2, 3])
    assert tokens == [1, 2, 3]


def test_tool_action_adapter_round_trips_actions():
    adapter = ToolActionAdapter(["inspect", "shift", "flip", "commit"])
    assert adapter.to_observations({"action": "flip"}) == [2]
    assert adapter.act(2) == "flip"


def test_curriculum_adapter_round_trips_families():
    adapter = CurriculumAdapter(["family_0", "family_1", "family_2"])
    tokens = adapter.to_observations({"family": "family_2", "phase": 0.5})
    assert tokens[0] == 2
    assert adapter.act(2) == "family_2"


def test_episode_bundle_adapter_encodes_and_decodes_descriptor():
    adapter = EpisodeBundleAdapter(obs_dim=16)
    tokens = adapter.encode_bundle({"kind": "bundle", "phase": "train", "level": 3, "count": 7})
    assert tokens
    decoded = adapter.decode_bundle(tokens)
    assert decoded["kind"] == "bundle"
    assert decoded["phase"] == "train"
    assert decoded["obs_dim"] == 3


def test_structured_text_adapter_round_trip_json():
    adapter = StructuredTextAdapter()
    payload = {"kind": "bundle", "phase": "train", "level": 3, "count": 7, "value": 0.75}
    text = adapter.to_text(payload)
    assert text == '{"count":7,"kind":"bundle","level":3,"phase":"train","value":0.75}'
    tokens = adapter.to_observations(payload)
    assert len(tokens) > 0
    decoded = adapter.from_text(text)
    assert decoded["kind"] == "bundle"
    assert decoded["phase"] == "train"
    assert decoded["level"] == 3


def test_math_text_adapter_canonicalizes_inline_equations():
    adapter = MathTextAdapter()
    text = "The area is A=pi r^2 and y = 2x + 1."
    canonical = adapter.to_text(text)
    spans = adapter.extract_math_spans(text)

    assert "A = pi r ^ 2" in canonical
    assert "y = 2x + 1" in canonical or "y = 2 x + 1" in canonical
    assert spans
    assert any("=" in span for span in spans)


def test_math_text_adapter_round_trips_observations():
    adapter = MathTextAdapter()
    text = "F = m a and E = m c^2"
    tokens = adapter.to_observations(text, max_length=64)
    decoded = adapter.from_observations(tokens)

    assert len(tokens) > 0
    assert isinstance(decoded, str)
    assert "=" in decoded
    assert "F" in decoded or "f" in decoded.lower()


def test_sentence_adapter_segments_and_classifies():
    adapter = SentenceAdapter()
    text = "Hello there. What do you mean? Please explain."
    spans = adapter.segment(text)

    assert len(spans) == 3
    assert spans[0].sentence_type == "declarative"
    assert spans[1].sentence_type == "question"
    assert spans[2].sentence_type == "request"


def test_sentence_adapter_observations_round_trip():
    adapter = SentenceAdapter()
    tokens = adapter.to_observations("Thanks. Can you clarify that?")
    decoded = adapter.from_observations(tokens)

    assert len(tokens) > 0
    assert "<closing:" in decoded
    assert "<question:" in decoded


def test_sentence_adapter_paragraph_markers_are_optional():
    adapter = SentenceAdapter()
    text = "Hello there.\n\nWhat do you mean?"
    default_tokens = adapter.to_observations(text)
    paragraph_tokens = adapter.to_observations(text, include_paragraph_markers=True)

    assert default_tokens != paragraph_tokens
    assert adapter.PARA_START_TOKEN in paragraph_tokens
    assert adapter.PARA_END_TOKEN in paragraph_tokens
    assert adapter.paragraph_markers(text) == [adapter.PARA_START_TOKEN, adapter.PARA_END_TOKEN, adapter.PARA_START_TOKEN, adapter.PARA_END_TOKEN]


def test_sentence_adapter_default_fallback_stays_usable():
    adapter = SentenceAdapter(use_spacy=False)
    spans = adapter.segment("Hello there. What do you mean?")

    assert len(spans) == 2
    assert spans[0].sentence_type == "declarative"
    assert spans[1].sentence_type == "question"


def test_sentence_adapter_can_use_spacy_like_segmenter(monkeypatch):
    class FakeSent:
        def __init__(self, text):
            self._text = text

        def __str__(self):
            return self._text

        def __iter__(self):
            return iter([])

    class FakeDoc:
        def __init__(self, text):
            self._sents = [FakeSent("Hello there."), FakeSent("Please explain.")]

        @property
        def sents(self):
            return self._sents

    class FakeNLP:
        pipe_names = ["sentencizer"]

        def __call__(self, text):
            return FakeDoc(text)

    monkeypatch.setattr("hpm_ai_v4.io.adapters.spacy", None, raising=False)
    adapter = SentenceAdapter(use_spacy=True)
    adapter._nlp = FakeNLP()
    spans = adapter.segment("Hello there. Please explain.")

    assert len(spans) == 2
    assert spans[1].sentence_type in {"request", "declarative"}


def test_sympy_math_adapter_parses_and_compares_equations():
    adapter = SympyMathAdapter()
    feedback = adapter.feedback("x + x", target_text="2*x")

    assert feedback["parseable"] is True
    assert feedback["equivalent"] is True
    assert "symbolic_score" in feedback


def test_sympy_math_adapter_simplifies_and_solves():
    adapter = SympyMathAdapter()
    simplified = adapter.simplify("x + x")
    solutions = adapter.solve("x + 2 = 5", symbol="x")

    assert simplified is not None
    assert "2" in simplified or "2*x" in simplified
    assert solutions == ["3"] or solutions == [3]


def test_code_dsl_adapter_canonicalizes_and_executes():
    adapter = CodeDSLAdapter()
    program = "push 2\npush 3\nadd\npush 4\nmul\nreturn"
    canonical = adapter.to_text(program)
    assert canonical == "PUSH 2\nPUSH 3\nADD\nPUSH 4\nMUL\nRETURN"
    assert adapter.execute(program) == 20
    assert adapter.from_text(canonical) == [("PUSH", 2), ("PUSH", 3), ("ADD", None), ("PUSH", 4), ("MUL", None), ("RETURN", None)]
