import pytest
from hpm_ai_v4.io.adapters import (
    CharClassAdapter,
    CodeDSLAdapter,
    EpisodeBundleAdapter,
    CurriculumAdapter,
    EnvironmentStateAdapter,
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


def test_code_dsl_adapter_canonicalizes_and_executes():
    adapter = CodeDSLAdapter()
    program = "push 2\npush 3\nadd\npush 4\nmul\nreturn"
    canonical = adapter.to_text(program)
    assert canonical == "PUSH 2\nPUSH 3\nADD\nPUSH 4\nMUL\nRETURN"
    assert adapter.execute(program) == 20
    assert adapter.from_text(canonical) == [("PUSH", 2), ("PUSH", 3), ("ADD", None), ("PUSH", 4), ("MUL", None), ("RETURN", None)]
