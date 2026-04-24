import pytest
from hpm_ai_v4.io.adapters import CharClassAdapter

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
