# tests/test_wordnet_kb.py
import nltk
# Ensure nltk knows where the data is
nltk.data.path.append('/home/mattthomson/nltk_data')

from hpm_ai_v5.adapter.nlp import KnowledgeBaseLookup
from hpm_ai_v5.adapter.packet import AdapterPacket

def _make_packet(tokens):
    return AdapterPacket(raw=" ".join(tokens), context={"tokens": tokens}, states=[])

def test_wordnet_kb_returns_synonyms_for_known_word():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["flight"]))
    candidates = result.context.get("semantic_candidates", [])
    assert len(candidates) > 0

def test_wordnet_kb_caps_at_five():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["run"]))
    candidates = result.context.get("semantic_candidates", [])
    assert len(candidates) <= 5

def test_wordnet_kb_unknown_word_returns_empty():
    kb = KnowledgeBaseLookup()
    result = kb.run(_make_packet(["xyzzy123abc"]))
    candidates = result.context.get("semantic_candidates", [])
    assert candidates == []
