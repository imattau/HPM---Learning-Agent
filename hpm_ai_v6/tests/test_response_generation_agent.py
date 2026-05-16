from types import SimpleNamespace

import numpy as np

from hpm_ai_v6.agents.response_generation_agent import ResponseGenerationAgent
from hpm_ai_v6.hpm_model.core.cell import Cell


class ContextualStub:
    def __init__(self, candidates):
        self._candidates = list(candidates)

    def predict_next_distribution(self, tokens):
        return list(self._candidates)


class SemanticStub:
    def _get_or_create_sent_cell(self, text):
        return None


class SemanticAnchorStub:
    def __init__(self, anchor_text):
        self.anchor_text = anchor_text
        anchor_cell = Cell(name="sent_anchor", dim=0, embedding=np.array([1.0, 0.0]))
        self.sent_cells = {anchor_text: anchor_cell}
        self.sent_text_by_name = {anchor_cell.name: anchor_text}

    def _get_or_create_sent_cell(self, text):
        if text == self.anchor_text:
            return self.sent_cells[text]
        score = 1.0 if "jane was" in text.lower() else 0.0
        return Cell(name="sent_query", dim=0, embedding=np.array([score, 1.0 - score]))


def _make_agent(reasoning_agent=None):
    contextual = ContextualStub([("cat", 0.9), ("dog", 0.8)])
    word_agent = SimpleNamespace(word_cells={}, patterns=[], get_weights=lambda: [])
    phrase_agent = SimpleNamespace(patterns=[], get_weights=lambda: [])
    semantic_agent = SemanticStub()
    return ResponseGenerationAgent(
        contextual_agent=contextual,
        word_agent=word_agent,
        phrase_agent=phrase_agent,
        semantic_agent=semantic_agent,
        tag_fn=lambda tokens: ["NOUN" for _ in tokens],
        reasoning_agent=reasoning_agent,
    )


def test_generate_uses_reasoning_bias_by_default():
    agent = _make_agent(
        reasoning_agent=SimpleNamespace(
            reason_with_trace=lambda prefix: {
                "question": prefix,
                "terms": ["alice"],
                "anchors": {},
                "chosen_path": {
                    "nodes": [{"label": "alice"}, {"label": "dog"}],
                    "steps": [],
                },
                "candidate_paths": [],
            }
        )
    )

    generated = agent.generate("Alice was", max_length=1, temperature=0.0)

    assert generated == "alice was dog"


def test_generate_without_reasoning_keeps_base_decoder_order():
    agent = _make_agent()

    generated = agent.generate("Alice was", max_length=1, temperature=0.0)

    assert generated == "alice was cat"


def test_generate_consults_reasoning_agent_once_per_call():
    calls = []

    agent = _make_agent(
        reasoning_agent=SimpleNamespace(
            reason_with_trace=lambda prefix: calls.append(prefix) or {
                "question": prefix,
                "terms": ["alice"],
                "anchors": {},
                "chosen_path": {
                    "nodes": [{"label": "alice"}, {"label": "dog"}],
                    "steps": [],
                },
                "candidate_paths": [],
            }
        )
    )

    generated = agent.generate("Alice was", max_length=3, temperature=0.0)

    assert generated.startswith("alice was dog")
    assert calls == ["Alice was"]


def test_generate_skips_dirty_reasoning_agent_instead_of_blocking():
    calls = []

    agent = _make_agent(
        reasoning_agent=SimpleNamespace(
            _dirty=True,
            reason_with_trace=lambda prefix: calls.append(prefix) or {
                "question": prefix,
                "terms": ["alice"],
                "anchors": {},
                "chosen_path": {
                    "nodes": [{"label": "alice"}, {"label": "dog"}],
                    "steps": [],
                },
                "candidate_paths": [],
            }
        )
    )

    generated = agent.generate("Alice was", max_length=1, temperature=0.0)

    assert generated == "alice was cat"
    assert calls == []


def test_generate_anchors_from_learned_sentence_suffix():
    contextual = ContextualStub([("tired", 0.9), ("cat", 0.1)])
    semantic_agent = SemanticAnchorStub(
        "jane was beginning to get very tired of sitting by her sister on the bank"
    )
    word_agent = SimpleNamespace(word_cells={}, patterns=[], get_weights=lambda: [])
    phrase_agent = SimpleNamespace(patterns=[], get_weights=lambda: [])
    agent = ResponseGenerationAgent(
        contextual_agent=contextual,
        word_agent=word_agent,
        phrase_agent=phrase_agent,
        semantic_agent=semantic_agent,
        tag_fn=lambda tokens: ["NOUN" for _ in tokens],
        reasoning_agent=None,
    )

    generated = agent.generate("Jane was", max_length=1, temperature=0.0)

    assert generated.startswith("jane was beginning to get very tired")


def test_generate_blocks_immediate_token_repetition():
    contextual = ContextualStub([("pool", 0.95), ("fish", 0.8)])
    word_agent = SimpleNamespace(word_cells={}, patterns=[], get_weights=lambda: [])
    phrase_agent = SimpleNamespace(patterns=[], get_weights=lambda: [])
    semantic_agent = SemanticStub()
    agent = ResponseGenerationAgent(
        contextual_agent=contextual,
        word_agent=word_agent,
        phrase_agent=phrase_agent,
        semantic_agent=semantic_agent,
        tag_fn=lambda tokens: ["NOUN" for _ in tokens],
        reasoning_agent=None,
    )

    generated = agent.generate("jane was pool", max_length=2, temperature=0.0)

    assert "pool pool" not in generated
