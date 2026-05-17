from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from hpm_ai_v6.hpm_model.core.cell import Cell

OPTION_KEYS = ("A", "B", "C", "D")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "is", "it", "its", "like", "me", "of", "on", "or", "our", "out",
    "so", "that", "the", "their", "there", "this", "to", "was", "were",
    "what", "when", "where", "why", "with", "you",
}


def normalize_quiz_text(text: str) -> str:
    """Normalize a question or answer into a stable lookup key."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def quiz_memory_name(question: str, correct_key: str) -> str:
    """Build a direct-memory key for an exact question->answer association."""
    return f"quiz_answer::{normalize_quiz_text(question)}=>{correct_key.lower()}"


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        token = raw.strip("'")
        if token and token not in _STOP_WORDS:
            tokens.append(token)
    return tokens


def _normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, Mapping):
        normalized: dict[str, str] = {}
        for key, value in options.items():
            if key is None:
                continue
            normalized[str(key).upper()] = str(value or "")
        return normalized

    if isinstance(options, Sequence) and not isinstance(options, (str, bytes, bytearray)):
        normalized = {}
        for idx, value in enumerate(options):
            if idx >= len(OPTION_KEYS):
                break
            normalized[OPTION_KEYS[idx]] = str(value or "")
        return normalized

    return {}


def _seed_embedding(token: str, dim: int = 16) -> np.ndarray:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    if values.size == 0:
        return np.zeros(dim, dtype=np.float32)
    tiled = np.resize(values / 255.0 - 0.5, dim)
    return tiled.astype(np.float32) * 0.2


def _question_sentence(
    question: str,
    options_map: Mapping[str, str],
    chosen_key: str,
    chosen_text: str,
    correct_key: str,
    correct_text: str,
    topic: Optional[str] = None,
) -> str:
    parts: list[str] = []
    if topic:
        parts.append(f"Topic: {topic}.")
    parts.append(f"Question: {question}.")
    if options_map:
        option_bits = [f"{key}) {options_map[key]}." for key in OPTION_KEYS if options_map.get(key)]
        if option_bits:
            parts.append("Options: " + " ".join(option_bits))
    if chosen_key and chosen_text:
        parts.append(f"Chosen answer: {chosen_key}) {chosen_text}.")
    if correct_key and correct_text:
        parts.append(f"Correct answer: {correct_key}) {correct_text}.")
    return " ".join(parts)


def _correction_sentence(
    question: str,
    correct_key: str,
    correct_text: str,
    chosen_key: str = "",
    chosen_text: str = "",
    topic: Optional[str] = None,
) -> str:
    parts: list[str] = []
    if topic:
        parts.append(f"Topic: {topic}.")
    parts.append(f"Question: {question}.")
    if correct_key and correct_text:
        parts.append(f"Correct answer: {correct_key}) {correct_text}.")
    if chosen_key and chosen_text and chosen_text != correct_text:
        parts.append(f"Chosen answer: {chosen_key}) {chosen_text}.")
    return " ".join(parts)


def _persist_all_patterns(reader) -> int:
    total = 0
    for agent in getattr(reader, "agents", {}).values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        for pattern in getattr(agent, "patterns", []):
            try:
                pager.enqueue_save(pattern)
                total += 1
            except Exception:
                pass
    return total


def _boost_question_answer_patterns(reader, question: str, correct_text: str, boost: float = 20.0) -> int:
    q_words = {word for word in _tokenize(question) if len(word) > 1}
    ans_words = [word for word in _tokenize(correct_text) if len(word) > 1]
    if not q_words or not ans_words:
        return 0

    total_boosted = 0
    word_agent = getattr(reader, "word_agent", None)

    for agent in getattr(reader, "agents", {}).values():
        patterns = getattr(agent, "patterns", [])
        if not patterns:
            continue
        pager = getattr(agent, "pattern_pager", None)
        for pattern in patterns:
            name = str(getattr(pattern, "name", "")).lower()
            has_ans = any(word in name for word in ans_words)
            has_q = any(word in name for word in q_words)
            if has_ans and has_q:
                try:
                    pattern.weight = float(getattr(pattern, "weight", 1.0)) * boost
                except Exception:
                    pass
                total_boosted += 1
                if pager is not None:
                    try:
                        pager.enqueue_save(pattern)
                    except Exception:
                        pass

    if word_agent is not None and getattr(word_agent, "pattern_pager", None) is not None:
        pager = word_agent.pattern_pager
        for q_word in q_words:
            for ans_word in ans_words:
                cell_name = f"w_{q_word}->{ans_word}"
                cell = Cell(
                    name=cell_name,
                    dim=1,
                    embedding=_seed_embedding(f"{q_word}->{ans_word}"),
                    weight=float(boost),
                )
                try:
                    pager.enqueue_save(cell)
                    total_boosted += 1
                except Exception:
                    pass

    return total_boosted


def _persist_quiz_answer(reader, question: str, correct_key: str, boost: float = 100.0) -> int:
    """Persist a direct question->answer memory for exact repeated quiz items."""
    if not question or not correct_key:
        return 0
    memory_name = quiz_memory_name(question, correct_key)
    cell = Cell(name=memory_name, dim=1, embedding=np.zeros(16, dtype=float), weight=float(boost))
    saved = 0
    for agent in getattr(reader, "agents", {}).values():
        pager = getattr(agent, "pattern_pager", None)
        if pager is None:
            continue
        try:
            pager.enqueue_save(cell)
            saved += 1
        except Exception:
            pass
    return saved


def learn_from_quiz_attempt(reader, payload: Mapping[str, Any]) -> dict[str, int]:
    """Reinforce quiz memory using the shared sentence/word/pager learning path."""
    question = str(payload.get("question", "") or "").strip()
    if not question:
        return {"trained": 0, "boosted": 0, "direct": 0, "saved": 0}

    topic = payload.get("topic")
    chosen_key = str(payload.get("chosen", "") or "").upper()
    correct_key = str(payload.get("correct", "") or "").upper()
    options_map = _normalize_options(payload.get("options") or payload.get("options_map") or {})
    chosen_text = str(payload.get("chosen_text", "") or "").strip()
    correct_text = str(payload.get("correct_text", "") or "").strip()

    if not chosen_text and chosen_key in options_map:
        chosen_text = options_map[chosen_key]
    if not correct_text and correct_key in options_map:
        correct_text = options_map[correct_key]

    if not correct_key and correct_text and options_map:
        for key, value in options_map.items():
            if value.strip() == correct_text:
                correct_key = key
                break

    question_sentence = _question_sentence(
        question=question,
        options_map=options_map,
        chosen_key=chosen_key,
        chosen_text=chosen_text,
        correct_key=correct_key,
        correct_text=correct_text,
        topic=str(topic).strip() if topic else None,
    )
    correction_sentence = _correction_sentence(
        question=question,
        correct_key=correct_key,
        correct_text=correct_text,
        chosen_key=chosen_key,
        chosen_text=chosen_text,
        topic=str(topic).strip() if topic else None,
    )

    trained = 0
    train_sequence = getattr(reader, "train_sequence", None)
    if callable(train_sequence):
        try:
            train_sequence([question_sentence, correction_sentence], enable_causal=False)
            trained = 1
        except Exception:
            pass

    boosted = _boost_question_answer_patterns(reader, question, correct_text, boost=20.0)
    direct = _persist_quiz_answer(reader, question, correct_key, boost=100.0)
    saved = _persist_all_patterns(reader)

    reasoning_agent = getattr(reader, "reasoning_agent", None)
    if reasoning_agent is not None and hasattr(reasoning_agent, "invalidate"):
        try:
            reasoning_agent.invalidate()
        except Exception:
            pass

    flush_all = getattr(reader, "flush_all", None)
    if callable(flush_all):
        try:
            flush_all()
        except Exception:
            pass

    return {"trained": trained, "boosted": boosted, "direct": direct, "saved": saved}


__all__ = [
    "learn_from_quiz_attempt",
    "normalize_quiz_text",
    "quiz_memory_name",
    "_boost_question_answer_patterns",
    "_persist_all_patterns",
    "_persist_quiz_answer",
]
