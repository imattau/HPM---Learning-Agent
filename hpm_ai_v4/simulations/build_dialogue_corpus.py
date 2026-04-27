"""Build a dialogue-heavy text corpus using two HPM chat agents."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hpm_ai_v4.simulations.chat_simulation import BasicChatSession, CHAT_SEED_CORPUS
from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


DEFAULT_TOPICS = [
    "text repair",
    "conversation",
    "patterns",
    "libraries",
    "memory",
    "planning",
    "code",
    "structured text",
    "validation",
    "control",
    "generalization",
    "repair",
    "chat",
    "symbols",
    "recall",
    "reasoning",
    "tool use",
    "curriculum",
    "dialogue",
    "grammar",
    "dictionary",
    "feedback",
    "compression",
]


def _load_text(path: str) -> str:
    if path == "/dev/stdin":
        return os.sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _default_topics(topics: Optional[Sequence[str]]) -> List[str]:
    if topics:
        return [t.strip() for t in topics if t.strip()]
    return list(DEFAULT_TOPICS)


def _looks_repetitive(text: str) -> bool:
    tokens = [tok.lower() for tok in text.split() if tok.strip()]
    if len(tokens) < 4:
        return True
    uniq_ratio = len(set(tokens)) / max(1, len(tokens))
    return uniq_ratio < 0.4


def _sanitize_question(text: str, topic: str, turn_idx: int) -> str:
    cleaned = text.strip()
    if not cleaned or _looks_repetitive(cleaned):
        return f"What do you think about {topic}?"
    cleaned = cleaned.replace("User:", "").replace("Assistant:", "").strip()
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    return cleaned


def _sanitize_answer(text: str, topic: str, turn_idx: int) -> str:
    cleaned = text.strip()
    if not cleaned or _looks_repetitive(cleaned):
        return f"{topic} is a useful topic to keep the exchange concrete."
    cleaned = cleaned.replace("User:", "").replace("Assistant:", "").strip()
    if cleaned.endswith("?"):
        cleaned = cleaned.rstrip("?") + "."
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _best_dialogue_candidate(
    candidates: Sequence[str],
    *,
    user_text: str,
    topic: str,
    agent: LayeredAgent,
    text_signals: TextSignalExtractor,
    context_texts: Sequence[str],
) -> str:
    best_text = ""
    best_score = -1e9
    for text in candidates:
        signal = text_signals.analyze(
            text,
            context_texts=[user_text, topic, *context_texts],
            target_text=topic,
            dictionary=agent.dictionary,
            grammar=agent.grammar,
        )
        score = signal.combined_score()
        if text.strip().endswith("?"):
            score += 0.04
        if text.strip().endswith("."):
            score += 0.02
        if len(text.split()) >= 4:
            score += 0.03
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def _load_seed_text(seed_corpus_path: str, extra_corpus_path: Optional[str] = None) -> str:
    chunks: List[str] = []
    if seed_corpus_path and os.path.exists(seed_corpus_path):
        chunks.append(_load_text(seed_corpus_path))
    if extra_corpus_path and os.path.exists(extra_corpus_path):
        chunks.append(_load_text(extra_corpus_path))
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def _warm_agent(agent: LayeredAgent, warmup_text: str) -> None:
    for ch in warmup_text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


@dataclass
class DialogueCorpusResult:
    corpus_path: str
    lines_written: int
    topics: List[str]
    question_preview: List[str]


def build_dialogue_corpus(
    output_path: str,
    *,
    topics: Optional[Sequence[str]] = None,
    turns_per_topic: int = 6,
    warmup_chars: int = 400,
    num_workers: int = 1,
    use_dict: bool = True,
    seed_corpus_path: str = CHAT_SEED_CORPUS,
    extra_corpus_path: Optional[str] = None,
    history_window: int = 3,
    response_steps: int = 32,
) -> DialogueCorpusResult:
    """Generate a dialogue-heavy corpus using a questioner/responder pair."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None

    questioner_agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
    responder_agent = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    seed_text = _load_seed_text(seed_corpus_path, extra_corpus_path)
    if seed_text:
        warm_text = seed_text[: max(1, warmup_chars)]
        _warm_agent(questioner_agent, warm_text)
        _warm_agent(responder_agent, warm_text)

    questioner = BasicChatSession(
        questioner_agent,
        history_window=history_window,
        response_steps=response_steps,
        system_prompt="You ask short, natural follow-up questions.",
        use_constraints=use_dict,
        learn_from_user=False,
        learn_from_reply=False,
    )
    responder = BasicChatSession(
        responder_agent,
        history_window=history_window,
        response_steps=response_steps,
        system_prompt="You answer concisely, naturally, and without repeating yourself.",
        use_constraints=use_dict,
        learn_from_user=False,
        learn_from_reply=False,
    )

    topic_list = _default_topics(topics)
    lines: List[str] = []
    question_preview: List[str] = []
    text_signals = TextSignalExtractor(use_spacy=False)

    for topic in topic_list:
        last_answer = f"Let's talk about {topic}."
        for turn_idx in range(max(1, int(turns_per_topic))):
            question_prompts = [
                f"Topic: {topic}\nPrevious answer: {last_answer}\nAsk one short follow-up question.",
                f"Topic: {topic}\nAsk for a concrete example or detail.",
                f"Topic: {topic}\nAsk a question that keeps the exchange moving forward.",
            ]
            question_candidates = [questioner.chat(prompt) for prompt in question_prompts]
            question = _sanitize_question(
                _best_dialogue_candidate(
                    question_candidates,
                    user_text=last_answer,
                    topic=topic,
                    agent=questioner_agent,
                    text_signals=text_signals,
                    context_texts=[last_answer],
                ),
                topic,
                turn_idx,
            )

            answer_prompts = [
                f"Topic: {topic}\nQuestion: {question}\nAnswer briefly and helpfully.",
                f"Topic: {topic}\nQuestion: {question}\nGive a direct answer with one useful detail.",
                f"Topic: {topic}\nQuestion: {question}\nRespond in one or two concise sentences.",
            ]
            answer_candidates = [responder.chat(prompt) for prompt in answer_prompts]
            answer = _sanitize_answer(
                _best_dialogue_candidate(
                    answer_candidates,
                    user_text=question,
                    topic=topic,
                    agent=responder_agent,
                    text_signals=text_signals,
                    context_texts=[last_answer, question],
                ),
                topic,
                turn_idx,
            )

            lines.append(f"User: {question}")
            lines.append(f"Assistant: {answer}")
            question_preview.append(question)
            last_answer = answer

            exchange = f"User: {question}\nAssistant: {answer}\n"
            questioner.agent.observe_text(exchange, feedback_mode="target")
            responder.agent.observe_text(exchange, feedback_mode="target")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return DialogueCorpusResult(
        corpus_path=output_path,
        lines_written=len(lines),
        topics=topic_list,
        question_preview=question_preview[:5],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a dialogue-heavy HPM corpus")
    p.add_argument("--output", required=True, help="Output dialogue corpus path")
    p.add_argument("--topic", action="append", default=[], help="Dialogue topic; may be repeated")
    p.add_argument("--turns-per-topic", type=int, default=6)
    p.add_argument("--warmup-chars", type=int, default=400)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--seed-corpus", default=CHAT_SEED_CORPUS, help="Dialogue seed corpus")
    p.add_argument("--extra-corpus", default=None, help="Additional text corpus for warmup")
    p.add_argument("--history-window", type=int, default=3)
    p.add_argument("--response-steps", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_dialogue_corpus(
        output_path=args.output,
        topics=args.topic or None,
        turns_per_topic=args.turns_per_topic,
        warmup_chars=args.warmup_chars,
        num_workers=args.workers,
        use_dict=args.dict,
        seed_corpus_path=args.seed_corpus,
        extra_corpus_path=args.extra_corpus,
        history_window=args.history_window,
        response_steps=args.response_steps,
    )
    print(f"[done] wrote {result.lines_written} dialogue lines to {result.corpus_path}")
    print(f"[done] topics: {', '.join(result.topics)}")
    if result.question_preview:
        print("[preview]")
        for q in result.question_preview:
            print(f"  - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
